"""Provider abstractions.

`LLMProvider` is the single interface the rest of the harness talks to.
Vendor wire formats, retry policies, and auth live in the adapters.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from harness.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from harness.models import Completion, Message, ToolSpec, Usage

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


class LLMProvider(ABC):
    """Abstract provider.

    Concrete implementations must be thread-safe and reusable across turns.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Completion:
        """Single non-streaming completion."""

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream the textual content chunk by chunk.

        Tool-call streaming is normalized inside `complete`; `stream` is for
        live user-facing token display.
        """
        # Yield nothing by default so adapters only need complete() if they
        # don't support text streaming.
        if False:
            yield ""


# --------------------------------------------------------------------------- #
# Decorators: retry + accounting
# --------------------------------------------------------------------------- #


class RetryProvider(LLMProvider):
    """Transparent retry wrapper with exponential backoff and jitter.

    Catches RateLimitError and TransientProviderError; re-raises everything else.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_retries: int = 4,
        base_delay_s: float = 0.5,
        max_delay_s: float = 10.0,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._base = base_delay_s
        self._max = max_delay_s

    async def complete(self, messages: list[Message], **kwargs: Any) -> Completion:
        delay = self._base
        for attempt in range(self._max_retries + 1):
            try:
                return await self._inner.complete(messages, **kwargs)
            except (RateLimitError, TransientProviderError) as err:
                if attempt >= self._max_retries:
                    raise
                logger.warning(
                    "Provider failure %s (attempt %d/%d). Sleeping %.2fs",
                    err,
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, self._max)
        raise ProviderError("Unreachable retry state")

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        # Streaming retry is unsound once chunks have been emitted.
        # Delegate directly.
        async for chunk in self._inner.stream(messages, **kwargs):
            yield chunk

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


class AccountingProvider(LLMProvider):
    """Tracks cumulative usage across all completions routed through it."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.usage = Usage()
        self.calls: int = 0

    async def complete(self, messages: list[Message], **kwargs: Any) -> Completion:
        comp = await self._inner.complete(messages, **kwargs)
        self.usage = self.usage + comp.usage
        self.calls += 1
        return comp

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        async for chunk in self._inner.stream(messages, **kwargs):
            yield chunk

    async def aclose(self) -> None:
        """Release the underlying HTTP client, wherever it is in the stack."""
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()
