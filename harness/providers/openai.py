"""OpenAI-compatible provider adapter.

Supports:
- Official OpenAI API
- Local servers that mimic OpenAI API (vLLM, Ollama, etc.)
- Any OpenAI-compatible endpoint with custom base URL, headers, and model aliasing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from harness.models import (
    AssistantMessage,
    Completion,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolSpec,
    Usage,
    UserMessage,
)
from harness.providers.base import AccountingProvider, LLMProvider, RetryProvider
from harness.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMProvider):
    """Adapter for OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._extra_headers = extra_headers or {}
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._default_headers(),
            timeout=timeout_s,
            follow_redirects=True,
        )

    def _default_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(self._extra_headers)
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------- #
    # Internal helpers: request shaping and response parsing
    # ------------------------------------------------------------------- #

    def _build_payload(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        model: str | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or "",
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [self._tool_to_openai(t) for t in tools]
            payload["tool_choice"] = "auto"
        return payload

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                out.append({"role": "system", "content": m.content})
            elif isinstance(m, UserMessage):
                out.append({"role": "user", "content": m.content})
            elif isinstance(m, AssistantMessage):
                entry: dict[str, Any] = {"role": "assistant"}
                if m.content:
                    entry["content"] = m.content
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                # provider_state is opaque; we ignore it here.
                out.append(entry)
            elif isinstance(m, ToolMessage):
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": m.content,
                    }
                )
            else:
                raise ProviderError(f"Unknown message type: {type(m)}")
        return out

    @staticmethod
    def _tool_to_openai(spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Convert HTTP errors to domain exceptions."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            if status == 401 or status == 403:
                raise AuthenticationError(f"Unauthorized: {detail}") from exc
            if status == 429:
                raise RateLimitError(f"Rate limited: {detail}") from exc
            if 500 <= status < 600:
                raise TransientProviderError(f"Server error: {detail}") from exc
            raise ProviderError(f"HTTP {status}: {detail}") from exc

    # ------------------------------------------------------------------- #
    # LLMProvider interface
    # ------------------------------------------------------------------- #

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Completion:
        payload = self._build_payload(
            messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        response = await self._client.post("/chat/completions", json=payload)
        self._raise_for_status(response)
        data = response.json()

        # Normalize to our Completion model
        choice = data["choices"][0]
        message = self._parse_choice_message(choice)
        # Normalize API usage field names to our model's field names
        raw_usage = data.get("usage", {})
        usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens", 0),
            output_tokens=raw_usage.get("completion_tokens", 0),
            cache_read_tokens=raw_usage.get("cache_read_input_tokens", 0),
        )
        return Completion(
            message=message,
            usage=usage,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            self._raise_for_status(response)
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]

    # ------------------------------------------------------------------- #
    # Message parsing helpers
    # ------------------------------------------------------------------- #

    def _parse_choice_message(self, choice: dict[str, Any]) -> AssistantMessage:
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls", []):
            if tc.get("type") != "function":
                continue
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=args,
                )
            )
        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            provider_state=None,  # we could store raw msg if needed
        )


# --------------------------------------------------------------------------- #
# Factory: wrap with retry and accounting
# --------------------------------------------------------------------------- #


def make_openai_provider(
    *,
    base_url: str,
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 4,
) -> LLMProvider:
    inner = OpenAIAdapter(
        base_url=base_url,
        api_key=api_key,
        extra_headers=extra_headers,
        timeout_s=timeout_s,
    )
    inner = RetryProvider(inner, max_retries=max_retries)
    return AccountingProvider(inner)
