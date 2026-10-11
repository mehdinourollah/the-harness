"""Anthropic provider adapter.

Talks to the Anthropic Messages API. Works against api.anthropic.com and any
Anthropic-compatible proxy (custom base URL / headers).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from harness.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from harness.models import (
    AssistantMessage,
    Completion,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolSpec,
    Usage,
)
from harness.providers.base import AccountingProvider, LLMProvider, RetryProvider

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAdapter(LLMProvider):
    """Adapter for the Anthropic Messages API."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._extra_headers = extra_headers or {}
        self._timeout_s = timeout_s
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if api_key:
            headers["x-api-key"] = api_key
        headers.update(self._extra_headers)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout_s,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------- #
    # Request shaping
    # ------------------------------------------------------------------- #

    def _split_system(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Anthropic takes system as a top-level param, not a message."""
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_parts.append(m.content)
            elif isinstance(m, ToolMessage):
                # Anthropic expects tool_result in a user turn.
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                                "is_error": m.is_error,
                            }
                        ],
                    }
                )
            elif m.__class__.__name__ == "UserMessage":
                converted.append({"role": "user", "content": m.content})
            elif isinstance(m, AssistantMessage):
                converted.append(
                    {"role": "assistant", "content": self._assistant_blocks(m)}
                )
        return ("\n\n".join(system_parts) if system_parts else None), converted

    @staticmethod
    def _assistant_blocks(m: AssistantMessage) -> list[dict[str, Any]]:
        # If we kept the vendor blob, replay it verbatim -- lossless round-trip.
        if m.provider_state and "content" in m.provider_state:
            return m.provider_state["content"]
        blocks: list[dict[str, Any]] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        for tc in m.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            )
        return blocks

    @staticmethod
    def _tool_to_anthropic(spec: ToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.parameters,
        }

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
        system, converted = self._split_system(messages)
        payload: dict[str, Any] = {
            "model": model or "",
            "messages": converted,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [self._tool_to_anthropic(t) for t in tools]
        return payload

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            if status in (401, 403):
                raise AuthenticationError(f"Unauthorized: {detail}") from exc
            if status == 429:
                raise RateLimitError(f"Rate limited: {detail}") from exc
            if status == 529 or 500 <= status < 600:
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
        response = await self._client.post("/messages", json=payload)
        self._raise_for_status(response)
        data = response.json()

        # Compatibility proxies sometimes answer /messages with an OpenAI-shaped
        # envelope (choices[].message, usage.prompt_tokens). Detect and normalize.
        if "choices" in data and "content" not in data:
            return self._from_openai_envelope(data)

        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        tool_calls = [
            ToolCall(id=b["id"], name=b["name"], arguments=b.get("input", {}))
            for b in blocks
            if b.get("type") == "tool_use"
        ]
        message = AssistantMessage(
            content=text,
            tool_calls=tool_calls,
            provider_state={"content": blocks},
        )
        usage = self._parse_usage(data.get("usage", {}))
        finish = self._map_stop_reason(data.get("stop_reason"))
        return Completion(message=message, usage=usage, finish_reason=finish, raw=data)

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
        async with self._client.stream("POST", "/messages", json=payload) as response:
            self._raise_for_status(response)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

    # ------------------------------------------------------------------- #
    # Parsing helpers
    # ------------------------------------------------------------------- #

    @staticmethod
    def _from_openai_envelope(data: dict[str, Any]) -> Completion:
        """Parse an OpenAI-shaped body a compatibility proxy returned instead."""
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls", []) or []:
            func = tc.get("function", {}) or {}
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=func.get("name", ""), arguments=args)
            )
        raw_usage = data.get("usage", {}) or {}
        cached = (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens", 0),
            output_tokens=raw_usage.get("completion_tokens", 0),
            cache_read_tokens=cached or raw_usage.get("cache_read_input_tokens", 0),
        )
        return Completion(
            message=AssistantMessage(
                content=msg.get("content", "") or "", tool_calls=tool_calls
            ),
            usage=usage,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    @staticmethod
    def _parse_usage(raw: dict[str, Any]) -> Usage:
        return Usage(
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cache_read_tokens=raw.get("cache_read_input_tokens", 0),
        )

    @staticmethod
    def _map_stop_reason(reason: str | None) -> str:
        return {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }.get(reason or "", "stop")


def make_anthropic_provider(
    *,
    base_url: str = "https://api.anthropic.com/v1",
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 4,
) -> LLMProvider:
    inner: LLMProvider = AnthropicAdapter(
        base_url=base_url,
        api_key=api_key,
        extra_headers=extra_headers,
        timeout_s=timeout_s,
    )
    inner = RetryProvider(inner, max_retries=max_retries)
    return AccountingProvider(inner)
