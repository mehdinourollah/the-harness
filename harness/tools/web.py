"""Web tools: search and page fetch. Both NETWORK-gated."""

from __future__ import annotations

import re

import httpx
from pydantic_ai import RunContext

from harness.deps import AgentDeps
from harness.errors import ToolError
from harness.policy import Action

_UA = "Mozilla/5.0 (compatible; harness/1.0)"


async def web_search(
    ctx: RunContext[AgentDeps], query: str, max_results: int = 5
) -> str:
    """Search the web (DuckDuckGo HTML endpoint) and return titles and URLs."""
    ctx.deps.check(Action.NETWORK, f"search: {query[:60]}")
    async with httpx.AsyncClient(
        headers={"User-Agent": _UA}, timeout=ctx.deps.timeout_s, follow_redirects=True
    ) as client:
        try:
            resp = await client.get(
                "https://html.duckduckgo.com/html/", params={"q": query}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"search failed: {exc}") from exc

    hits = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', resp.text, re.DOTALL
    )
    if not hits:
        return "(no results)"
    lines = []
    for url, title in hits[:max_results]:
        clean = re.sub(r"<[^>]+>", "", title).strip()
        lines.append(f"- {clean}\n  {url}")
    return "\n".join(lines)


async def browse_url(
    ctx: RunContext[AgentDeps], url: str, max_chars: int = 8000
) -> str:
    """Fetch a URL and return its content reduced to plain text."""
    ctx.deps.check(Action.NETWORK, f"fetch: {url}")
    async with httpx.AsyncClient(
        headers={"User-Agent": _UA}, timeout=ctx.deps.timeout_s, follow_redirects=True
    ) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"fetch failed: {exc}") from exc

    text = resp.text
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] or "(empty page)"
