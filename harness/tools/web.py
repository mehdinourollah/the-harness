"""Web tools: search and page fetch.

Built on httpx (already a dependency) plus a small stdlib HTML->text reducer.
Both declare NETWORK actions so policies can gate them.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx
from pydantic import BaseModel

from harness.errors import ToolError
from harness.models import Action, ActionRequest, Request
from harness.tools.base import ToolHandler

_UA = "Mozilla/5.0 (compatible; harness/1.0)"


class _TextExtractor(HTMLParser):
    """Collapse an HTML document to readable text, dropping script/style."""

    _SKIP = {"script", "style", "noscript", "svg", "head"}
    _BLOCK = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "pre", "blockquote",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r"\n\s*\n\s*\n+", "\n\n", joined)
        return joined.strip()


class WebSearchArgs(BaseModel):
    query: str
    max_results: int = 5


class WebSearchTool(ToolHandler):
    name = "web_search"
    description = "Search the web (DuckDuckGo HTML endpoint) and return titles/URLs."
    Args = WebSearchArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        assert isinstance(args, WebSearchArgs)
        return [ActionRequest(action=Action.NETWORK, summary=f"search: {args.query[:60]}")]

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, WebSearchArgs)
        async with httpx.AsyncClient(
            headers={"User-Agent": _UA}, timeout=req.timeout_s, follow_redirects=True
        ) as client:
            try:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/", params={"q": args.query}
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise ToolError(f"search failed: {exc}") from exc

        html = resp.text
        # Cards look like: <a class="result__a" href="URL">TITLE</a>
        hits = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S
        )
        if not hits:
            return "(no results)"
        lines = []
        for url, title in hits[: args.max_results]:
            clean = re.sub(r"<[^>]+>", "", title).strip()
            lines.append(f"- {clean}\n  {url}")
        return "\n".join(lines)


class BrowseUrlArgs(BaseModel):
    url: str
    max_chars: int = 8000


class BrowseUrlTool(ToolHandler):
    name = "browse_url"
    description = "Fetch a URL and return its content reduced to plain text."
    Args = BrowseUrlArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        assert isinstance(args, BrowseUrlArgs)
        return [ActionRequest(action=Action.NETWORK, summary=f"fetch {args.url[:80]}")]

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, BrowseUrlArgs)
        if not args.url.startswith(("http://", "https://")):
            raise ToolError("url must start with http:// or https://")
        async with httpx.AsyncClient(
            headers={"User-Agent": _UA}, timeout=req.timeout_s, follow_redirects=True
        ) as client:
            try:
                resp = await client.get(args.url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise ToolError(f"fetch failed: {exc}") from exc

        ctype = resp.headers.get("content-type", "")
        body = resp.text
        if "html" in ctype or body.lstrip().startswith("<"):
            parser = _TextExtractor()
            parser.feed(body)
            body = parser.text()
        limit = min(args.max_chars, req.max_output_chars)
        return body[:limit]
