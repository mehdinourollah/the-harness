"""Tool registry.

Exposes:
- register(name, handler): bind a tool name to a ToolHandler subclass.
- schema(): merge JSON schemas for all tools (what the model sees).
- execute(name, arguments, req): validate args with pydantic, policy-check, run.

Handlers are instantiated once per call (stateless by design); expensive
setup should live in a singleton service accessed from inside `run`.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from harness.errors import ToolError
from harness.models import Request, ToolSpec
from harness.tools.base import ToolHandler

log = logging.getLogger(__name__)


class ToolRegistry:
    """Mutable registry. Tools are added at startup; never removed."""

    def __init__(self) -> None:
        self._handlers: dict[str, type[ToolHandler]] = {}

    def register(self, name: str, handler_cls: type[ToolHandler]) -> None:
        if name in self._handlers:
            raise ValueError(f"tool already registered: {name}")
        if not issubclass(handler_cls, ToolHandler):
            raise TypeError("handler must subclass ToolHandler")
        self._handlers[name] = handler_cls
        log.debug("registered tool %s", name)

    def schema(self) -> list[ToolSpec]:
        """JSON Schema for every tool, ready to inject into LLM calls."""
        specs: list[ToolSpec] = []
        for name, cls in self._handlers.items():
            try:
                spec = ToolSpec(
                    name=name,
                    description=cls.description,
                    parameters=cls().schema(),
                )
                specs.append(spec)
            except Exception as exc:  # pragma: no cover - defensive
                log.error("failed to build schema for %s: %s", name, exc)
        return specs

    async def execute(self, name: str, arguments: dict[str, Any], req: Request) -> str:
        """Validate args, enforce policy, run the tool, return stdout string."""
        handler_cls = self._handlers.get(name)
        if handler_cls is None:
            raise ToolError(f"unknown tool: {name}")

        # Instantiate args model and validate.
        try:
            args = handler_cls.Args(**arguments)
        except ValidationError as exc:
            raise ToolError(f"invalid arguments for {name}: {exc}") from exc

        handler = handler_cls()
        handler.check(args, req)  # may raise PolicyDenied
        return await handler.run(args, req)
