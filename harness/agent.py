"""Agent factory and event-driven runner.

`build_agent` wires model + toolset + deps. `run_agent` drives the loop and
pushes renderable events (tool calls and their outputs) to a callback as they
happen, returning the final answer.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)

from harness.config import HarnessConfig
from harness.deps import AgentDeps
from harness.model import build_model
from harness.tools import build_toolset

SYSTEM_PROMPT = (
    "You are a precise software-engineering agent working inside a workspace. "
    "Use the available tools to inspect and change code. Read a file before "
    "editing it. Prefer the smallest change that solves the task. When you "
    "finish, state plainly what you did and what was skipped."
)


def build_agent(config: HarnessConfig) -> Agent[AgentDeps, str]:
    """Create the agent described by `config`."""
    return Agent(
        build_model(config),
        deps_type=AgentDeps,
        toolsets=[build_toolset()],
        system_prompt=SYSTEM_PROMPT,
        retries=config.max_retries,
    )


@dataclass
class StreamEvent:
    """One renderable thing that happened during a run."""

    kind: str  # "text" | "tool_call" | "tool_result" | "final"
    text: str = ""
    name: str = ""
    payload: object = field(default=None, repr=False)


EventHandler = Callable[[StreamEvent], None]


def _describe_call(part: object) -> tuple[str, str]:
    name = str(getattr(part, "tool_name", "?"))
    args = getattr(part, "args", None)
    if isinstance(args, str):
        # Streamed calls deliver the arguments as a JSON string.
        try:
            args = json.loads(args) or None
        except json.JSONDecodeError:
            pass
    if isinstance(args, dict):
        shown = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:5])
    elif args:
        shown = str(args)
    else:
        shown = ""
    return name, shown[:400]


def _extract_content(event: FunctionToolResultEvent) -> str:
    # The tool's return (or a RetryPrompt for soft failures) lives on the part.
    part = getattr(event, "part", None)
    content = getattr(part, "content", None)
    if content is None:
        content = getattr(event, "content", None)
    return str(content)


def translate(event: AgentStreamEvent) -> StreamEvent | None:
    """Map a PydanticAI stream event to a renderable one (or None)."""
    if isinstance(event, FunctionToolCallEvent):
        name, shown = _describe_call(event.part)
        return StreamEvent("tool_call", text=shown, name=name)
    if isinstance(event, FunctionToolResultEvent):
        return StreamEvent(
            "tool_result", text=_extract_content(event)[:4000], payload=event
        )
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        if event.delta.content_delta:
            return StreamEvent("text", text=event.delta.content_delta)
    if isinstance(event, PartStartEvent) and event.part.part_kind == "text":
        content = getattr(event.part, "content", "")
        if content:
            return StreamEvent("text", text=content)
    return None


async def run_agent(
    agent: Agent[AgentDeps, str],
    prompt: str,
    deps: AgentDeps,
    on_event: EventHandler | None = None,
    *,
    max_turns: int = 24,
    max_tool_calls: int = 50,
) -> str:
    """Run the agent, forwarding live events to `on_event`; return the answer."""
    limits = UsageLimits(request_limit=max_turns, tool_calls_limit=max_tool_calls)

    async def handler(
        _ctx: RunContext[AgentDeps], events: AsyncIterable[AgentStreamEvent]
    ) -> None:
        if on_event is None:
            async for _event in events:
                pass
            return
        async for event in events:
            rendered = translate(event)
            if rendered is not None:
                on_event(rendered)

    result = await agent.run(
        prompt, deps=deps, usage_limits=limits, event_stream_handler=handler
    )
    if on_event is not None:
        on_event(StreamEvent("final", text=str(result.output)))
    return result.output
