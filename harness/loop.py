"""Agent event loop, sub-agent delegation, and self-healing.

One generic loop drives every role. Specialization is by system prompt plus a
narrower tool set, which keeps the control flow in a single auditable place.

Self-healing: tool failures and provider errors are appended to the
conversation as error messages, so the next model turn sees the traceback and
can correct itself. That is the whole mechanism -- no retry heuristics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from harness.context import ContextStrategy, SlidingWindowStrategy, render_pruned_note
from harness.errors import ContextOverflow, PolicyDenied, ProviderError, ToolError
from harness.models import (
    AgentRole,
    AssistantMessage,
    Completion,
    Conversation,
    ExecutionPolicy,
    HarnessConfig,
    Message,
    Plan,
    Request,
    SubAgentResult,
    SystemMessage,
    Task,
    TaskStatus,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from harness.providers import LLMProvider
from harness.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# System prompts per role
# --------------------------------------------------------------------------- #

ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.PRIMARY: (
        "You are a pragmatic engineering agent. Investigate before you change "
        "anything, make the smallest correct change, and verify it with a real "
        "command before you claim success. Prefer reading files over guessing. "
        "When a tool fails, read the error and fix the cause, not the symptom."
    ),
    AgentRole.ARCHITECT: (
        "You are a software architect. Break the request into an ordered list of "
        "concrete, independently verifiable tasks. State the goal, the tasks, and "
        "the main risks. Do not write code; produce a plan. Finish by calling "
        "submit_plan."
    ),
    AgentRole.CODER: (
        "You are a precise implementer. Execute the assigned task with the "
        "smallest diff that works. Read files before editing them. Run the "
        "project's tests or a check command and fix failures. Report exactly "
        "what you changed and what you ran."
    ),
    AgentRole.REVIEWER: (
        "You are a skeptical reviewer. Inspect the change, run its tests and "
        "linters, and report concrete defects with evidence (command, output). "
        "Do not praise; do not fix. State pass/fail and the exact commands run."
    ),
}


# --------------------------------------------------------------------------- #
# Tool execution
# --------------------------------------------------------------------------- #


async def execute_tool_calls(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
    req: Request,
) -> list[ToolMessage]:
    """Run each tool call, turning failures into error ToolMessages.

    Failures never abort the turn: the model must see them to self-correct.
    """
    results: list[ToolMessage] = []
    for call in tool_calls:
        try:
            output = await registry.execute(call.name, call.arguments, req)
            results.append(
                ToolMessage(
                    tool_call_id=call.id, name=call.name, content=output, is_error=False
                )
            )
        except (ToolError, PolicyDenied) as exc:
            log.info("tool %s refused: %s", call.name, exc)
            results.append(
                ToolMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content=str(exc),
                    is_error=True,
                )
            )
        except Exception as exc:  # unexpected: still feed it back, never crash
            log.exception("tool %s raised", call.name)
            results.append(
                ToolMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content=f"{type(exc).__name__}: {exc}",
                    is_error=True,
                )
            )
    return results


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


@dataclass
class LoopOutcome:
    output: str
    usage: Usage
    turns: int
    conversation: Conversation


class Agent:
    """A single agent: provider + registry + policy + context strategy.

    Construct directly (explicit DI) or via `Agent.for_role`.
    """

    def __init__(
        self,
        *,
        role: AgentRole,
        provider: LLMProvider,
        registry: ToolRegistry,
        config: HarnessConfig,
        policy: ExecutionPolicy,
        strategy: ContextStrategy | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.role = role
        self.provider = provider
        self.registry = registry
        self.config = config
        self.policy = policy
        self.strategy = strategy or SlidingWindowStrategy()
        self.system_prompt = system_prompt or ROLE_PROMPTS[role]

    # -- helpers --------------------------------------------------------- #

    def _request(self) -> Request:
        return Request(
            workspace=self.config.workspace,
            policy=self.policy,
            timeout_s=self.config.request_timeout_s,
        )

    async def _step(
        self, conversation: Conversation, tools: list | None
    ) -> Completion | None:
        """One provider call, with context fitting. None on unrecoverable error."""
        budget = self.config.context_window - self.config.reserve_tokens
        try:
            kept, dropped = self.strategy.fit(conversation, budget=budget)
        except ContextOverflow as exc:
            log.error("context overflow: %s", exc)
            return None

        if dropped:
            # Replace the elided span with a one-line marker so the model knows.
            kept = [SystemMessage(content=render_pruned_note(len(dropped))), *kept]
        conversation.replace(kept)

        try:
            return await self.provider.complete(
                conversation.messages,
                tools=tools,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except ProviderError as exc:
            # Feed provider failures back for one more attempt; the RetryProvider
            # already handled transient cases, so reaching here means terminal.
            log.error("provider error: %s", exc)
            conversation.add(
                AssistantMessage(content=f"I hit a provider error: {exc}")
            )
            return None

    async def run(
        self,
        prompt: str,
        *,
        history: list[Message] | None = None,
        max_turns: int | None = None,
        tools: list | None = None,
    ) -> LoopOutcome:
        """Drive the loop until the model stops calling tools."""
        max_turns = max_turns or self.config.max_turns
        conversation = Conversation()
        conversation.add(SystemMessage(content=self.system_prompt))
        if history:
            conversation.extend(history)
        conversation.add(UserMessage(content=prompt))

        schema = self.registry.schema() if tools is None else tools
        usage = Usage()
        turns = 0

        while turns < max_turns:
            turns += 1
            completion = await self._step(conversation, schema)
            if completion is None:
                break

            usage = usage + completion.usage
            conversation.add(completion.message)

            if completion.message.tool_calls:
                results = await execute_tool_calls(
                    completion.message.tool_calls, self.registry, self._request()
                )
                conversation.extend(results)
                continue

            # No tool calls: this is the answer.
            return LoopOutcome(
                output=completion.message.content,
                usage=usage,
                turns=turns,
                conversation=conversation,
            )

        last = next(
            (m.content for m in reversed(conversation.messages)
             if isinstance(m, AssistantMessage) and m.content),
            "",
        )
        return LoopOutcome(output=last, usage=usage, turns=turns, conversation=conversation)


# --------------------------------------------------------------------------- #
# Coordinator / delegation
# --------------------------------------------------------------------------- #

PLAN_TOOL = Task  # alias for readability in the prompt above


class Coordinator:
    """Runs a primary agent that can delegate to architect/coder/reviewer.

    Delegation is explicit and synchronous: `delegate` builds a fresh Agent with
    its own isolated conversation and returns a structured SubAgentResult. No
    global state, no hidden sharing.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        registry: ToolRegistry,
        config: HarnessConfig,
        policy: ExecutionPolicy,
        strategy: ContextStrategy | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.config = config
        self.policy = policy
        self.strategy = strategy or SlidingWindowStrategy()

    def _agent(
        self,
        role: AgentRole,
        *,
        registry: ToolRegistry | None = None,
        prompt: str | None = None,
    ) -> Agent:
        return Agent(
            role=role,
            provider=self.provider,
            registry=registry or self.registry,
            config=self.config,
            policy=self.policy,
            strategy=self.strategy,
            system_prompt=prompt,
        )

    async def plan(self, request: str) -> tuple[Plan, Usage]:
        """Architect pass: requirement -> structured checklist."""
        architect = self._agent(AgentRole.ARCHITECT)
        outcome = await architect.run(
            f"Produce an implementation plan for this request:\n\n{request}",
            max_turns=4,
        )
        plan = Plan(
            goal=request,
            tasks=[Task(title=request, detail=outcome.output)],
        )
        return plan, outcome.usage

    async def delegate(
        self,
        role: AgentRole,
        prompt: str,
        *,
        tools: list | None = None,
        max_turns: int | None = None,
    ) -> SubAgentResult:
        """Run a sub-agent in its own context window and collect the result."""
        agent = self._agent(role)
        outcome = await agent.run(prompt, max_turns=max_turns, tools=tools)
        return SubAgentResult(
            role=role,
            summary=outcome.output,
            tasks=[
                Task(
                    title=f"{role.value} run",
                    status=TaskStatus.DONE if outcome.output else TaskStatus.FAILED,
                    owner=role,
                    result=outcome.output[:2000],
                )
            ],
            usage=outcome.usage,
        )

    async def run(
        self,
        request: str,
        *,
        max_turns: int | None = None,
        plan_first: bool = False,
    ) -> tuple[LoopOutcome, list[SubAgentResult]]:
        """Full run. Optionally plan first, then drive the primary agent."""
        sub_results: list[SubAgentResult] = []
        usage = Usage()

        history: list[Message] = []
        if plan_first:
            plan, plan_usage = await self.plan(request)
            usage = usage + plan_usage
            history.append(
                UserMessage(content=f"Implementation plan:\n{plan.render()}")
            )
            sub_results.append(
                SubAgentResult(
                    role=AgentRole.ARCHITECT,
                    summary=plan.render(),
                    usage=plan_usage,
                )
            )

        primary = self._agent(AgentRole.PRIMARY)
        outcome = await primary.run(request, history=history, max_turns=max_turns)
        outcome.usage = usage + outcome.usage
        return outcome, sub_results
