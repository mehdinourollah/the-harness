"""Domain models.

Every value crossing a module boundary is one of these types. No loose dicts.
Wire-format conversion to each vendor's JSON lives in the adapters, never here.
"""

from __future__ import annotations

import uuid
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Self, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A model's request to run one tool."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Outcome of running one tool. `ok` is derived, never passed in."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    output: str = ""
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Vendor round-trip blob (e.g. Anthropic's content blocks). Opaque to the core.
    provider_state: dict[str, Any] | None = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


Message = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role"),
]


# --------------------------------------------------------------------------- #
# Usage / cost
# --------------------------------------------------------------------------- #


class Usage(BaseModel):
    """Token counters. `+` accumulates across turns and sub-agents."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


class ModelPricing(BaseModel):
    """USD per million tokens."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


class Completion(BaseModel):
    """One model response: normalised, vendor-neutral."""

    model_config = ConfigDict(frozen=True)

    message: AssistantMessage
    usage: Usage = Field(default_factory=Usage)
    finish_reason: Literal["stop", "tool_calls", "length", "error"] = "stop"
    raw: dict[str, Any] | None = None  # kept for debugging, never read by core


class Conversation(BaseModel):
    """Ordered message log plus cumulative accounting."""

    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    def add(self, message: Message) -> Self:
        self.messages.append(message)
        return self

    def extend(self, messages: list[Message]) -> Self:
        self.messages.extend(messages)
        return self

    def replace(self, messages: list[Message]) -> Self:
        self.messages = list(messages)
        return self


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


class ToolSpec(BaseModel):
    """A tool as advertised to the model."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


class Request(BaseModel):
    """Everything a ToolHandler needs. Explicit -- no globals, no ambient state."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    workspace: Path
    policy: ExecutionPolicy  # defined below; forward ref resolved by model_rebuild
    max_output_chars: int = 40_000
    timeout_s: float = 120.0


# --------------------------------------------------------------------------- #
# Policy / permissions
# --------------------------------------------------------------------------- #


class Action(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ActionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Action
    summary: str
    detail: str = ""


#: Approval hook for interactive policies: (ActionRequest) -> approved?
ApprovalFn = Callable[[ActionRequest], bool]


class ExecutionPolicy(BaseModel):
    """Base policy: allow reads, deny everything else. Subclass to relax."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    approve: ApprovalFn | None = None

    def decide(self, request: ActionRequest) -> Decision:
        if request.action is Action.READ:
            return Decision.ALLOW
        return Decision.DENY

    def check(self, request: ActionRequest) -> None:
        """Raise PolicyDenied unless the action may proceed."""
        from harness.errors import PolicyDenied

        decision = self.decide(request)
        if decision is Decision.DENY:
            raise PolicyDenied(f"{request.action.value} denied: {request.summary}")
        if decision is Decision.ASK:
            if self.approve is None or not self.approve(request):
                raise PolicyDenied(f"{request.action.value} not approved: {request.summary}")


class ReadOnlyPolicy(ExecutionPolicy):
    """Audit mode. Reads only."""


class AutoApprovePolicy(ExecutionPolicy):
    """Unattended / CI. Everything goes."""

    def decide(self, request: ActionRequest) -> Decision:
        return Decision.ALLOW


class InteractivePolicy(ExecutionPolicy):
    """Prompt before writes, commands and network access."""

    def decide(self, request: ActionRequest) -> Decision:
        if request.action is Action.READ:
            return Decision.ALLOW
        return Decision.ASK


Request.model_rebuild()


# --------------------------------------------------------------------------- #
# Errors fed back to the model (self-healing loop)
# --------------------------------------------------------------------------- #


class ToolFailure(BaseModel):
    """A structured failure the agent loop hands back to the model verbatim."""

    tool: str
    kind: str  # "policy" | "validation" | "runtime" | "timeout"
    message: str

    def render(self) -> str:
        return f"[{self.kind}] {self.tool}: {self.message}"


# --------------------------------------------------------------------------- #
# Sub-agent protocol
# --------------------------------------------------------------------------- #


class AgentRole(str, Enum):
    ARCHITECT = "architect"
    CODER = "coder"
    REVIEWER = "reviewer"
    PRIMARY = "primary"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str
    detail: str = ""
    status: TaskStatus = TaskStatus.PENDING
    owner: AgentRole | None = None
    result: str | None = None

    def render(self) -> str:
        mark = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.RUNNING: "[~]",
            TaskStatus.DONE: "[x]",
            TaskStatus.FAILED: "[!]",
            TaskStatus.BLOCKED: "[-]",
        }[self.status]
        return f"{mark} {self.id} {self.title} ({self.status.value})"


class Plan(BaseModel):
    """Structured output of the Architect agent."""

    goal: str
    tasks: list[Task] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    @field_validator("tasks")
    @classmethod
    def _at_least_one(cls, tasks: list[Task]) -> list[Task]:
        if not tasks:
            raise ValueError("a plan needs at least one task")
        return tasks

    def render(self) -> str:
        lines = [f"Goal: {self.goal}", *[t.render() for t in self.tasks]]
        if self.risks:
            lines += ["Risks:", *[f"  - {r}" for r in self.risks]]
        return "\n".join(lines)


class SubAgentResult(BaseModel):
    """What a delegated sub-agent returns to its coordinator."""

    role: AgentRole
    summary: str
    tasks: list[Task] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    def render(self) -> str:
        body = "\n".join(t.render() for t in self.tasks) if self.tasks else "(no tasks)"
        return f"<{self.role.value}>\n{self.summary}\n{body}\n</{self.role.value}>"


# --------------------------------------------------------------------------- #
# Run outcome
# --------------------------------------------------------------------------- #


class RunResult(BaseModel):
    status: Literal["ok", "incomplete", "failed"]
    output: str
    turns: int
    usage: Usage
    cost_usd: float = 0.0


class HarnessConfig(BaseModel):
    """Top-level knobs. Overridable from the CLI."""

    model: str = "gpt-4o-mini"
    provider: Literal["openai", "anthropic"] = "openai"
    base_url: str | None = None
    api_key: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 4096
    max_turns: int = 24
    context_window: int = 128_000
    reserve_tokens: int = 8_192  # headroom for the reply
    max_retries: int = 4
    request_timeout_s: float = 120.0
    workspace: Path = Field(default_factory=Path.cwd)

    @model_validator(mode="after")
    def _sane(self) -> Self:
        if self.reserve_tokens >= self.context_window:
            raise ValueError("reserve_tokens must be < context_window")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        return self

    @classmethod
    def with_cli_overrides(cls, base: HarnessConfig, **overrides: Any) -> HarnessConfig:
        """Apply only non-None overrides (CLI flags beat config file)."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return base.model_copy(update=clean)
