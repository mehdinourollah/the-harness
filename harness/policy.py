"""Execution policies.

Pattern over booleans: callers pass a policy object, never a flag. Every side
effectful tool call declares the `Action`s it performs; the policy decides.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from harness.errors import PolicyDenied


class Action(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"


class Decision(StrEnum):
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
        decision = self.decide(request)
        if decision is Decision.DENY:
            raise PolicyDenied(f"{request.action.value} denied: {request.summary}")
        if decision is Decision.ASK and (
            self.approve is None or not self.approve(request)
        ):
            raise PolicyDenied(
                f"{request.action.value} not approved: {request.summary}"
            )


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


POLICIES: dict[str, type[ExecutionPolicy]] = {
    "readonly": ReadOnlyPolicy,
    "auto-approve": AutoApprovePolicy,
    "interactive": InteractivePolicy,
}


def build_policy(name: str, *, approve: ApprovalFn | None = None) -> ExecutionPolicy:
    try:
        cls = POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy: {name}") from None
    return cls(approve=approve) if cls is InteractivePolicy else cls()
