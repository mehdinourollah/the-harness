"""Per-run dependencies handed to every tool through PydanticAI's RunContext.

Explicit injection, no globals: the workspace, the policy, and the output cap
travel with the run rather than living in module state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.policy import Action, ActionRequest, ExecutionPolicy


@dataclass
class AgentDeps:
    workspace: Path
    policy: ExecutionPolicy
    max_output_chars: int = 40_000
    timeout_s: float = 120.0

    def check(self, action: Action, summary: str, detail: str = "") -> None:
        """Run one policy check. Raises PolicyDenied when refused."""
        self.policy.check(ActionRequest(action=action, summary=summary, detail=detail))

    @property
    def root(self) -> Path:
        return self.workspace.resolve()
