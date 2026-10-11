"""Tool abstraction.

A `ToolHandler` bundles four things that must stay in sync:
its name/description, its pydantic argument model, the JSON Schema the model
sees, and the action(s) it performs (used for policy checks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from harness.errors import ToolError
from harness.models import Action, ActionRequest, Request


class ToolHandler(ABC):
    """Base class for every tool.

    Subclasses set `name`, `description`, `Args`, and implement `run`.
    `actions` defaults to a single READ action; write/execute tools override it.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]

    def schema(self) -> dict:
        """JSON Schema for the model. Titles are noise, so we drop them."""
        return self.Args.model_json_schema()

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        """What permissions this invocation needs. Read tools get this default."""
        return [ActionRequest(action=Action.READ, summary=f"{self.name}")]

    @abstractmethod
    async def run(self, args: BaseModel, req: Request) -> str:
        """Do the work. Return text. Raise ToolError on failure."""

    def check(self, args: BaseModel, req: Request) -> None:
        """Enforce the policy for every action this call declares."""
        for action in self.actions(args):
            req.policy.check(action)


def workspace_guard(req: Request, path: str | Path) -> None:
    """Refuse paths escaping the workspace.

    The single trust boundary for every filesystem tool -- one implementation,
    used by all of them.
    """
    root = req.workspace.resolve()
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ToolError(f"path escapes workspace: {path}")
