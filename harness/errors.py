"""Exception hierarchy. Everything the harness raises is one of these.

Model/transport failures are PydanticAI's to raise; only harness-owned
concerns live here.

`ToolError` and `PolicyDenied` also subclass PydanticAI's `ModelRetry`: raised
from inside a tool they are handed back to the model as a retry prompt instead
of aborting the run. That is what makes a refused write or a failing command
drive the agent's self-correction rather than crash it.
"""

from __future__ import annotations

from pydantic_ai.exceptions import ModelRetry


class HarnessError(Exception):
    """Base class for every error this package raises."""


class ConfigError(HarnessError):
    """Misconfiguration: bad policy name, tracing enabled without keys."""


class ToolError(ModelRetry, HarnessError):
    """A tool refused or could not complete its work."""


class PolicyDenied(ModelRetry, HarnessError):
    """The execution policy blocked the action."""
