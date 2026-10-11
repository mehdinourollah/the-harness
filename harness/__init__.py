"""Public surface of the harness."""

from harness.agent import StreamEvent, build_agent, run_agent
from harness.config import HarnessConfig
from harness.deps import AgentDeps
from harness.errors import HarnessError, PolicyDenied, ToolError
from harness.model import build_model
from harness.policy import (
    Action,
    ActionRequest,
    AutoApprovePolicy,
    Decision,
    ExecutionPolicy,
    InteractivePolicy,
    ReadOnlyPolicy,
    build_policy,
)
from harness.tools import build_toolset
from harness.tracing import setup_tracing

__all__ = [
    "Action",
    "ActionRequest",
    "AgentDeps",
    "AutoApprovePolicy",
    "Decision",
    "ExecutionPolicy",
    "HarnessConfig",
    "HarnessError",
    "InteractivePolicy",
    "PolicyDenied",
    "ReadOnlyPolicy",
    "StreamEvent",
    "ToolError",
    "build_agent",
    "build_model",
    "build_policy",
    "build_toolset",
    "run_agent",
    "setup_tracing",
]
