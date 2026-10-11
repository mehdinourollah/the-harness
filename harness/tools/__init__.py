"""Tool suite: filesystem, terminal, git, web."""

from harness.tools.base import ToolHandler, workspace_guard
from harness.tools.files import (
    GlobTool,
    GrepTool,
    ListDirTool,
    PatchFileTool,
    ViewFileTool,
    WriteFileTool,
)
from harness.tools.git import (
    GitCheckoutTool,
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
)
from harness.tools.registry import ToolRegistry
from harness.tools.terminal import CommandTool
from harness.tools.web import BrowseUrlTool, WebSearchTool

DEFAULT_TOOLS: dict[str, type[ToolHandler]] = {
    "view_file": ViewFileTool,
    "write_file": WriteFileTool,
    "patch_file": PatchFileTool,
    "list_directory": ListDirTool,
    "grep_search": GrepTool,
    "glob_search": GlobTool,
    "execute_command": CommandTool,
    "git_status": GitStatusTool,
    "git_diff": GitDiffTool,
    "git_commit": GitCommitTool,
    "git_log": GitLogTool,
    "git_checkout_branch": GitCheckoutTool,
    "web_search": WebSearchTool,
    "browse_url": BrowseUrlTool,
}


def build_registry(
    *, include: set[str] | None = None, exclude: set[str] | None = None
) -> ToolRegistry:
    """Create a registry with the default tools (optionally filtered)."""
    registry = ToolRegistry()
    for name, cls in DEFAULT_TOOLS.items():
        if include is not None and name not in include:
            continue
        if exclude is not None and name in exclude:
            continue
        registry.register(name, cls)
    return registry


__all__ = [
    "ToolHandler",
    "workspace_guard",
    "ToolRegistry",
    "build_registry",
    "DEFAULT_TOOLS",
    "ViewFileTool",
    "WriteFileTool",
    "PatchFileTool",
    "ListDirTool",
    "GrepTool",
    "GlobTool",
    "CommandTool",
    "GitStatusTool",
    "GitDiffTool",
    "GitCommitTool",
    "GitLogTool",
    "GitCheckoutTool",
    "WebSearchTool",
    "BrowseUrlTool",
]
