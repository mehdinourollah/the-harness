"""The tool suite, as one PydanticAI FunctionToolset.

Registration is plain-function based, so tools are testable without an Agent.
"""

from __future__ import annotations

from pydantic_ai import FunctionToolset

from harness.deps import AgentDeps
from harness.tools import files, git, terminal, web


def build_toolset() -> FunctionToolset[AgentDeps]:
    """The 14 tools, registered one by one so each signature is type-checked."""
    ts: FunctionToolset[AgentDeps] = FunctionToolset()
    ts.add_function(files.view_file)
    ts.add_function(files.write_file)
    ts.add_function(files.patch_file)
    ts.add_function(files.list_directory)
    ts.add_function(files.grep_search)
    ts.add_function(files.glob_search)
    ts.add_function(terminal.execute_command)
    ts.add_function(git.git_status)
    ts.add_function(git.git_diff)
    ts.add_function(git.git_commit)
    ts.add_function(git.git_log)
    ts.add_function(git.git_checkout_branch)
    ts.add_function(web.web_search)
    ts.add_function(web.browse_url)
    return ts


__all__ = ["build_toolset"]
