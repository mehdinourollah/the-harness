"""The tool suite, as one PydanticAI FunctionToolset.

Registration is plain-function based, so tools are testable without an Agent.
"""

from __future__ import annotations

from pydantic_ai import FunctionToolset

from harness.tools import files, git, terminal, web


def build_toolset() -> FunctionToolset:
    ts: FunctionToolset = FunctionToolset()
    for fn in (
        files.view_file,
        files.write_file,
        files.patch_file,
        files.list_directory,
        files.grep_search,
        files.glob_search,
        terminal.execute_command,
        git.git_status,
        git.git_diff,
        git.git_commit,
        git.git_log,
        git.git_checkout_branch,
        web.web_search,
        web.browse_url,
    ):
        ts.add_function(fn)
    return ts


__all__ = ["build_toolset"]
