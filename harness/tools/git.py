"""Git tools.

Thin wrappers around the `git` CLI. All operate inside the workspace.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from harness.errors import ToolError
from harness.models import Action, ActionRequest, Request
from harness.tools.base import ToolHandler


async def _git(req: Request, *args: str) -> str:
    """Run `git <args>` in the workspace, returning stdout."""
    cmd = ["git", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(req.workspace.resolve()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    text = out.decode(errors="replace")
    if proc.returncode != 0:
        raise ToolError(f"git {' '.join(args)} failed: {err.decode(errors='replace')}")
    return text.strip()


class GitStatusArgs(BaseModel):
    pass


class GitStatusTool(ToolHandler):
    name = "git_status"
    description = "Show working-tree status as porcelain output."
    Args = GitStatusArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        return [ActionRequest(action=Action.READ, summary="git status")]

    async def run(self, args: BaseModel, req: Request) -> str:
        return await _git(req, "status", "--porcelain=v1") or "(clean)"


class GitDiffArgs(BaseModel):
    path: str | None = None
    staged: bool = False


class GitDiffTool(ToolHandler):
    name = "git_diff"
    description = "Show the diff of the working tree or the index."
    Args = GitDiffArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        return [ActionRequest(action=Action.READ, summary="git diff")]

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, GitDiffArgs)
        cmd = ["diff", "--stat", "--no-color"]
        if args.staged:
            cmd.append("--cached")
        if args.path:
            cmd.append(args.path)
        return await _git(req, *cmd) or "(no changes)"


class GitCommitArgs(BaseModel):
    message: str
    files: list[str] = Field(default_factory=list)


class GitCommitTool(ToolHandler):
    name = "git_commit"
    description = "Stage given files (or everything if empty) and commit."
    Args = GitCommitArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        assert isinstance(args, GitCommitArgs)
        return [ActionRequest(action=Action.EXECUTE, summary=f"git commit {args.message[:60]}")]

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, GitCommitArgs)
        if args.files:
            await _git(req, "add", "--", *args.files)
        else:
            await _git(req, "add", "-A")
        result = await _git(req, "commit", "-m", args.message)
        return result


class GitLogArgs(BaseModel):
    count: int = 10


class GitLogTool(ToolHandler):
    name = "git_log"
    description = "Show recent commits in the current branch."
    Args = GitLogArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        return [ActionRequest(action=Action.READ, summary="git log")]

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, GitLogArgs)
        return await _git(req, "log", f"-{max(1, args.count)}", "--oneline") or "(no commits)"


class GitCheckoutArgs(BaseModel):
    branch: str


class GitCheckoutTool(ToolHandler):
    name = "git_checkout_branch"
    description = "Checkout a branch. Creates it if it does not exist locally."
    Args = GitCheckoutArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        assert isinstance(args, GitCheckoutArgs)
        return [ActionRequest(action=Action.EXECUTE, summary=f"git checkout {args.branch}")]

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, GitCheckoutArgs)
        try:
            return await _git(req, "checkout", args.branch)
        except ToolError:
            return await _git(req, "checkout", "-b", args.branch)