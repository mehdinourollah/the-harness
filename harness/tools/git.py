"""Git tools. Thin wrappers around the `git` CLI, all inside the workspace."""

from __future__ import annotations

import asyncio

from pydantic_ai import RunContext

from harness.deps import AgentDeps
from harness.errors import ToolError
from harness.policy import Action


async def _git(deps: AgentDeps, *args: str) -> str:
    """Run git, returning stdout. Raises ToolError on a non-zero exit."""
    stdout, _ = await _git_raw(deps, *args)
    return stdout


async def _git_raw(deps: AgentDeps, *args: str) -> tuple[str, str]:
    """Run git, returning (stdout, stderr). Raises ToolError on failure."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(deps.root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    text = out.decode(errors="replace")
    if proc.returncode != 0:
        raise ToolError(f"git {' '.join(args)} failed: {err.decode(errors='replace')}")
    return text, err.decode(errors="replace")


async def git_status(ctx: RunContext[AgentDeps]) -> str:
    """Show working-tree status as porcelain output."""
    return (await _git(ctx.deps, "status", "--porcelain=v1")).strip() or "(clean)"


async def git_diff(
    ctx: RunContext[AgentDeps], path: str | None = None, staged: bool = False
) -> str:
    """Show the diff of the working tree or the index."""
    cmd = ["diff", "--stat", "--no-color"]
    if staged:
        cmd.append("--cached")
    if path:
        cmd.append(path)
    return (await _git(ctx.deps, *cmd)).strip() or "(no changes)"


async def git_commit(ctx: RunContext[AgentDeps], message: str) -> str:
    """Stage everything in the workspace and commit with `message`."""
    ctx.deps.check(Action.EXECUTE, f"git commit {message[:60]}")
    await _git(ctx.deps, "add", "-A")
    return (await _git(ctx.deps, "commit", "-m", message)).strip()


async def git_log(ctx: RunContext[AgentDeps], count: int = 10) -> str:
    """Show recent commits in the current branch."""
    return (await _git(ctx.deps, "log", f"-{max(1, count)}", "--oneline")).strip() or "(no commits)"


async def git_checkout_branch(ctx: RunContext[AgentDeps], branch: str) -> str:
    """Checkout a branch. Creates it if it does not exist locally.

    git reports checkout results on stderr, so both streams are returned.
    """
    ctx.deps.check(Action.EXECUTE, f"git checkout {branch}")
    try:
        _stdout, stderr = await _git_raw(ctx.deps, "checkout", branch)
    except ToolError:
        _stdout, stderr = await _git_raw(ctx.deps, "checkout", "-b", branch)
    return stderr.strip() or f"on {branch}"
