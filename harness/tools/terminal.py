"""Shell execution tool. EXECUTE-gated, so ReadOnly always blocks it."""

from __future__ import annotations

import asyncio

from pydantic_ai import RunContext

from harness.deps import AgentDeps
from harness.errors import ToolError
from harness.policy import Action


async def execute_command(
    ctx: RunContext[AgentDeps],
    command: str,
    timeout_s: float | None = None,
    cwd: str | None = None,
) -> str:
    """Run a shell command asynchronously in the workspace directory.

    Returns stdout, stderr, and exit code. Raises if the exit code is non-zero.
    """
    deps = ctx.deps
    deps.check(Action.EXECUTE, f"sh: {command[:80]}", detail=command)
    timeout = timeout_s or deps.timeout_s
    workdir = deps.workspace if not cwd else (deps.workspace / cwd).resolve()

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ToolError(f"spawn failed: {exc}") from exc

    try:
        out_bytes, err_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(f"command timed out after {timeout:.1f}s: {command}")

    out = out_bytes.decode(errors="replace")
    err = err_bytes.decode(errors="replace")
    exit_code = proc.returncode

    report = [f"$ {command}", f"exit: {exit_code}"]
    if out:
        report += ["--- stdout ---", out.rstrip()]
    if err:
        report += ["--- stderr ---", err.rstrip()]

    full = "\n".join(report)
    if exit_code != 0:
        raise ToolError(f"command exited {exit_code}:\n{full}")
    return full
