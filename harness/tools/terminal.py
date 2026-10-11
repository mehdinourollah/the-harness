"""Subprocess executor tool.

Runs an arbitrary command in a subshell, bounded by timeout and output size.
Declared as an EXECUTE action, so ReadOnlyPolicy always blocks it and
InteractivePolicy prompts for confirmation.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from harness.errors import ToolError
from harness.models import Action, ActionRequest, Request
from harness.tools.base import ToolHandler


class CommandArgs(BaseModel):
    command: str
    timeout_s: float | None = None
    cwd: str | None = None


class CommandTool(ToolHandler):
    name = "execute_command"
    description = (
        "Run a shell command asynchronously in the workspace directory. "
        "Returns stdout, stderr, and exit code. Fails if the exit code is non-zero."
    )
    Args = CommandArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        assert isinstance(args, CommandArgs)
        return [
            ActionRequest(
                action=Action.EXECUTE,
                summary=f"sh: {args.command[:80]}",
                detail=args.command,
            )
        ]

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, CommandArgs)
        timeout = args.timeout_s or req.timeout_s
        cwd = req.workspace if not args.cwd else (req.workspace / args.cwd).resolve()

        try:
            proc = await asyncio.create_subprocess_shell(
                args.command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ToolError(f"spawn failed: {exc}") from exc

        try:
            out_bytes, err_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ToolError(f"command timed out after {timeout:.1f}s: {args.command}")

        out = out_bytes.decode(errors="replace")
        err = err_bytes.decode(errors="replace")
        exit_code = proc.returncode

        report = [
            f"$ {args.command}",
            f"exit: {exit_code}",
        ]
        if out:
            report += ["--- stdout ---", out.rstrip()]
        if err:
            report += ["--- stderr ---", err.rstrip()]

        full = "\n".join(report)[: req.max_output_chars]
        if exit_code != 0:
            raise ToolError(f"command exited {exit_code}:\n{full}")
        return full
