"""Filesystem tools: view / write / patch / list / grep / glob.

All paths resolve inside `req.workspace`; anything escaping raises ToolError.
Destructive ops go through policy.check(WRITE) before touching disk.
Read ops stay at READ so ReadOnly/Interactive policies don't nag on audit use.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
import shutil
import os
from pathlib import Path

from pydantic import BaseModel, Field

from harness.errors import ToolError
from harness.models import Action, ActionRequest, Request
from harness.tools.base import ToolHandler, workspace_guard

_DEFAULT_IGNORES = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}


def _resolve(req: Request, raw: str) -> Path:
    workspace_guard(req, raw)
    p = Path(raw)
    full = (req.workspace / p).resolve() if not p.is_absolute() else p.resolve()
    return full


class ViewFileArgs(BaseModel):
    path: str
    start_line: int = 1
    end_line: int | None = None


class ViewFileTool(ToolHandler):
    name = "view_file"
    description = "Read a file's content, optionally restricted to a line range."
    Args = ViewFileArgs

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, ViewFileArgs)
        target = _resolve(req, args.path)
        if not target.is_file():
            raise ToolError(f"not a file: {args.path}")
        start = max(1, args.start_line)
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        end = args.end_line if args.end_line is not None else len(lines)
        chunk = lines[start - 1 : end]
        numbered = "\n".join(f"{i:6d}  {line}" for i, line in enumerate(chunk, start))
        return f"--- {args.path} (lines {start}-{start + len(chunk) - 1} of {len(lines)}) ---\n{numbered}"


class WriteFileArgs(BaseModel):
    path: str
    content: str


class WriteFileTool(ToolHandler):
    name = "write_file"
    description = "Create or overwrite a file with the given content. Parents are created."
    Args = WriteFileArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        assert isinstance(args, WriteFileArgs)
        return [ActionRequest(action=Action.WRITE, summary=f"write {args.path}")]

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, WriteFileArgs)
        target = _resolve(req, args.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        return f"wrote {len(args.content)} chars to {args.path}"


class PatchFileArgs(BaseModel):
    path: str
    old_text: str
    new_text: str


class PatchFileTool(ToolHandler):
    name = "patch_file"
    description = (
        "Replace one exact occurrence of old_text with new_text. "
        "Fails unless old_text appears exactly once."
    )
    Args = PatchFileArgs

    def actions(self, args: BaseModel) -> list[ActionRequest]:
        assert isinstance(args, PatchFileArgs)
        return [ActionRequest(action=Action.WRITE, summary=f"patch {args.path}")]

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, PatchFileArgs)
        target = _resolve(req, args.path)
        if not target.is_file():
            raise ToolError(f"not a file: {args.path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(args.old_text)
        if count != 1:
            raise ToolError(
                f"old_text occurs {count} times in {args.path}; need exactly 1"
            )
        target.write_text(text.replace(args.old_text, args.new_text), encoding="utf-8")
        return f"patched {args.path}"


class ListDirArgs(BaseModel):
    path: str = "."
    recursive: bool = False
    ignore: list[str] = Field(default_factory=list)


class ListDirTool(ToolHandler):
    name = "list_directory"
    description = "List a directory. Set recursive=true for a full tree walk."
    Args = ListDirArgs

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, ListDirArgs)
        target = _resolve(req, args.path)
        if not target.is_dir():
            raise ToolError(f"not a directory: {args.path}")
        ignored = _DEFAULT_IGNORES | set(args.ignore)

        if not args.recursive:
            entries = sorted(
                (("d " if e.is_dir() else "f ") + e.name)
                for e in target.iterdir()
                if e.name not in ignored
            )
            return "\n".join(entries) or "(empty)"

        lines: list[str] = []
        for root, dirs, files in os.walk(target):
            dirs[:] = sorted(d for d in dirs if d not in ignored)
            rel = Path(root).relative_to(target)
            for d in sorted(dirs):
                lines.append(f"d {(rel / d).as_posix()}/")
            for f in sorted(files):
                lines.append(f"f {(rel / f).as_posix()}")
            if len(lines) > 4000:
                lines.append("... (truncated)")
                break
        return "\n".join(lines) or "(empty)"


class GrepArgs(BaseModel):
    pattern: str
    path: str = "."
    glob: str | None = None


class GrepTool(ToolHandler):
    name = "grep_search"
    description = (
        "Search file contents for a regex pattern. "
        "Uses ripgrep when available, otherwise falls back to Python regex."
    )
    Args = GrepArgs

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, GrepArgs)
        target = _resolve(req, args.path)
        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "--no-heading", "--line-number", "-e", args.pattern]
            if args.glob:
                cmd += ["-g", args.glob]
            cmd.append(str(target))
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, _ = await proc.communicate()
            text = out.decode(errors="replace")
        else:
            rx = re.compile(args.pattern)
            hits: list[str] = []
            roots = [target] if target.is_file() else list(target.rglob("*"))
            for f in roots:
                if not f.is_file() or f.name in _DEFAULT_IGNORES:
                    continue
                if args.glob and not fnmatch.fnmatch(f.name, args.glob):
                    continue
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for i, line in enumerate(content.splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f}:{i}:{line}")
                if len(hits) > 500:
                    hits.append("... (truncated)")
                    break
            text = "\n".join(hits)
        return (text.strip() or "(no matches)")[: req.max_output_chars]


class GlobArgs(BaseModel):
    pattern: str
    path: str = "."


class GlobTool(ToolHandler):
    name = "glob_search"
    description = "Find files by glob pattern, e.g. '**/*.py'."
    Args = GlobArgs

    async def run(self, args: BaseModel, req: Request) -> str:
        assert isinstance(args, GlobArgs)
        target = _resolve(req, args.path)
        base = target if target.is_dir() else target.parent
        matches = sorted(
            str(p.relative_to(req.workspace.resolve()))
            for p in base.glob(args.pattern)
            if ".git" not in p.parts
        )
        return "\n".join(matches) or "(no matches)"
