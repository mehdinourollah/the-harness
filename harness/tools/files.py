"""Filesystem tools as plain async functions.

PydanticAI builds the JSON Schema from the signature + docstring. Policy is
checked through the injected deps; the schema stays lean on purpose — schemas
are tokenizer-cheap because there is one args model per tool, defined by the
function itself.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
from pathlib import Path

from pydantic_ai import RunContext

from harness.deps import AgentDeps
from harness.errors import ToolError
from harness.policy import Action
from harness.tools.common import DEFAULT_IGNORES, resolve, truncate


async def view_file(
    ctx: RunContext[AgentDeps],
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read a file's content, optionally restricted to a line range."""
    target = resolve(ctx.deps, path)
    if not target.is_file():
        raise ToolError(f"not a file: {path}")
    start = max(1, start_line)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    end = end_line if end_line is not None else len(lines)
    chunk = lines[start - 1 : end]
    numbered = "\n".join(f"{i:6d}  {line}" for i, line in enumerate(chunk, start))
    header = f"--- {path} (lines {start}-{start + len(chunk) - 1} of {len(lines)}) ---"
    return f"{header}\n{numbered}"


async def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
    """Create or overwrite a file with the given content. Parents are created."""
    ctx.deps.check(Action.WRITE, f"write {path}")
    target = resolve(ctx.deps, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


async def patch_file(
    ctx: RunContext[AgentDeps], path: str, old_text: str, new_text: str
) -> str:
    """Replace one exact occurrence of old_text with new_text.

    Fails unless old_text appears exactly once.
    """
    ctx.deps.check(Action.WRITE, f"patch {path}")
    target = resolve(ctx.deps, path)
    if not target.is_file():
        raise ToolError(f"not a file: {path}")
    text = target.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_text)
    if count != 1:
        raise ToolError(f"old_text occurs {count} times in {path}; need exactly 1")
    target.write_text(text.replace(old_text, new_text), encoding="utf-8")
    return f"patched {path}"


async def list_directory(
    ctx: RunContext[AgentDeps], path: str = ".", recursive: bool = False
) -> str:
    """List a directory. Set recursive=true for a full tree walk."""
    target = resolve(ctx.deps, path)
    if not target.is_dir():
        raise ToolError(f"not a directory: {path}")

    if not recursive:
        entries = sorted(
            (("d " if e.is_dir() else "f ") + e.name)
            for e in target.iterdir()
            if e.name not in DEFAULT_IGNORES
        )
        return "\n".join(entries) or "(empty)"

    lines: list[str] = []
    for root, dirs, files in os.walk(target):
        dirs[:] = sorted(d for d in dirs if d not in DEFAULT_IGNORES)
        rel = Path(root).relative_to(target)
        for d in sorted(dirs):
            lines.append(f"d {(rel / d).as_posix()}/")
        for f in sorted(files):
            lines.append(f"f {(rel / f).as_posix()}")
        if len(lines) > 4000:
            lines.append("... (truncated)")
            break
    return "\n".join(lines) or "(empty)"


async def grep_search(
    ctx: RunContext[AgentDeps],
    pattern: str,
    path: str = ".",
    glob: str | None = None,
) -> str:
    """Search file contents for a regex pattern.

    Uses ripgrep when available, otherwise falls back to Python regex.
    """
    target = resolve(ctx.deps, path)
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--no-heading", "--line-number", "-e", pattern]
        if glob:
            cmd += ["-g", glob]
        cmd.append(str(target))
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        text = out.decode(errors="replace")
    else:
        rx = re.compile(pattern)
        hits: list[str] = []
        roots = [target] if target.is_file() else list(target.rglob("*"))
        for f in roots:
            if not f.is_file() or f.name in DEFAULT_IGNORES:
                continue
            if glob and not fnmatch.fnmatch(f.name, glob):
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f}:{i}:{line}")
            if len(hits) > 500:
                break  # `truncate` below adds the visible marker
        text = "\n".join(hits)
    return truncate(text.strip() or "(no matches)", ctx.deps)


async def glob_search(ctx: RunContext[AgentDeps], pattern: str, path: str = ".") -> str:
    """Find files by glob pattern, e.g. '**/*.py'."""
    target = resolve(ctx.deps, path)
    base = target if target.is_dir() else target.parent
    matches = sorted(
        str(p.relative_to(ctx.deps.root))
        for p in base.glob(pattern)
        if not DEFAULT_IGNORES.intersection(p.parts)
    )
    return "\n".join(matches) or "(no matches)"
