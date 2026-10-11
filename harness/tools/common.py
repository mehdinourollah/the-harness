"""Shared helpers for tools: path resolution and the trust boundary."""

from __future__ import annotations

from pathlib import Path

from harness.deps import AgentDeps
from harness.errors import ToolError

DEFAULT_IGNORES = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}


def resolve(deps: AgentDeps, raw: str) -> Path:
    """Resolve `raw` inside the workspace, refusing escapes.

    The single filesystem trust boundary; every file tool goes through it.
    """
    root = deps.root
    candidate = Path(raw)
    full = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if full != root and root not in full.parents:
        raise ToolError(f"path escapes workspace: {raw}")
    return full


def truncate(text: str, deps: AgentDeps) -> str:
    """Cap output at the deps limit, keeping the marker visible.

    Slicing blindly would amputate a caller's trailing "... (truncated)" note,
    so the note is re-appended after the cut.
    """
    limit = deps.max_output_chars
    if len(text) <= limit:
        return text
    note = "\n... (truncated)"
    return text[: max(0, limit - len(note))] + note
