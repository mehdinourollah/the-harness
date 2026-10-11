"""Terminal presentation: styled output (rich) and REPL input (prompt_toolkit).

Input degrades to plain ``input()`` whenever stdin is not a TTY, so piped
input, tests and CI behave exactly as before. Every renderable is written
through a module-level ``Console`` whose file is resolved lazily, which keeps
output capturable (capsys) and ANSI-free when not attached to a terminal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from harness.agent import StreamEvent
from harness.config import HarnessConfig
from harness.policy import ActionRequest

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession

#: Palette — one colour per role, so turns are distinguishable at a glance.
PROMPT_STYLE = "bold cyan"
USER_STYLE = "bold cyan"
TOOL_NAME_STYLE = "bold magenta"
TOOL_ARGS_STYLE = "dim"
TOOL_RESULT_STYLE = "dim"
FOOTER_STYLE = "dim"
WARN_STYLE = "yellow"

console = Console()
err_console = Console(stderr=True)

HISTORY_PATH = Path.home() / ".the-harness" / "history"

_session: PromptSession | None = None


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
def _interactive() -> bool:
    """True only when stdin is a real terminal (not a pipe, file or test)."""
    return bool(sys.stdin and sys.stdin.isatty())


def _history_file() -> Path:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    return HISTORY_PATH


def _get_session() -> PromptSession:
    global _session
    if _session is None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        _session = PromptSession(history=FileHistory(str(_history_file())))
    return _session


def _read(plain: str, styled: Any) -> str:
    """Styled prompt when interactive, plain ``input()`` otherwise."""
    if _interactive():
        return _get_session().prompt(styled)
    return input(plain)


def read_line() -> str:
    """One REPL turn from the user."""
    return _read("❯ ", [("bold cyan", "❯ ")]).strip()


def confirm(prompt_text: str, *, default: bool = False) -> bool:
    """Yes/no approval. Empty answer applies ``default``."""
    hint = "[Y/n]" if default else "[y/N]"
    message = f"{prompt_text} {hint} "
    answer = _read(message, [("bold yellow", message)]).strip().lower()
    return default if not answer else answer in {"y", "yes"}


def terminal_approval(request: ActionRequest) -> bool:
    """ApprovalFn for the interactive policy: show what is being asked."""
    label = f"allow {request.action.value}? {request.summary}"
    if request.detail:
        label = f"{label}\n  {request.detail}"
    return confirm(label)


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def banner(config: HarnessConfig, policy: str) -> None:
    body = Text()
    body.append("harness\n", style="bold")
    body.append(f"{config.provider}/{config.model}", style="cyan")
    body.append(f"  ·  policy={policy}\n", style="dim")
    body.append(f"workspace {config.workspace}", style="dim")
    console.print(Panel(body, border_style="cyan", padding=(0, 1)))
    console.print("[dim]type 'exit' to quit[/dim]\n")


def user_turn(prompt: str) -> None:
    line = Text("❯ ", style=USER_STYLE)
    line.append(prompt, style="bold")
    console.print(line, soft_wrap=True)


def tool_call(name: str, args: str) -> None:
    console.print()
    line = Text("⏺ ", style=TOOL_NAME_STYLE)
    line.append(name, style="bold")
    if args:
        line.append(f"({args})", style=TOOL_ARGS_STYLE)
    console.print(line, soft_wrap=True)


def tool_result(body: str) -> None:
    body = body.strip()
    if not body:
        return
    first, *rest = body.splitlines()
    text = Text()
    text.append("  ⎿ ", style=TOOL_RESULT_STYLE)
    text.append(first[:400], style=TOOL_RESULT_STYLE)
    for line in rest[:12]:
        text.append("\n    ", style=TOOL_RESULT_STYLE)
        text.append(line[:400], style=TOOL_RESULT_STYLE)
    console.print(text, soft_wrap=True)


def stream_text(chunk: str) -> None:
    """Model text as it arrives — written raw to avoid interleaving artefacts."""
    sys.stdout.write(chunk)
    sys.stdout.flush()


def turn_end(usage: str | None) -> None:
    """Close a turn with a thin rule and the token/cost footer."""
    console.print()
    console.print(Rule(style="dim"))
    if usage is not None:
        console.print(Text(usage, style=FOOTER_STYLE))


def error(exc: Exception, *, stderr: bool = False) -> None:
    panel = Panel(
        Text(str(exc), style="red"),
        title="error",
        title_align="left",
        border_style="red",
        padding=(0, 1),
    )
    (err_console if stderr else console).print(panel)


def warn(message: str, *, stderr: bool = False) -> None:
    (err_console if stderr else console).print(
        Text(message, style=WARN_STYLE), soft_wrap=True
    )


def info(message: str, *, stderr: bool = False) -> None:
    (err_console if stderr else console).print(
        Text(message, style=FOOTER_STYLE), soft_wrap=True
    )


def render(event: StreamEvent) -> None:
    """Render one live event (the ``final`` event is handled by the caller)."""
    if event.kind == "tool_call":
        tool_call(event.name, event.text)
    elif event.kind == "tool_result":
        tool_result(event.text)
    elif event.kind == "text":
        stream_text(event.text)
