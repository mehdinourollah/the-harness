"""CLI entrypoint: a REPL that renders tool calls live and traces to Langfuse.

Presentation lives in :mod:`harness.ui`; this module owns argument parsing,
the run loop, and how failures are reported.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pydantic_ai.exceptions import AgentRunError, UsageLimitExceeded
from pydantic_ai.usage import RunUsage

from harness import ui
from harness.agent import Session, StreamEvent, build_agent, run_agent
from harness.config import HarnessConfig
from harness.deps import AgentDeps
from harness.errors import HarnessError
from harness.policy import build_policy
from harness.tracing import setup_tracing
from harness.ui import render, terminal_approval

#: Harness failures plus PydanticAI's run-level failures (provider 4xx/5xx,
#: retry exhaustion, usage limits). Both are reported, never traceback-dumped.
RUN_FAILURES = (HarnessError, AgentRunError)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="harness", description="A PydanticAI agent harness."
    )
    p.add_argument("prompt", nargs="*", help="task to run; omit for a REPL")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--policy",
        choices=["readonly", "auto-approve", "interactive"],
        default="interactive",
    )
    p.add_argument("--workspace", default=".")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-turns", type=int, default=24)
    p.add_argument(
        "--no-history",
        action="store_true",
        help="REPL: each turn starts fresh (no carried context)",
    )
    p.add_argument("--trace", action="store_true", help="export spans to Langfuse")
    p.add_argument("--langfuse-host", default="http://localhost:3000")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> HarnessConfig:
    return HarnessConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_turns=args.max_turns,
        workspace=Path(args.workspace),
        trace=args.trace,
        langfuse_host=args.langfuse_host,
    )


def format_usage(usage: RunUsage) -> str:
    """One-line token/cost footer for the run that just finished."""
    parts = [
        f"{usage.requests} req",
        f"{usage.tool_calls} tool",
        f"{usage.total_tokens} tok",
    ]
    if usage.cost is not None:
        parts.append(f"${usage.cost:.4f}")
    return " · ".join(parts)


def report_failure(exc: Exception) -> None:
    """Report a failed turn without discarding the REPL session."""
    if isinstance(exc, UsageLimitExceeded):
        ui.warn(f"stopped: step budget exhausted ({exc})")
    else:
        ui.error(exc)


async def run_once(
    agent,
    deps: AgentDeps,
    prompt: str,
    *,
    max_turns: int,
    session: Session | None = None,
) -> None:
    def handle(event: StreamEvent) -> None:
        if event.kind == "final":
            # The answer already streamed as text; the final event only
            # carries usage, so close the turn instead of reprinting it.
            usage = format_usage(event.usage) if event.usage is not None else None
            ui.turn_end(usage)
        else:
            render(event)

    await run_agent(agent, prompt, deps, handle, session=session, max_turns=max_turns)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    policy = build_policy(args.policy, approve=terminal_approval)
    agent = build_agent(config)
    deps = AgentDeps(
        workspace=config.workspace,
        policy=policy,
        max_output_chars=config.max_output_chars,
        timeout_s=config.request_timeout_s,
    )

    if config.trace:
        try:
            if setup_tracing(config):
                ui.info(f"tracing → {config.langfuse_host}", stderr=True)
        except HarnessError as exc:
            ui.warn(f"tracing disabled: {exc}", stderr=True)

    if args.prompt:
        prompt = " ".join(args.prompt)
        try:
            asyncio.run(run_once(agent, deps, prompt, max_turns=config.max_turns))
        except RUN_FAILURES as exc:
            ui.error(exc, stderr=True)
            return 1
        return 0

    session = Session()
    ui.banner(config, args.policy)
    while True:
        try:
            line = ui.read_line()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            return 0
        if line in {"exit", "quit"}:
            return 0
        if not line:
            continue
        ui.user_turn(line)
        try:
            asyncio.run(
                run_once(
                    agent,
                    deps,
                    line,
                    max_turns=config.max_turns,
                    session=None if args.no_history else session,
                )
            )
        except RUN_FAILURES as exc:
            report_failure(exc)
        except KeyboardInterrupt:
            ui.warn("interrupted")


if __name__ == "__main__":
    raise SystemExit(main())
