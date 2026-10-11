"""CLI entrypoint: a REPL that renders tool calls live and traces to Langfuse."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from pydantic_ai.exceptions import AgentRunError, UsageLimitExceeded
from pydantic_ai.usage import RunUsage

from harness.agent import Session, StreamEvent, build_agent, run_agent
from harness.config import HarnessConfig
from harness.deps import AgentDeps
from harness.errors import HarnessError
from harness.policy import ActionRequest, build_policy
from harness.tracing import setup_tracing

#: Harness failures plus PydanticAI's run-level failures (provider 4xx/5xx,
#: retry exhaustion, usage limits). Both are reported, never traceback-dumped.
RUN_FAILURES = (HarnessError, AgentRunError)

DIM, BOLD, RESET, CYAN, YELLOW, RED = (
    "\033[2m",
    "\033[1m",
    "\033[0m",
    "\033[36m",
    "\033[33m",
    "\033[31m",
)


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


def terminal_approval(request: ActionRequest) -> bool:
    detail = f"\n  {DIM}{request.detail}{RESET}" if request.detail else ""
    prompt = (
        f"{YELLOW}allow {request.action.value}?{RESET} {request.summary}{detail} [y/N] "
    )
    return input(prompt).strip().lower() in {"y", "yes"}


def render(event: StreamEvent) -> None:
    if event.kind == "tool_call":
        print(f"\n{BOLD}⏺ {event.name}{RESET}({event.text})")
    elif event.kind == "tool_result":
        body = event.text.strip()
        if body:
            first, *rest = body.splitlines()
            print(f"  {DIM}⎿ {first[:400]}{RESET}")
            for line in rest[:12]:
                print(f"    {DIM}{line[:400]}{RESET}")
    elif event.kind == "text":
        sys.stdout.write(event.text)
        sys.stdout.flush()


def report_failure(exc: Exception) -> None:
    if isinstance(exc, UsageLimitExceeded):
        print(f"{YELLOW}stopped: step budget exhausted ({exc}){RESET}")
    else:
        print(f"{RED}error: {exc}{RESET}")


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
            print(f"\n\n{BOLD}{event.text.strip()}{RESET}")
            if event.usage is not None:
                print(f"{DIM}{format_usage(event.usage)}{RESET}")
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
                print(f"{DIM}tracing → {config.langfuse_host}{RESET}", file=sys.stderr)
        except HarnessError as exc:
            print(f"{YELLOW}tracing disabled: {exc}{RESET}", file=sys.stderr)

    if args.prompt:
        prompt = " ".join(args.prompt)
        try:
            asyncio.run(run_once(agent, deps, prompt, max_turns=config.max_turns))
        except RUN_FAILURES as exc:
            print(f"{RED}{exc}{RESET}", file=sys.stderr)
            return 1
        return 0

    session = Session()
    print(
        f"{BOLD}harness{RESET} {DIM}— {config.provider}/{config.model} "
        f"policy={args.policy}{RESET}\ntype 'exit' to quit\n"
    )
    while True:
        try:
            line = input(f"{CYAN}›{RESET} ").strip()  # noqa: RUF001
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line in {"exit", "quit"}:
            return 0
        if not line:
            continue
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
            print(f"\n{DIM}interrupted{RESET}")


if __name__ == "__main__":
    raise SystemExit(main())
