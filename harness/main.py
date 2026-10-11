"""CLI entrypoint for the AI Agent Harness.

Examples:

  # OpenAI-compatible endpoint (vLLM, Ollama, Groq, DeepSeek, ...)
  python -m harness.main --provider openai --base-url http://localhost:8000 --model my-model "add a --version flag"

  # Anthropic-compatible
  python -m harness.main --provider anthropic --api-key "$ANTHROPIC_API_KEY" "explain this repo"

  # Read-only audit mode: no writes, no commands
  python -m harness.main --provider openai --base-url "$BASE_URL" --policy readonly "review main.py"

  # Interactive: prompt before each write/execute
  python -m harness.main --provider openai --base-url "$BASE_URL" --policy interactive "fix the bug"

  # Auto-approve for CI
  python -m harness.main --provider openai --base-url "$BASE_URL" --policy auto-approve "run the tests"

With no PROMPT argument it starts a REPL.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from harness.errors import HarnessError
from harness.loop import Coordinator, LoopOutcome
from harness.models import (
    ActionRequest,
    AgentRole,
    AutoApprovePolicy,
    ExecutionPolicy,
    HarnessConfig,
    InteractivePolicy,
    ModelPricing,
    ReadOnlyPolicy,
)
from harness.providers import LLMProvider, make_anthropic_provider, make_openai_provider
from harness.tools import build_registry

DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


# --------------------------------------------------------------------------- #
# Policy construction (pattern over booleans: no --allow-danger flag)
# --------------------------------------------------------------------------- #


def _confirm(request: ActionRequest) -> bool:
    detail = f"\n{request.detail}" if request.detail else ""
    sys.stderr.write(f"\n  [{request.action.value}] {request.summary}{detail}\n  allow? [y/N] ")
    sys.stderr.flush()
    return input().strip().lower() in {"y", "yes"}


def build_policy(name: str, *, approve=_confirm) -> ExecutionPolicy:
    match name:
        case "readonly":
            return ReadOnlyPolicy()
        case "auto-approve":
            return AutoApprovePolicy()
        case "interactive":
            return InteractivePolicy(approve=approve)
        case _:
            raise ValueError(f"unknown policy: {name}")


def build_provider(config: HarnessConfig) -> LLMProvider:
    base_url = config.base_url or DEFAULT_URLS[config.provider]
    api_key = config.api_key or os.environ.get(KEY_ENV[config.provider])
    kwargs = dict(
        base_url=base_url,
        api_key=api_key,
        extra_headers=config.extra_headers or None,
        timeout_s=config.request_timeout_s,
        max_retries=config.max_retries,
    )
    if config.provider == "anthropic":
        return make_anthropic_provider(**kwargs)
    return make_openai_provider(**kwargs)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="harness",
        description="A CLI AI agent harness for any OpenAI/Anthropic-compatible endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("prompt", nargs="*", help="task for the agent; omit for a REPL")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--base-url", default=None, help="endpoint base URL (default: provider official)")
    p.add_argument("--api-key", default=None, help="API key (default: provider env var)")
    p.add_argument("--model", default=None, help="model id or alias")
    p.add_argument("--policy", choices=["readonly", "interactive", "auto-approve"], default="interactive")
    p.add_argument("--workspace", default=None, help="workspace root (default: cwd)")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--context-window", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--plan-first", action="store_true", help="run an architect pass before acting")
    p.add_argument("--delegate", choices=[r.value for r in AgentRole if r is not AgentRole.PRIMARY],
                   default=None, help="run a single sub-agent instead of the primary")
    p.add_argument("--header", action="append", default=[], metavar="K:V",
                   help="extra request header (repeatable)")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress the token/cost footer")
    return p.parse_args(argv)


def _parse_headers(pairs: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for pair in pairs:
        if ":" not in pair:
            raise SystemExit(f"--header expects K:V, got {pair!r}")
        key, _, value = pair.partition(":")
        headers[key.strip()] = value.strip()
    return headers


def build_config(args: argparse.Namespace) -> HarnessConfig:
    base = HarnessConfig()
    return HarnessConfig.with_cli_overrides(
        base,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        workspace=Path(args.workspace).resolve() if args.workspace else None,
        max_turns=args.max_turns,
        context_window=args.context_window,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        extra_headers=_parse_headers(args.header) or None,
    )


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #

PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(input_per_mtok=0.15, output_per_mtok=0.60),
}


def report(outcome: LoopOutcome, *, quiet: bool) -> None:
    print(outcome.output or "(no output)")
    if quiet:
        return
    usage = outcome.usage
    cost = PRICING.get("gpt-4o-mini", ModelPricing()).cost(usage)
    print(
        f"\n-- {outcome.turns} turns | in {usage.input_tokens} out "
        f"{usage.output_tokens} tokens | est. ${cost:.4f}",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


async def run_once(args: argparse.Namespace, config: HarnessConfig, prompt: str) -> LoopOutcome:
    provider = build_provider(config)
    registry = build_registry()
    policy = build_policy(args.policy)

    coordinator = Coordinator(
        provider=provider, registry=registry, config=config, policy=policy
    )

    if args.delegate:
        role = AgentRole(args.delegate)
        result = await coordinator.delegate(role, prompt, max_turns=config.max_turns)
        return LoopOutcome(
            output=result.render(), usage=result.usage, turns=0, conversation=None  # type: ignore[arg-type]
        )

    outcome, _subs = await coordinator.run(prompt, max_turns=config.max_turns, plan_first=args.plan_first)
    return outcome


async def repl(args: argparse.Namespace, config: HarnessConfig) -> int:
    provider = build_provider(config)
    registry = build_registry()
    policy = build_policy(args.policy)
    coordinator = Coordinator(
        provider=provider, registry=registry, config=config, policy=policy
    )
    print(
        f"harness REPL -- {config.provider} / {config.model} / policy={args.policy}\n"
        "type 'exit' to quit",
        file=sys.stderr,
    )
    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if line.strip() in {"exit", "quit"}:
            return 0
        if not line.strip():
            continue
        try:
            outcome, _ = await coordinator.run(line, max_turns=config.max_turns)
            report(outcome, quiet=args.quiet)
        except HarnessError as exc:
            print(f"error: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    prompt = " ".join(args.prompt).strip()
    try:
        if prompt:
            outcome = asyncio.run(run_once(args, config, prompt))
            report(outcome, quiet=args.quiet)
            return 0 if outcome.output else 1
        return asyncio.run(repl(args, config))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
