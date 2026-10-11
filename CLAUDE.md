# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run and test

Always run Python through the repo venv (Python 3.11). Do not use system `python3`.

```bash
.venv/bin/python -m pytest                          # whole suite
.venv/bin/python -m pytest -k <substring>           # one test
.venv/bin/python -m pytest --cov=harness --cov-report=term-missing --cov-fail-under=100
```

Test config (including `asyncio_mode = auto`, so plain `async def test_...` works)
lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

Run the harness:

```bash
export ANTHROPIC_API_KEY=...  # only when hitting a real/proxied endpoint
.venv/bin/python -m harness.main --provider openai --model gpt-4o-mini "your task"
.venv/bin/python -m harness.main --help
```

Deps: `uv pip install -e ".[dev]" --python .venv/bin/python`.

## Quality gates

The same gates run locally and in CI (`.github/workflows/ci.yml`); `check` is the
single branch-protection target.

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .   # lint
.venv/bin/pyright                                           # types (scoped to harness/)
.venv/bin/pre-commit run --all-files                        # hooks
.venv/bin/pre-commit install                                # pre-commit + pre-push
```

`pre-commit` runs ruff/codespell/whitespace checks on commit and the full pytest
suite with a **100% coverage floor** on push. `pyrightconfig.json` scopes
type-checking to `harness/` — the test file deliberately duck-types PydanticAI
events and is excluded.

## Architecture

PydanticAI owns the agent loop, streaming, retries, provider adapters, tool schemas, and history. This repo owns the tools, policy layer, CLI rendering, and tracing config.

- `harness/agent.py`: `build_agent(config)` wires model + toolset; `run_agent()` streams events through `translate()` and returns the final answer. `Session` carries message history and cumulative `RunUsage` across REPL turns; the `final` `StreamEvent` carries `usage` for the token/cost footer.
- `harness/model.py`: `OpenAIChatModel`/`AnthropicModel` with custom `base_url` support. The OpenAI-compatible path is the one exercised against the local proxy.
- `harness/tools/`: 14 async tools behind one `FunctionToolset[AgentDeps]` (`build_toolset()`). Signatures plus docstrings are the tool schema; each tool is registered with an explicit `ts.add_function(...)` line so it type-checks individually.
- `harness/deps.py`: `AgentDeps` (workspace, policy, output cap, timeout) is injected per run via PydanticAI `RunContext`; no module-level state.
- `harness/policy.py`: `ReadOnlyPolicy` / `AutoApprovePolicy` / `InteractivePolicy` (`Action`/`Decision` are `StrEnum`). Side-effecting tools call `ctx.deps.check(Action, ...)`.
- `harness/errors.py`: `ToolError` and `PolicyDenied` subclass PydanticAI's `ModelRetry`, so failures inside a run become retry prompts (self-correction), not crashes.
- `harness/main.py`: CLI + REPL with live tool-call rendering; run failures (`HarnessError`, PydanticAI `AgentRunError`) are reported, never dumped as tracebacks.
- `harness/ui.py`: all presentation — rich output (banner panel, tool/user/footer/error styles) and the input seam (`read_line`/`confirm` use prompt_toolkit on a TTY, plain `input()` otherwise; history at `~/.the-harness/history`). The final answer streams live and is **not** reprinted; the `final` event only closes the turn with the usage footer.
- `harness/tracing.py`: when `--trace` is on, wires the OTLP exporter to Langfuse (`{host}/api/public/otel`) and calls `Agent.instrument_all()`.

## Langfuse (local)

The self-hosted stack comes from `docker-compose.langfuse.yml`; secrets live in gitignored `.env.langfuse` (copy `.env.example`). Start with `docker compose -f docker-compose.langfuse.yml --env-file .env.langfuse up -d`.

Run traced: set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`, then pass `--trace` (and `--langfuse-host` if not `http://localhost:3000`). Verify ingestion via `GET /api/public/v2/observations` with the same key pair; the legacy `/api/public/traces` endpoint is disabled in v4 events-only mode.
