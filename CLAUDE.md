# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run and test

Always run Python through the repo venv (Python 3.11). Do not use system `python3`.

```bash
.venv/bin/python -m pytest test_harness.py -q
.venv/bin/python -m pytest test_harness.py -q -k <substring>
.venv/bin/python -m pytest test_harness.py -q --cov=harness --cov-report=term-missing
```

`pytest.ini` already sets `asyncio_mode = auto`; plain `async def test_...` functions work.

Run the harness:

```bash
export ANTHROPIC_API_KEY=...  # only when hitting a real/proxied endpoint
.venv/bin/python -m harness.main --provider openai --model gpt-4o-mini "your task"
.venv/bin/python -m harness.main --help
```

Deps: `uv pip install -r requirements.txt --python .venv/bin/python`.

## Architecture

PydanticAI owns the agent loop, streaming, retries, provider adapters, tool schemas, and history. This repo owns the tools, policy layer, CLI rendering, and tracing config.

- `harness/agent.py`: `build_agent(config)` wires model + toolset; `run_agent()` streams events through `translate()` and returns the final answer.
- `harness/model.py`: `OpenAIChatModel`/`AnthropicModel` with custom `base_url` support. The OpenAI-compatible path is the one exercised against the local proxy.
- `harness/tools/`: 14 async tools behind one `FunctionToolset` (`build_toolset()`). Signatures plus docstrings are the tool schema.
- `harness/deps.py`: `AgentDeps` (workspace, policy, output cap, timeout) is injected per run via PydanticAI `RunContext`; no module-level state.
- `harness/policy.py`: `ReadOnlyPolicy` / `AutoApprovePolicy` / `InteractivePolicy`. Side-effecting tools call `ctx.deps.check(Action, ...)`.
- `harness/errors.py`: `ToolError` and `PolicyDenied` subclass PydanticAI's `ModelRetry`, so failures inside a run become retry prompts (self-correction), not crashes.
- `harness/main.py`: CLI + REPL with live tool-call rendering; run failures (`HarnessError`, PydanticAI `AgentRunError`) are reported, never dumped as tracebacks.
- `harness/tracing.py`: when `--trace` is on, wires the OTLP exporter to Langfuse (`{host}/api/public/otel`) and calls `Agent.instrument_all()`.

## Langfuse (local)

The self-hosted stack comes from `docker-compose.langfuse.yml`; secrets live in gitignored `.env.langfuse`. Start with `docker compose -f docker-compose.langfuse.yml --env-file .env.langfuse up -d`.

Run traced: set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`, then pass `--trace` (and `--langfuse-host` if not `http://localhost:3000`). Verify ingestion via `GET /api/public/v2/observations` with the same key pair; the legacy `/api/public/traces` endpoint is disabled in v4 events-only mode.
