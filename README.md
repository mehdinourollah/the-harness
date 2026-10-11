<div align="center">

# The Harness

**A small, policy-gated CLI coding agent built on [PydanticAI](https://github.com/pydantic/pydantic-ai).**

<a href="https://github.com/mehdinourollah/the-harness/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/mehdinourollah/the-harness/actions/workflows/ci.yml/badge.svg?event=push" alt="CI"></a>
<img src="https://img.shields.io/badge/coverage-100%25-brightgreen.svg" alt="Coverage">
<a href="https://github.com/mehdinourollah/the-harness/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mehdinourollah/the-harness.svg" alt="license"></a>
<img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="python">

Talk to your codebase. The agent reads, edits, greps, runs commands, and uses
git — behind an explicit policy object, with live tool-call rendering and
optional [Langfuse](https://langfuse.com) tracing.

</div>

## Features

- **Any OpenAI- or Anthropic-compatible endpoint** — OpenAI, vLLM, LiteLLM,
  Ollama, Groq, DeepSeek, or a local proxy. Point `--base-url` at it and go.
- **14 workspace tools** — read/write/patch files, list/grep/glob the tree,
  run shell commands, full git flow (status/diff/commit/log/checkout), web
  search and page fetch.
- **Policies, not flags** — `readonly`, `auto-approve`, `interactive`. Reads
  always pass; writes, commands, and network obey the policy. A refused action
  is fed back to the model as a retry prompt, so it self-corrects instead of
  crashing.
- **Conversation memory** — the REPL carries message history and cumulative
  token/cost usage across turns (`2 req · 3 tool · 1.2k tok · $0.004` after
  every answer). `--no-history` starts each turn fresh.
- **Live Claude-Code-style rendering** — every tool call and its output streams
  to the terminal as it happens.
- **Langfuse observability** — `--trace` exports model and tool spans over OTLP
  to a self-hosted Langfuse stack (`docker-compose.langfuse.yml` included).

PydanticAI owns the agent loop, streaming, retries, provider adapters, tool
schemas, and history. This repo owns the tools, the policy layer, the CLI, and
the tracing wiring. See [ARCHITECTURE.md](ARCHITECTURE.md) for the blueprint.

## Getting Started

**Install with one command:**

```bash
curl -LsSf https://raw.githubusercontent.com/mehdinourollah/the-harness/main/install.sh | sh
```

That drops the `harness` command into `~/.local/bin`. Or install from source:

```bash
git clone https://github.com/mehdinourollah/the-harness
cd the-harness
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**Run it:**

```bash
export ANTHROPIC_API_KEY=...   # or --api-key, or OPENAI_API_KEY
harness --provider anthropic --model claude-sonnet-5-5 "find the auth bug"
harness                        # REPL
```

Against a local proxy:

```bash
harness --provider openai --base-url http://localhost:8080/v1 \
  --model my-model --policy auto-approve "write a test for the cache"
```

Traced, to your local Langfuse:

```bash
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
harness --trace --policy readonly "audit this repo for publish risks"
```

```text
harness --provider PROVIDER --model MODEL --policy POLICY [prompt ...]
  --provider      openai | anthropic        (default: openai)
  --base-url      any compatible endpoint  (default: vendor default)
  --api-key       or set OPENAI_API_KEY / ANTHROPIC_API_KEY
  --policy        readonly | auto-approve | interactive
  --no-history    REPL: each turn starts fresh
  --trace         export spans to Langfuse
  --langfuse-host (default: http://localhost:3000)
  --workspace     confine tools to this tree (default: .)
  --max-turns     per-run step budget (default: 24)
```

## Tools

| Area | Tools |
|---|---|
| Files | `view_file` · `write_file` · `patch_file` · `list_directory` · `grep_search` · `glob_search` |
| Shell | `execute_command` |
| Git | `git_status` · `git_diff` · `git_commit` · `git_log` · `git_checkout_branch` |
| Web | `web_search` · `browse_url` |

Every side-effecting tool declares the `Action`s it performs (`read`, `write`,
`execute`, `network`); the policy decides. Adding a tool is one async function
plus one `ts.add_function(...)` line in
[`harness/tools/__init__.py`](harness/tools/__init__.py) — signatures and
docstrings become the tool schema.

## Local Langfuse

A ready-to-boot self-hosted stack, no account needed:

```bash
cp .env.example .env.langfuse   # fill in if desired; defaults boot fine
docker compose -f docker-compose.langfuse.yml --env-file .env.langfuse up -d
```

Open `http://localhost:3000`, then pass `--trace`. Ingestion can be verified
without the UI via `GET /api/public/v2/observations` (the legacy `/traces`
endpoint is disabled in Langfuse v4 events-only mode).

## Development

```bash
pytest                                  # 118 tests
pytest --cov=harness --cov-report=term-missing --cov-fail-under=100
ruff check . && ruff format --check .   # lint
pyright harness                         # types (0 errors enforced in CI)
```

Quality gates, in order of speed: pre-commit hooks (ruff, codespell, whitespace)
run on commit; the full pytest suite with a **100% coverage gate** runs on
push; GitHub Actions runs lint, pyright, and the test matrix (3.11/3.12/3.13)
behind a single `all checks passed` gate. Install the hooks with
`.venv/bin/pre-commit install`.

## Contributing

Bug reports and small, focused PRs are welcome. Keep the diff tight: PydanticAI
already handles the loop, providers, and retries, so new code belongs in tools,
policy, CLI, or tracing — and every new branch needs a test (the coverage gate
is 100%). Run the hooks before you push.

## License

[MIT](LICENSE) — do what you want, keep the notice.
