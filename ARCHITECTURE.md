# AI Agent Harness — Architectural Blueprint

A CLI coding agent built on **PydanticAI**. The framework owns the agent loop,
provider adapters, tool schemas and message history; this repository owns the
workspace tools, the execution-policy layer, the CLI, and the tracing wiring.

Verified source of truth: `harness/` (12 modules, 547 statements),
`test_harness.py` (110 tests, 100% line coverage), entrypoint
`python -m harness.main`.

## 1. Component diagram

```
                        ┌───────────────────────┐
                        │   harness/main.py     │  argparse + REPL
                        │   CLI / REPL          │  --provider/--model/--base-url/
                        └──────────┬────────────┘  --api-key/--policy/--trace
                                   │ builds HarnessConfig + policy + AgentDeps
                                   ▼
                        ┌───────────────────────┐
                        │   harness/agent.py    │  build_agent(config)
                        │  build + run + render │  run_agent(agent, prompt, deps,
                        └───┬───────────┬───────┘            on_event) → str
                            │           │
             ┌──────────────┘           └────────────────┐
             ▼                                            ▼
   ┌───────────────────┐                     ┌──────────────────────────┐
   │ harness/model.py  │                     │ harness/tools/__init__.py│
   │ build_model()     │                     │ build_toolset()          │
   │ OpenAIChatModel + │                     │ FunctionToolset of 14    │
   │ OpenAIProvider    │                     │ async tool functions     │
   │ AnthropicModel +  │                     └───┬──────────────────────┘
   │ AnthropicProvider │                         │
   └───────────────────┘                         ▼
                                    ┌────────────────────────────────────┐
                                    │ files.py  terminal.py  git.py web.py│
                                    │ every tool: ctx.deps.check(Action)  │
                                    └────────────┬───────────────────────┘
                                                 ▼
                                    ┌────────────────────────────────────┐
                                    │ deps.py   AgentDeps                │
                                    │ workspace · policy · output cap ·  │
                                    │ timeout   (injected per run)       │
                                    └────────────┬───────────────────────┘
                                                 ▼
                                    ┌────────────────────────────────────┐
                                    │ policy.py  ExecutionPolicy         │
                                    │ ReadOnly · AutoApprove · Interactive│
                                    │ + common.py  resolve() trust guard │
                                    └────────────────────────────────────┘

   harness/tracing.py ── OTLP ──▶ Langfuse  (AGENT / GENERATION / TOOL spans)
```

## 2. What PydanticAI owns vs. what we own

| Concern | Owner | Notes |
|---|---|---|
| Agent loop, retries, streaming | PydanticAI `Agent` | `event_stream_handler` gives live events |
| Provider adapters | PydanticAI models | `OpenAIChatModel`, `AnthropicModel` + providers |
| Tool schema generation | PydanticAI | inferred from tool signatures + docstrings |
| Message history | PydanticAI | no hand-rolled conversation model |
| Tools | **us** | `harness/tools/*` |
| Execution policy | **us** | `harness/policy.py` (Policy pattern) |
| Trust boundary | **us** | `common.resolve()` — every path goes through it |
| CLI / rendering | **us** | `harness/main.py`, `agent.translate()` |
| Observability | **us** | `harness/tracing.py` → OTLP → Langfuse |

The previous hand-rolled provider adapters, agent loop, context-pruning
strategy and coordinator were deleted: PydanticAI subsumes them.

## 3. Tool suite (14 tools, one `FunctionToolset`)

| Module | Tools |
|---|---|
| `tools/files.py` | `view_file`, `write_file`, `patch_file`, `list_directory`, `grep_search`, `glob_search` |
| `tools/terminal.py` | `execute_command` |
| `tools/git.py` | `git_status`, `git_diff`, `git_commit`, `git_log`, `git_checkout_branch` |
| `tools/web.py` | `web_search`, `browse_url` |

Every side-effecting tool declares its `Action` and calls
`ctx.deps.check(...)`, so one policy object governs all of them. `resolve()`
rejects any path escaping the workspace.

## 4. Execution policies (Policy pattern, not booleans)

`ReadOnlyPolicy` (audit), `AutoApprovePolicy` (CI), `InteractivePolicy`
(prompts via an injected `ApprovalFn`). A refusal raises `PolicyDenied`.

## 5. Self-correction

`ToolError` and `PolicyDenied` subclass PydanticAI's `ModelRetry`, so a failed
command or a refused write is handed back to the model as a retry prompt
instead of aborting the run — the loop self-heals.

## 6. Live rendering

`run_agent` receives a callback and pushes `StreamEvent`s translated from
PydanticAI's stream events:

```
⏺ view_file(path='a.txt')
  ⎿ --- a.txt (lines 1-1 of 1) ---
        1  hello
```

## 7. Observability

`setup_tracing(config)` points the OTLP exporter at
`{host}/api/public/otel` with Basic auth over `pk:sk`, then calls
`Agent.instrument_all()`. Langfuse receives one trace per run containing
`AGENT`, `GENERATION` and `TOOL` observations.
