# AI Agent Harness — Architectural Blueprint

Deliverable 1 of the harness spec. Modules below are the verified source of
truth: `harness/` (18 modules, 1265 statements), `test_harness.py`
(162 tests, 100% coverage), entrypoint `python -m harness.main`.

## 1. Component diagram

```
                        ┌──────────────┐
                        │  harness/    │
                        │  main.py     │  CLI + REPL, --provider/--base-url/
                        │  (argparse)  │  --model/--policy, plan-first, delegate
                        └──────┬───────┘
                               │ builds config, provider, registry, policy
                               ▼
                        ┌──────────────┐
                        │ Coordinator  │  loop.py
                        │  primary +   │  plan() → architect pass (max 4 turns)
                        │  delegate()  │  delegate(role) → isolated sub-agent
                        └──────┬───────┘  run() → primary + optional plan_first
                               │ fresh Agent per role
                               ▼
                        ┌──────────────┐
                        │    Agent     │  loop.py — ONE loop for every role.
                        │  role/prompt │  Specialization = system prompt +
                        │  + strategy  │  narrowed tool list, not control flow.
                        └─┬─┬────┬─────┘
                          │ │    │
              ┌───────────┘ │    └───────────┐
              ▼             ▼                ▼
     ┌────────────┐ ┌─────────────┐ ┌──────────────┐
     │ LLMProvider│ │ ToolRegistry│ │ContextStrategy│
     │ (ABC)      │ │ validate →  │ │ SlidingWindow │
     │            │ │ policy→run  │ │ keep sys+goal │
     │ OpenAI     │ │             │ │ +recent, drop │
     │ Anthropic  │ │ 14 handlers │ │ middle, pin   │
     │ Retry ➜    │ │ files/term/ │ │ first user msg│
     │ Account ➜  │ │ git/web     │ │               │
     │ adapter    │ │             │ │ heuristic     │
     └────────────┘ └──────┬──────┘ │ counter (~c/4)│
                           │       └──────────────┘
                           │  every call declares
                           ▼  ActionRequest vec
                    ┌──────────────┐
                    │ExecutionPolicy│  models.py
                    │ ReadOnly —   │  check(action) per call;
                    │   deny WRITE/│  no --allow-danger flag.
                    │   EXECUTE/NET│  ReadOnly even denies commands.
                    │ AutoApprove —│
                    │   allow all  │
                    │ Interactive —│
                    │   prompt user│
                    └──────────────┘
```

## 2. Turn flow (what `Agent.run` does)

1. System prompt (per `ROLE_PROMPTS`) + history + user message → `Conversation`.
2. `strategy.fit(conversation, budget)` → `(kept, dropped)`; a dropped span
   becomes one `SystemMessage("[context pruned: N …]")` so the model knows.
3. `provider.complete(messages, tools=schema)` → `Completion`
   (assistant content + `tool_calls`, `Usage`).
4. `execute_tool_calls(...)` → `ToolMessage` per call.
   **Failure never aborts the turn**: `ToolError`/`PolicyDenied`/unexpected
   exceptions become `is_error=True` tool messages, so the next provider turn
   sees the traceback and self-corrects. That is the self-healing loop — no
   retry heuristics, just the error in context.
5. No tool calls → `LoopOutcome(output, usage, turns, conversation)`.

## 3. Design patterns (as specified)

| Pattern | Where |
|---|---|
| Adapter | `OpenAIAdapter` (official + vLLM/Ollama/Groq/DeepSeek via `base_url`, headers, aliases), `AnthropicAdapter` (official + compatibility proxies; normalizes an OpenAI-shaped body — `choices[].message`, `usage.prompt_tokens` — when a proxy answers `/messages` with one); both behind `LLMProvider` ABC |
| Policy over booleans | `ExecutionPolicy` objects (`ReadOnly`/`AutoApprove`/`Interactive`) injected into every tool call via `Request`; `ToolHandler.check` → `policy.check(action)` per declared `Action` |
| Strategy | `ContextStrategy` (`SlidingWindowStrategy` default); injectable `TokenCounter` (`heuristic_counter` default, no tokenizer dep) |
| Decorator | `RetryProvider` (exponential backoff + jitter on `RateLimitError`/`TransientProviderError`; streams delegate without retry — retry is unsound mid-stream) then `AccountingProvider` (cumulative `Usage` + call count) |
| Trust boundary | `workspace_guard` — one implementation, every filesystem tool |

## 4. Sub-agent delegation protocol (`Coordinator`)

- `plan(request)` → architect `Agent` (own `Conversation`, max 4 turns) →
  `Plan` (goal + ordered `Task` checklist).
- `delegate(role, prompt, tools?, max_turns?)` → fresh `Agent` with an
  **isolated context window**, returns structured `SubAgentResult`
  (role, summary, tasks with `TaskStatus`, usage). No global state shared.
- `run(request, plan_first?)` → optional architect pass archived as a
  `UserMessage("Implementation plan:\n…")` in history, then the primary agent.
  Primary usage is the sum of plan + acting usage.

## 5. Typed domain (no loose dicts across boundaries)

Pydantic models: `Message` subclasses (`System`/`User`/`Assistant`/`Tool`),
`ToolCall`, `Completion`, `Usage` (+ `ModelPricing.cost`), `Conversation`
(append-only log with `replace` for pruning), `Plan`/`Task`/`SubAgentResult`,
`ToolSpec` (what the model sees), `Action`/`ActionRequest` (READ/WRITE/EXECUTE/
NETWORK), `Request` (workspace + policy + timeouts), `HarnessConfig`
(CLI-overridable knobs, validated: reserve < window, turns ≥ 1).

## 6. Tool suite (14 handlers, `ToolRegistry.execute` = validate → check → run)

- **Files**: `view_file` (line ranges), `write_file` (creates parents),
  `patch_file` (exact-match, rejects ambiguous/missing), `list_directory`
  (recursive walk, >4000-line truncation), `grep_search` (`rg` if present,
  else Python regex over `rglob`, skips ignored dirs, >500-hit truncation),
  `glob_search`.
- **Terminal**: `execute_command` (`create_subprocess_shell`, timeout with
  kill+wait, `max_output_chars` truncation).
- **Git**: `git_status`/`git_diff` (staged+path flags)/`git_commit`
  (file list or `-A`)/`git_log`/`git_checkout_branch` (creates if missing);
  failures raise `ToolError`, never raw stderr.
- **Web**: `web_search` (DuckDuckGo HTML cards), `browse_url`
  (HTML→text, `max_chars` cap); both NETWORK-gated, `httpx` errors wrapped.

## 7. Verification

```
.venv/bin/python -m pytest test_harness.py
162 passed                                          # unit + integration + E2E
--cov=harness --cov=main → TOTAL 1265/1265, 100%    # every module 100%
python -m harness.main --help                       # exit 0
```

Also verified live against a local Anthropic-compatible proxy
(`--provider anthropic --base-url http://localhost:20128/v1 --policy readonly`):
plain completion returned text, and a tool-calling task drove `glob_search`
and answered in 2 turns.

E2E coverage includes scripted full-loop runs (prompt → architect plan →
delegation → tool calls → terminal verification → summary) and a
self-correction test (failing command fed back, fixed next iteration),
proven against a `ScriptedProvider`, never a live API.

Two environment notes: the venv targets Python 3.11 (3.14 lacks wheels for
the pinned deps) — run tests with `.venv/bin/python`; Pyright's
"could not resolve import" diagnostics are IDE path noise (venv not on its
path), disproven by the green suite.
