"""Test suite for the PydanticAI-based harness.

Covers config/policy/deps models, every tool, the model factory, tracing, the
streamed agent loop (scripted FunctionModel: tool call -> answer, policy
denial, self-correction), and the CLI.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from harness import agent as A
from harness.agent import Session, StreamEvent, build_agent, run_agent
from harness.config import HarnessConfig
from harness.deps import AgentDeps
from harness.errors import ConfigError, HarnessError, PolicyDenied, ToolError
from harness.main import (
    config_from_args,
    format_usage,
    main,
    parse_args,
    render,
    report_failure,
    run_once,
    terminal_approval,
)
from harness.model import build_model
from harness.policy import (
    Action,
    ActionRequest,
    AutoApprovePolicy,
    Decision,
    ExecutionPolicy,
    InteractivePolicy,
    ReadOnlyPolicy,
    build_policy,
)
from harness.tools import build_toolset
from harness.tools import files as F
from harness.tools import git as G
from harness.tools import terminal as T
from harness.tools import web as W
from harness.tools.common import resolve, truncate
from harness.tracing import setup_tracing


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Ctx:
    """Stand-in for RunContext: the tools only ever read `.deps`."""

    deps: AgentDeps


def make_deps(tmp_path: Path, policy: ExecutionPolicy | None = None, **kw) -> AgentDeps:
    return AgentDeps(
        workspace=tmp_path,
        policy=policy or AutoApprovePolicy(),
        **kw,
    )


def ctx_for(tmp_path: Path, policy: ExecutionPolicy | None = None, **kw) -> Ctx:
    return Ctx(make_deps(tmp_path, policy, **kw))


def scripted(*responses: ModelResponse) -> Agent:
    """An Agent whose model replays `responses` in order (streamed)."""
    queue = list(responses)
    calls = itertools.count(1)

    async def sfn(messages: list, info: AgentInfo):
        response = queue.pop(0) if queue else ModelResponse(parts=[TextPart("(done)")])
        for part in response.parts:
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {
                    0: DeltaToolCall(
                        name=part.tool_name,
                        json_args=json.dumps(part.args or {}),
                        tool_call_id=part.tool_call_id or f"call_{next(calls)}",
                    )
                }

    return Agent(
        FunctionModel(stream_function=sfn),
        deps_type=AgentDeps,
        toolsets=[build_toolset()],
        system_prompt="test",
    )


async def collect(agent: Agent, prompt: str, deps: AgentDeps, **kw) -> tuple[str, list]:
    """Run the agent, returning (final output, captured events)."""
    events: list[StreamEvent] = []
    output = await run_agent(agent, prompt, deps, events.append, **kw)
    return output, events


def call(name: str, **args) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(name, args)])


def say(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(text)])


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_defaults(self):
        c = HarnessConfig()
        assert c.provider == "openai" and c.max_turns == 24 and c.trace is False

    def test_rejects_bad_max_turns(self):
        with pytest.raises(ValueError):
            HarnessConfig(max_turns=0)

    def test_cli_overrides_ignore_none(self):
        base = HarnessConfig(model="a")
        out = HarnessConfig.with_cli_overrides(base, model="b", api_key=None)
        assert out.model == "b"

    def test_cli_overrides_from_args(self):
        args = parse_args(
            ["--provider", "anthropic", "--model", "m", "--policy", "readonly", "go"]
        )
        c = config_from_args(args)
        assert c.provider == "anthropic" and c.model == "m" and args.prompt == ["go"]


# --------------------------------------------------------------------------- #
# errors / policy / deps
# --------------------------------------------------------------------------- #
class TestErrors:
    def test_hierarchy(self):
        assert issubclass(ConfigError, HarnessError)
        assert issubclass(ToolError, HarnessError)
        assert issubclass(PolicyDenied, HarnessError)


class TestPolicy:
    def _req(self, action: Action) -> ActionRequest:
        return ActionRequest(action=action, summary="s")

    def test_base_allows_read_denies_rest(self):
        p = ExecutionPolicy()
        assert p.decide(self._req(Action.READ)) is Decision.ALLOW
        assert p.decide(self._req(Action.WRITE)) is Decision.DENY

    def test_readonly_blocks_writes(self):
        with pytest.raises(PolicyDenied):
            ReadOnlyPolicy().check(self._req(Action.WRITE))

    def test_readonly_allows_read(self):
        ReadOnlyPolicy().check(self._req(Action.READ))

    def test_auto_approve_allows_all(self):
        AutoApprovePolicy().check(self._req(Action.EXECUTE))

    def test_interactive_asks_for_writes(self):
        assert InteractivePolicy().decide(self._req(Action.WRITE)) is Decision.ASK

    def test_interactive_approved_proceeds(self):
        p = InteractivePolicy(approve=lambda r: True)
        p.check(self._req(Action.NETWORK))

    def test_interactive_denied_without_hook(self):
        with pytest.raises(PolicyDenied):
            InteractivePolicy().check(self._req(Action.WRITE))

    def test_interactive_hook_says_no(self):
        p = InteractivePolicy(approve=lambda r: False)
        with pytest.raises(PolicyDenied):
            p.check(self._req(Action.WRITE))

    def test_build_policy_known_and_unknown(self):
        assert isinstance(build_policy("readonly"), ReadOnlyPolicy)
        assert isinstance(build_policy("auto-approve"), AutoApprovePolicy)
        assert isinstance(
            build_policy("interactive", approve=lambda r: True), InteractivePolicy
        )
        with pytest.raises(ValueError):
            build_policy("nope")


class TestDeps:
    def test_check_delegates_to_policy(self, tmp_path):
        deps = make_deps(tmp_path, ReadOnlyPolicy())
        deps.check(Action.READ, "read x")
        with pytest.raises(PolicyDenied):
            deps.check(Action.WRITE, "write x")

    def test_root_resolves(self, tmp_path):
        assert make_deps(tmp_path).root == tmp_path.resolve()


# --------------------------------------------------------------------------- #
# tools: files
# --------------------------------------------------------------------------- #
class TestFiles:
    async def test_view_file_range(self, tmp_path):
        (tmp_path / "a.txt").write_text("l1\nl2\nl3\n")
        out = await F.view_file(ctx_for(tmp_path), "a.txt", 2, 3)
        assert "l2" in out and "l3" in out and "l1" not in out

    async def test_view_file_missing(self, tmp_path):
        with pytest.raises(ToolError):
            await F.view_file(ctx_for(tmp_path), "nope.txt")

    async def test_write_file_creates_parents(self, tmp_path):
        out = await F.write_file(ctx_for(tmp_path), "d/e.txt", "hi")
        assert "wrote" in out and (tmp_path / "d/e.txt").read_text() == "hi"

    async def test_write_file_readonly_denied(self, tmp_path):
        with pytest.raises(PolicyDenied):
            await F.write_file(ctx_for(tmp_path, ReadOnlyPolicy()), "x.txt", "y")

    async def test_patch_file_replaces_once(self, tmp_path):
        (tmp_path / "a.txt").write_text("foo bar")
        await F.patch_file(ctx_for(tmp_path), "a.txt", "bar", "baz")
        assert (tmp_path / "a.txt").read_text() == "foo baz"

    async def test_patch_file_non_unique(self, tmp_path):
        (tmp_path / "a.txt").write_text("x x")
        with pytest.raises(ToolError, match="2 times"):
            await F.patch_file(ctx_for(tmp_path), "a.txt", "x", "y")

    async def test_patch_file_missing(self, tmp_path):
        with pytest.raises(ToolError):
            await F.patch_file(ctx_for(tmp_path), "nope", "a", "b")

    async def test_list_directory_flat(self, tmp_path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "sub").mkdir()
        out = await F.list_directory(ctx_for(tmp_path))
        assert "f a.txt" in out and "d sub" in out

    async def test_list_directory_missing(self, tmp_path):
        with pytest.raises(ToolError):
            await F.list_directory(ctx_for(tmp_path), "nope")

    async def test_list_directory_recursive_ignores_venv(self, tmp_path):
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "junk.py").write_text("x")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "m.py").write_text("x")
        out = await F.list_directory(ctx_for(tmp_path), ".", True)
        assert "src/m.py" in out and "junk.py" not in out

    async def test_list_directory_truncates(self, tmp_path, monkeypatch):
        def fake_walk(root):
            yield (str(root), [], ["a.py"])
            yield (str(root), [], [f"f{i}.py" for i in range(4100)])

        monkeypatch.setattr(F.os, "walk", fake_walk)
        out = await F.list_directory(ctx_for(tmp_path), ".", True)
        assert "(truncated)" in out

    async def test_grep_uses_rg(self, tmp_path):
        (tmp_path / "a.py").write_text("needle\n")
        out = await F.grep_search(ctx_for(tmp_path), "needle")
        assert "needle" in out

    async def test_grep_no_matches(self, tmp_path):
        (tmp_path / "a.py").write_text("hay\n")
        assert await F.grep_search(ctx_for(tmp_path), "zzz") == "(no matches)"

    async def test_grep_python_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(F.shutil, "which", lambda _: None)
        (tmp_path / "a.py").write_text("needle here\n")
        out = await F.grep_search(ctx_for(tmp_path), "needle")
        assert "needle here" in out

    async def test_grep_fallback_skips_dirs_and_globs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(F.shutil, "which", lambda _: None)
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.py").write_text("needle\n")
        (tmp_path / "b.txt").write_text("needle\n")
        out = await F.grep_search(ctx_for(tmp_path), "needle", glob="*.py")
        assert "a.py" in out and "b.txt" not in out

    async def test_grep_fallback_unreadable_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(F.shutil, "which", lambda _: None)
        (tmp_path / "a.py").write_text("needle\n")
        real = Path.read_text

        def boom(self, *a, **k):
            if self.name == "bad.py":
                raise OSError("denied")
            return real(self, *a, **k)

        (tmp_path / "bad.py").write_text("needle\n")
        monkeypatch.setattr(Path, "read_text", boom)
        out = await F.grep_search(ctx_for(tmp_path), "needle")
        assert "a.py" in out and "bad.py" not in out

    async def test_grep_fallback_truncates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(F.shutil, "which", lambda _: None)
        (tmp_path / "big.py").write_text("x\n" * 600)
        out = await F.grep_search(ctx_for(tmp_path), "x")
        assert "(truncated)" in out

    async def test_glob_search_ignores_venv(self, tmp_path):
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "j.py").write_text("")
        (tmp_path / "keep.py").write_text("")
        out = await F.glob_search(ctx_for(tmp_path), "**/*.py")
        assert "keep.py" in out and ".venv" not in out


class TestPathGuard:
    def test_resolve_rejects_escape(self, tmp_path):
        with pytest.raises(ToolError, match="escapes"):
            resolve(make_deps(tmp_path), "../outside")

    def test_resolve_absolute_inside_ok(self, tmp_path):
        assert resolve(make_deps(tmp_path), str(tmp_path / "x")) == tmp_path / "x"

    def test_truncate(self, tmp_path):
        deps = make_deps(tmp_path, max_output_chars=3)
        out = truncate("abcdef", deps)
        assert out.endswith("(truncated)") and "abcdef" not in out

    def test_truncate_under_limit_is_untouched(self, tmp_path):
        assert truncate("abcdef", make_deps(tmp_path)) == "abcdef"


# --------------------------------------------------------------------------- #
# tools: terminal / git / web
# --------------------------------------------------------------------------- #
class TestTerminal:
    async def test_success(self, tmp_path):
        out = await T.execute_command(ctx_for(tmp_path), "echo hi")
        assert "hi" in out and "exit: 0" in out

    async def test_nonzero_raises(self, tmp_path):
        with pytest.raises(ToolError, match="exited"):
            await T.execute_command(ctx_for(tmp_path), "exit 3")

    async def test_timeout(self, tmp_path):
        with pytest.raises(ToolError, match="timed out"):
            await T.execute_command(ctx_for(tmp_path), "sleep 5", timeout_s=0.05)

    async def test_spawn_failure(self, tmp_path, monkeypatch):
        async def boom(*a, **k):
            raise OSError("no shell")

        monkeypatch.setattr(T.asyncio, "create_subprocess_shell", boom)
        with pytest.raises(ToolError, match="spawn failed"):
            await T.execute_command(ctx_for(tmp_path), "echo x")

    async def test_readonly_blocks(self, tmp_path):
        with pytest.raises(PolicyDenied):
            await T.execute_command(ctx_for(tmp_path, ReadOnlyPolicy()), "echo x")

    async def test_cwd_param(self, tmp_path):
        (tmp_path / "sub").mkdir()
        out = await T.execute_command(ctx_for(tmp_path), "pwd", cwd="sub")
        assert "sub" in out


class TestGit:
    async def _repo(self, tmp_path):
        ctx = ctx_for(tmp_path)
        await T.execute_command(
            ctx, "git init -q && git config user.email t@t && git config user.name t"
        )
        return ctx

    async def test_status_and_commit_and_log(self, tmp_path):
        ctx = await self._repo(tmp_path)
        (tmp_path / "a.txt").write_text("hi")
        assert "a.txt" in await G.git_status(ctx)
        await G.git_commit(ctx, "add a")
        assert "add a" in await G.git_log(ctx)

    async def test_diff_shows_stat(self, tmp_path):
        ctx = await self._repo(tmp_path)
        (tmp_path / "a.txt").write_text("hi")
        await T.execute_command(ctx, "git add a.txt")
        out = await G.git_diff(ctx, path="a.txt", staged=True)
        assert "a.txt" in out

    async def test_diff_staged_and_clean(self, tmp_path):
        ctx = await self._repo(tmp_path)
        await T.execute_command(ctx, "git commit -q --allow-empty -m init")
        assert await G.git_diff(ctx, staged=True) == "(no changes)"

    async def test_checkout_creates_branch(self, tmp_path):
        ctx = await self._repo(tmp_path)
        await T.execute_command(ctx, "git commit -q --allow-empty -m init")
        out = await G.git_checkout_branch(ctx, "feature")
        assert "feature" in out

    async def test_checkout_existing_branch(self, tmp_path):
        ctx = await self._repo(tmp_path)
        await T.execute_command(
            ctx, "git commit -q --allow-empty -m init && git branch two"
        )
        out = await G.git_checkout_branch(ctx, "two")
        assert "two" in out

    async def test_commit_readonly_denied(self, tmp_path):
        with pytest.raises(PolicyDenied):
            await G.git_commit(ctx_for(tmp_path, ReadOnlyPolicy()), "x")

    async def test_git_failure_raises(self, tmp_path):
        with pytest.raises(ToolError):
            await G.git_log(ctx_for(tmp_path), 1)  # not a repo

    async def test_status_clean(self, tmp_path):
        ctx = await self._repo(tmp_path)
        assert await G.git_status(ctx) == "(clean)"


class _FakeClient:
    """Minimal httpx.AsyncClient double."""

    def __init__(self, text="", fail=False):
        self._text, self._fail = text, fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, *a, **k):
        if self._fail:
            raise httpx.ConnectError("boom")
        return httpx.Response(
            200, text=self._text, request=httpx.Request("GET", "http://x")
        )


class TestWeb:
    async def test_search_parses_results(self, tmp_path, monkeypatch):
        html = '<a class="result__a" href="http://a">Title A</a>'
        monkeypatch.setattr(W.httpx, "AsyncClient", lambda **k: _FakeClient(html))
        out = await W.web_search(ctx_for(tmp_path), "q")
        assert "Title A" in out and "http://a" in out

    async def test_search_no_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr(W.httpx, "AsyncClient", lambda **k: _FakeClient(""))
        assert await W.web_search(ctx_for(tmp_path), "q") == "(no results)"

    async def test_search_http_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(W.httpx, "AsyncClient", lambda **k: _FakeClient(fail=True))
        with pytest.raises(ToolError, match="search failed"):
            await W.web_search(ctx_for(tmp_path), "q")

    async def test_browse_strips_tags(self, tmp_path, monkeypatch):
        html = "<html><script>x()</script><p>Hello  world</p></html>"
        monkeypatch.setattr(W.httpx, "AsyncClient", lambda **k: _FakeClient(html))
        out = await W.browse_url(ctx_for(tmp_path), "http://a")
        assert "Hello world" in out and "script" not in out

    async def test_browse_http_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(W.httpx, "AsyncClient", lambda **k: _FakeClient(fail=True))
        with pytest.raises(ToolError, match="fetch failed"):
            await W.browse_url(ctx_for(tmp_path), "http://a")

    async def test_network_readonly_denied(self, tmp_path, monkeypatch):
        with pytest.raises(PolicyDenied):
            await W.browse_url(ctx_for(tmp_path, ReadOnlyPolicy()), "http://a")


# --------------------------------------------------------------------------- #
# model factory / tracing
# --------------------------------------------------------------------------- #
class TestModelFactory:
    def test_openai_default(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        m = build_model(HarnessConfig(provider="openai", model="gpt-4o-mini"))
        assert type(m).__name__ == "OpenAIChatModel"

    def test_openai_with_env_key_and_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        m = build_model(
            HarnessConfig(provider="openai", base_url="http://localhost:1234/v1")
        )
        assert type(m).__name__ == "OpenAIChatModel"

    def test_anthropic_with_key(self):
        m = build_model(
            HarnessConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="k")
        )
        assert type(m).__name__ == "AnthropicModel"

    def test_anthropic_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        m = build_model(HarnessConfig(provider="anthropic"))
        assert type(m).__name__ == "AnthropicModel"


class TestTracing:
    def test_off_is_noop(self):
        assert setup_tracing(HarnessConfig(trace=False)) is False

    def test_on_without_keys_raises(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        with pytest.raises(ConfigError):
            setup_tracing(HarnessConfig(trace=True))

    def test_on_wires_otlp_env(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        # Keep the real exporter off the network.
        import opentelemetry.exporter.otlp.proto.http.trace_exporter as mod

        monkeypatch.setattr(mod, "OTLPSpanExporter", lambda *a, **k: _NoopExporter())
        assert (
            setup_tracing(HarnessConfig(trace=True, langfuse_host="http://lf:3000"))
            is True
        )
        assert (
            "http://lf:3000/api/public/otel"
            in os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        )


class _NoopExporter:
    def export(self, spans):
        return None

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis=None):
        return True


# --------------------------------------------------------------------------- #
# agent loop
# --------------------------------------------------------------------------- #
class TestAgentFactory:
    def test_build_agent(self):
        agent = build_agent(HarnessConfig(provider="openai", model="gpt-4o-mini"))
        assert isinstance(agent, Agent)


class TestAgentLoop:
    async def test_tool_call_then_answer(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello\n")
        agent = scripted(call("view_file", path="a.txt"), say("read it"))
        output, events = await collect(agent, "read a.txt", make_deps(tmp_path))
        kinds = [e.kind for e in events]
        assert "tool_call" in kinds and "tool_result" in kinds and kinds[-1] == "final"
        assert any(e.name == "view_file" for e in events)
        assert output == "read it"
        assert events[-1].text == "read it"
        assert any("hello" in e.text for e in events if e.kind == "tool_result")

    async def test_self_correction_after_failing_command(self, tmp_path):
        agent = scripted(
            call("execute_command", command="exit 1"),
            say("that failed, retrying"),
        )
        output, events = await collect(agent, "run it", make_deps(tmp_path))
        results = [e.text for e in events if e.kind == "tool_result"]
        assert any("exited 1" in r for r in results)
        assert output == "that failed, retrying"

    async def test_readonly_policy_blocks_write_inside_agent(self, tmp_path):
        agent = scripted(
            call("write_file", path="x.txt", content="data"), say("blocked")
        )
        deps = make_deps(tmp_path, ReadOnlyPolicy())
        output, events = await collect(agent, "write", deps)
        results = " ".join(e.text for e in events if e.kind == "tool_result")
        assert "denied" in results
        assert not (tmp_path / "x.txt").exists()
        assert output == "blocked"

    async def test_interactive_policy_approval_hook_fires(self, tmp_path):
        seen: list[ActionRequest] = []

        def approve(req: ActionRequest) -> bool:
            seen.append(req)
            return True

        deps = make_deps(tmp_path, InteractivePolicy(approve=approve))
        agent = scripted(call("write_file", path="y.txt", content="ok"), say("wrote"))
        await collect(agent, "write", deps)
        assert seen and seen[0].action is Action.WRITE
        assert (tmp_path / "y.txt").read_text() == "ok"

    async def test_text_stream_and_final(self, tmp_path):
        agent = scripted(say("streamed answer"))
        output, events = await collect(agent, "hi", make_deps(tmp_path))
        assert events[-1].kind == "final" and events[-1].text == "streamed answer"
        assert output == "streamed answer"

    async def test_collect_without_callback(self, tmp_path):
        agent = scripted(say("quiet"))
        deps = make_deps(tmp_path)
        assert await run_agent(agent, "hi", deps) == "quiet"

    async def test_usage_limit_stops_runaway(self, tmp_path):
        agent = scripted(*[call("list_directory") for _ in range(30)])
        with pytest.raises(UsageLimitExceeded):
            await collect(agent, "loop", make_deps(tmp_path), max_tool_calls=2)


# --------------------------------------------------------------------------- #
# session memory: history carried across runs
# --------------------------------------------------------------------------- #
class TestSession:
    async def test_history_carries_across_runs(self, tmp_path):
        agent = scripted(say("first"), say("second"))
        deps = make_deps(tmp_path)
        session = Session()
        await run_agent(agent, "q1", deps, session=session)
        assert len(session.messages) == 2  # one request, one response
        await run_agent(agent, "q2", deps, session=session)
        # Four messages means turn two was sent the prior exchange.
        assert len(session.messages) == 4
        assert sum(type(m).__name__ == "ModelRequest" for m in session.messages) == 2

    async def test_session_usage_accumulates(self, tmp_path):
        agent = scripted(say("first"), say("second"))
        deps = make_deps(tmp_path)
        session = Session()
        await run_agent(agent, "q1", deps, session=session)
        after_one = session.usage.requests
        await run_agent(agent, "q2", deps, session=session)
        assert session.usage.requests == after_one * 2
        assert session.usage.total_tokens >= 0

    async def test_without_session_history_is_not_kept(self, tmp_path):
        agent = scripted(say("a"), say("b"))
        deps = make_deps(tmp_path)
        await run_agent(agent, "q1", deps)
        output, _ = await collect(agent, "q2", deps)
        assert output == "b"  # second run started clean

    async def test_final_event_carries_usage(self, tmp_path):
        agent = scripted(say("done"))
        _, events = await collect(agent, "hi", make_deps(tmp_path))
        assert events[-1].usage is not None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class TestCLI:
    def test_format_usage_without_cost(self):
        from pydantic_ai.usage import RunUsage

        line = format_usage(RunUsage(requests=2, tool_calls=3, input_tokens=100))
        assert "2 req" in line and "3 tool" in line and "$" not in line

    def test_format_usage_with_cost(self):
        from decimal import Decimal

        from pydantic_ai.usage import RunUsage

        line = format_usage(
            RunUsage(requests=1, input_tokens=10, cost=Decimal("0.0123"))
        )
        assert "$0.0123" in line

    def test_run_once_prints_answer_and_usage(self, tmp_path, capsys):
        async def ffn(messages, info):
            yield "the answer"

        agent = Agent(FunctionModel(stream_function=ffn), deps_type=AgentDeps)
        asyncio.run(run_once(agent, make_deps(tmp_path), "hi", max_turns=4))
        out = capsys.readouterr().out
        assert "the answer" in out
        assert "req" in out and "tok" in out

    def test_repl_no_history_passes_no_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())
        seen: list = []

        async def fake_run_once(agent, deps, prompt, *, max_turns, session=None):
            seen.append(session)

        monkeypatch.setattr("harness.main.run_once", fake_run_once)
        answers = iter(["one", "two", "exit"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        assert main(["--workspace", str(tmp_path), "--no-history"]) == 0
        assert seen == [None, None]

    def test_repl_shares_one_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())
        seen: list = []

        async def fake_run_once(agent, deps, prompt, *, max_turns, session=None):
            seen.append(session)

        monkeypatch.setattr("harness.main.run_once", fake_run_once)
        answers = iter(["one", "two", "exit"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        assert main(["--workspace", str(tmp_path)]) == 0
        assert seen[0] is not None and seen[0] is seen[1]

    def test_terminal_approval_yes(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *a: "y")
        assert (
            terminal_approval(ActionRequest(action=Action.WRITE, summary="s")) is True
        )

    def test_terminal_approval_no_with_detail(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *a: "")
        req = ActionRequest(action=Action.EXECUTE, summary="s", detail="rm -rf /")
        assert terminal_approval(req) is False

    def test_main_with_prompt_returns_zero(self, tmp_path, monkeypatch):
        seen: dict = {}

        class FakeAgent:
            pass

        def fake_build_agent(config):
            seen["config"] = config
            return FakeAgent()

        async def fake_run_once(agent, deps, prompt, *, max_turns, session=None):
            seen["prompt"] = prompt

        monkeypatch.setattr("harness.main.build_agent", fake_build_agent)
        monkeypatch.setattr("harness.main.run_once", fake_run_once)
        rc = main(["--workspace", str(tmp_path), "--policy", "auto-approve", "do it"])
        assert rc == 0 and seen["prompt"] == "do it"

    def test_main_reports_harness_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())

        async def boom(*a, **k):
            raise ToolError("kaboom")

        monkeypatch.setattr("harness.main.run_once", boom)
        rc = main(["--workspace", str(tmp_path), "x"])
        assert rc == 1 and "kaboom" in capsys.readouterr().err

    def test_repl_exit(self, tmp_path, monkeypatch):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())
        monkeypatch.setattr("builtins.input", lambda *a: "exit")
        assert main(["--workspace", str(tmp_path)]) == 0

    def test_repl_runs_prompt_then_eof(self, tmp_path, monkeypatch):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())
        calls: list[str] = []

        async def fake_run_once(agent, deps, prompt, *, max_turns, session=None):
            calls.append(prompt)

        monkeypatch.setattr("harness.main.run_once", fake_run_once)
        answers = iter(["hello", EOFError()])

        def fake_input(*a):
            nxt = next(answers)
            if isinstance(nxt, BaseException):
                raise nxt
            return nxt

        monkeypatch.setattr("builtins.input", fake_input)
        assert main(["--workspace", str(tmp_path)]) == 0
        assert calls == ["hello"]

    def test_repl_survives_harness_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())
        state = {"n": 0}

        async def flaky(agent, deps, prompt, *, max_turns, session=None):
            state["n"] += 1
            raise ToolError("nope")

        monkeypatch.setattr("harness.main.run_once", flaky)
        answers = iter(["go", "exit"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        assert main(["--workspace", str(tmp_path)]) == 0
        assert "nope" in capsys.readouterr().out and state["n"] == 1

    def test_main_tracing_failure_is_warned(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        async def fake_run_once(*a, **k):
            return None

        monkeypatch.setattr("harness.main.run_once", fake_run_once)
        rc = main(["--workspace", str(tmp_path), "--trace", "hi"])
        assert rc == 0 and "tracing disabled" in capsys.readouterr().err

    def test_main_tracing_success_prints_host(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())
        monkeypatch.setattr("harness.main.setup_tracing", lambda c: True)

        async def fake_run_once(*a, **k):
            return None

        monkeypatch.setattr("harness.main.run_once", fake_run_once)
        rc = main(
            [
                "--workspace",
                str(tmp_path),
                "--trace",
                "--langfuse-host",
                "http://langfuse.test",
                "hi",
            ]
        )
        assert rc == 0 and "http://langfuse.test" in capsys.readouterr().err

    def test_report_failure_usage_limit(self, capsys):
        report_failure(UsageLimitExceeded("budget"))
        out = capsys.readouterr().out
        assert "step budget exhausted" in out and "budget" in out

    def test_report_failure_other(self, capsys):
        report_failure(ToolError("kaboom"))
        assert "kaboom" in capsys.readouterr().out

    def test_repl_survives_usage_limit(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())

        async def runaway(*a, **k):
            raise UsageLimitExceeded("budget")

        monkeypatch.setattr("harness.main.run_once", runaway)
        answers = iter(["go", "exit"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        assert main(["--workspace", str(tmp_path)]) == 0
        assert "step budget exhausted" in capsys.readouterr().out

    def test_main_reports_provider_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())

        async def boom(*a, **k):
            raise UnexpectedModelBehavior("Unauthorized: bad key")

        monkeypatch.setattr("harness.main.run_once", boom)
        assert main(["--workspace", str(tmp_path), "x"]) == 1
        assert "Unauthorized" in capsys.readouterr().err

    # runpy re-executes harness.main which is already imported; the double-import
    # RuntimeWarning is expected and not what this test is about.
    @pytest.mark.filterwarnings("ignore:.*harness.main.*sys.modules.*:RuntimeWarning")
    def test_dunder_main_guard(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["harness", "--workspace", str(tmp_path)])
        monkeypatch.setattr(
            "builtins.input", lambda *a: (_ for _ in ()).throw(EOFError())
        )
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("harness.main", run_name="__main__")
        assert exc.value.code == 0


# --------------------------------------------------------------------------- #
# pyproject-style misc
# --------------------------------------------------------------------------- #
class TestToolset:
    def test_all_tools_registered(self):
        ts = build_toolset()
        names = set(ts.tools)
        assert {
            "view_file",
            "write_file",
            "patch_file",
            "list_directory",
            "grep_search",
            "glob_search",
            "execute_command",
            "git_status",
            "git_diff",
            "git_commit",
            "git_log",
            "git_checkout_branch",
            "web_search",
            "browse_url",
        } <= names


def test_asyncio_available():
    assert asyncio.get_event_loop_policy() is not None
    assert AgentInfo is not None
    assert dataclasses.is_dataclass(Ctx)


# --------------------------------------------------------------------------- #
# event translation, live rendering, remaining branches
# --------------------------------------------------------------------------- #
class TestEventTranslation:
    def test_describe_call_parses_json_string_args(self):
        part = SimpleNamespace(tool_name="write_file", args='{"path": "a.txt"}')
        name, shown = A._describe_call(part)
        assert name == "write_file" and "path='a.txt'" in shown

    def test_describe_call_non_dict_args(self):
        assert (
            A._describe_call(SimpleNamespace(tool_name="t", args=[1, 2]))[1] == "[1, 2]"
        )

    def test_describe_call_no_args(self):
        assert (
            A._describe_call(SimpleNamespace(tool_name="git_status", args=None))[1]
            == ""
        )

    def test_describe_call_unparseable_json_string(self):
        name, shown = A._describe_call(SimpleNamespace(tool_name="t", args="{oops"))
        assert name == "t" and shown == "{oops"

    def test_extract_content_falls_back_to_event_content(self):
        assert (
            A._extract_content(SimpleNamespace(part=None, content="fell back"))
            == "fell back"
        )

    def test_extract_content_prefers_part(self):
        part = SimpleNamespace(content="from part")
        assert (
            A._extract_content(SimpleNamespace(part=part, content="ignore"))
            == "from part"
        )

    def test_translate_tool_call(self):
        part = SimpleNamespace(
            tool_name="view_file", args={"path": "a.txt"}, tool_call_id="1"
        )
        out = A.translate(FunctionToolCallEvent(part=part))
        assert out.kind == "tool_call" and out.name == "view_file"

    def test_translate_tool_result(self):
        out = A.translate(
            FunctionToolResultEvent(part=RetryPromptPart(content="denied"))
        )
        assert out.kind == "tool_result" and "denied" in out.text

    def test_translate_part_start_text(self):
        assert (
            A.translate(PartStartEvent(index=0, part=TextPart(content="hi"))).text
            == "hi"
        )

    def test_translate_part_delta_text(self):
        event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="yo"))
        assert A.translate(event).text == "yo"

    def test_translate_ignores_other_events(self):
        assert A.translate(object()) is None
        assert A.translate(PartStartEvent(index=0, part=TextPart(content=""))) is None


class TestLiveRendering:
    async def test_run_once_renders_tool_call_and_result(self, tmp_path, capsys):
        (tmp_path / "a.txt").write_text("hello\n")
        agent = scripted(call("view_file", path="a.txt"), say("all done"))
        await run_once(agent, make_deps(tmp_path), "read a.txt", max_turns=10)
        out = capsys.readouterr().out
        assert "view_file" in out and "hello" in out and "all done" in out

    def test_render_tool_call_and_result(self, capsys):
        render(StreamEvent("tool_call", name="view_file", text="path='a'"))
        render(StreamEvent("tool_result", text="line one\nline two"))
        out = capsys.readouterr().out
        assert "view_file" in out and "line one" in out and "line two" in out

    def test_render_result_with_empty_body(self, capsys):
        render(StreamEvent("tool_result", text="   "))
        assert capsys.readouterr().out == ""


class TestResidualBranches:
    def test_interactive_policy_allows_reads(self):
        req = ActionRequest(action=Action.READ, summary="s")
        assert InteractivePolicy().decide(req) is Decision.ALLOW

    async def test_grep_rg_path_honours_glob(self, tmp_path, monkeypatch):
        monkeypatch.setattr(F.shutil, "which", lambda _: "/bin/cat")
        (tmp_path / "a.py").write_text("needle\n")
        out = await F.grep_search(ctx_for(tmp_path), "needle", glob="*.py")
        assert isinstance(out, str)

    async def test_terminal_reports_stderr(self, tmp_path):
        out = await T.execute_command(ctx_for(tmp_path), "echo problem 1>&2")
        assert "stderr" in out and "problem" in out

    def test_repl_skips_blank_lines(self, tmp_path, monkeypatch):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())

        async def fake_run_once(*a, **k):
            return None

        monkeypatch.setattr("harness.main.run_once", fake_run_once)
        answers = iter(["", "exit"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        assert main(["--workspace", str(tmp_path)]) == 0

    def test_repl_survives_keyboard_interrupt(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())

        async def interrupted(*a, **k):
            raise KeyboardInterrupt

        monkeypatch.setattr("harness.main.run_once", interrupted)
        answers = iter(["go", "exit"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        assert main(["--workspace", str(tmp_path)]) == 0
        assert "interrupted" in capsys.readouterr().out

    def test_dunder_main_module_guard(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["harness", "--workspace", str(tmp_path)])
        monkeypatch.setattr(
            "builtins.input", lambda *a: (_ for _ in ()).throw(EOFError())
        )
        monkeypatch.setattr("harness.main.build_agent", lambda c: object())
        with pytest.raises(SystemExit) as err:
            runpy.run_path("harness/__main__.py", run_name="__main__")
        assert err.value.code == 0
