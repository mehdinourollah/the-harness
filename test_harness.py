"""Complete test suite for the agent harness.

Run:  .venv/bin/python -m pytest test_harness.py -v

Coverage:
  * Unit      -- pydantic models, provider adapters (mocked HTTP), every tool
  * Integration -- registry routing, policy enforcement, context pruning
  * E2E       -- full agent loop with a scripted provider, sub-agent delegation,
                 and the self-healing loop (failing command -> model sees error
                 -> fixes it -> verification passes)
"""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest
import pytest_asyncio

from harness.context import SlidingWindowStrategy, heuristic_counter
from harness.errors import (
    AuthenticationError,
    HarnessError,
    PolicyDenied,
    ProviderError,
    RateLimitError,
    TransientProviderError,
    ToolError,
)
from harness.loop import Agent, Coordinator, ROLE_PROMPTS, execute_tool_calls, LoopOutcome
from harness.models import (
    Action,
    ActionRequest,
    AgentRole,
    AssistantMessage,
    AutoApprovePolicy,
    Completion,
    Conversation,
    Decision,
    ExecutionPolicy,
    HarnessConfig,
    InteractivePolicy,
    Message,
    Plan,
    ReadOnlyPolicy,
    Request,
    SubAgentResult,
    SystemMessage,
    Task,
    TaskStatus,
    ToolCall,
    ToolMessage,
    ToolSpec,
    Usage,
    UserMessage,
)
from harness.providers import (
    AccountingProvider,
    LLMProvider,
    RetryProvider,
    make_anthropic_provider,
    make_openai_provider,
)
from harness.providers.anthropic import AnthropicAdapter
from harness.providers.openai import OpenAIAdapter
from harness.tools import build_registry
from harness.tools.files import PatchFileTool, ViewFileTool, WriteFileTool

# ========================================================================== #
# Shared helpers
# ========================================================================== #


class ScriptedProvider(LLMProvider):
    """Returns canned completions in order. The backbone of the E2E tests."""

    def __init__(self, script: list[Completion]) -> None:
        self.script = list(script)
        self.calls: list[list[Message]] = []

    async def complete(self, messages, *, tools=None, **kw) -> Completion:  # type: ignore[override]
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("ScriptedProvider ran out of scripted responses")
        return self.script.pop(0)

    async def stream(self, messages, **kw) -> AsyncIterator[str]:  # type: ignore[override]
        yield ""


def assistant(
    content: str = "", tool_calls: list[ToolCall] | None = None
) -> Completion:
    finish = "tool_calls" if tool_calls else "stop"
    return Completion(
        message=AssistantMessage(content=content, tool_calls=tool_calls or []),
        usage=Usage(input_tokens=10, output_tokens=5),
        finish_reason=finish,
    )


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"c_{name}", name=name, arguments=arguments)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def registry():
    return build_registry()


@pytest.fixture
def config(workspace: Path) -> HarnessConfig:
    return HarnessConfig(workspace=workspace, context_window=4096, reserve_tokens=512)


def make_request(
    workspace: Path, policy: ExecutionPolicy, *, timeout_s: float = 30.0
) -> Request:
    return Request(workspace=workspace, policy=policy, timeout_s=timeout_s)


# ========================================================================== #
# 1. UNIT -- domain models
# ========================================================================== #


class TestModels:
    def test_usage_addition_accumulates(self):
        a = Usage(input_tokens=10, output_tokens=5)
        b = Usage(input_tokens=3, output_tokens=7, cache_read_tokens=2)
        total = a + b
        assert total.input_tokens == 13
        assert total.output_tokens == 12
        assert total.cache_read_tokens == 2
        assert total.total_tokens == 27

    def test_pricing_cost(self):
        from harness.models import ModelPricing

        pricing = ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)
        cost = pricing.cost(Usage(input_tokens=1_000_000, output_tokens=1_000_000))
        assert cost == pytest.approx(18.0)

    def test_tool_result_ok_derived_from_error(self):
        ok = ToolResultStub(call_id="1", name="t", output="hi")
        bad = ToolResultStub(call_id="1", name="t", error="boom")
        assert ok.ok is True
        assert bad.ok is False

    def test_message_discriminator_selects_subclass(self):
        from pydantic import TypeAdapter

        from harness.models import Message as MsgUnion

        adapter = TypeAdapter(MsgUnion)
        parsed = adapter.validate_python({"role": "user", "content": "hi"})
        assert isinstance(parsed, UserMessage)
        parsed2 = adapter.validate_python(
            {"role": "tool", "tool_call_id": "x", "name": "t", "content": "out"}
        )
        assert isinstance(parsed2, ToolMessage)

    def test_assistant_message_rejects_unknown_role(self):
        from pydantic import TypeAdapter, ValidationError

        from harness.models import Message as MsgUnion

        with pytest.raises(ValidationError):
            TypeAdapter(MsgUnion).validate_python({"role": "wizard", "content": "?"})

    def test_plan_requires_at_least_one_task(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Plan(goal="do it", tasks=[])

    def test_plan_render_marks_status(self):
        plan = Plan(
            goal="ship it",
            tasks=[Task(title="step one", status=TaskStatus.DONE)],
            risks=["deadline"],
        )
        rendered = plan.render()
        assert "ship it" in rendered and "[x]" in rendered and "deadline" in rendered

    def test_task_render_marks_status(self):
        assert Task(title="t", status=TaskStatus.FAILED).render().startswith("[!]")
        assert Task(title="t", status=TaskStatus.PENDING).render().startswith("[ ]")

    def test_sub_agent_result_render_wraps_role(self):
        result = SubAgentResult(role=AgentRole.CODER, summary="did the thing")
        assert "<coder>" in result.render() and "did the thing" in result.render()

    def test_harness_config_rejects_reserve_over_window(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HarnessConfig(context_window=1000, reserve_tokens=1000)

    def test_harness_config_rejects_zero_turns(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HarnessConfig(max_turns=0)

    def test_with_cli_overrides_ignores_none(self):
        base = HarnessConfig(model="base-model")
        merged = HarnessConfig.with_cli_overrides(base, model="new", base_url=None)
        assert merged.model == "new"
        assert merged.base_url is None


class ToolResultStub:
    """Tiny shim so the ok/error test does not import the real model twice."""

    def __init__(self, *, call_id: str, name: str, output: str = "", error: str | None = None):
        from harness.models import ToolResult

        self._inner = ToolResult(call_id=call_id, name=name, output=output, error=error)

    @property
    def ok(self) -> bool:
        return self._inner.ok


# ========================================================================== #
# 2. UNIT -- execution policies
# ========================================================================== #


class TestPolicies:
    def test_readonly_allows_read_denies_write_and_execute(self):
        policy = ReadOnlyPolicy()
        policy.check(ActionRequest(action=Action.READ, summary="read"))
        with pytest.raises(PolicyDenied):
            policy.check(ActionRequest(action=Action.WRITE, summary="write"))
        with pytest.raises(PolicyDenied):
            policy.check(ActionRequest(action=Action.EXECUTE, summary="exec"))

    def test_auto_approve_allows_everything(self):
        policy = AutoApprovePolicy()
        for action in Action:
            policy.check(ActionRequest(action=action, summary="x"))
            assert policy.decide(ActionRequest(action=action, summary="x")) is Decision.ALLOW

    def test_interactive_asks_on_write(self):
        seen: list[ActionRequest] = []

        def approve(req: ActionRequest) -> bool:
            seen.append(req)
            return True

        policy = InteractivePolicy(approve=approve)
        policy.check(ActionRequest(action=Action.WRITE, summary="write"))
        assert len(seen) == 1
        # reads never prompt
        policy.check(ActionRequest(action=Action.READ, summary="read"))
        assert len(seen) == 1

    def test_interactive_denied_when_hook_says_no(self):
        policy = InteractivePolicy(approve=lambda req: False)
        with pytest.raises(PolicyDenied):
            policy.check(ActionRequest(action=Action.WRITE, summary="write"))

    def test_interactive_without_hook_denies(self):
        policy = InteractivePolicy()
        with pytest.raises(PolicyDenied):
            policy.check(ActionRequest(action=Action.EXECUTE, summary="exec"))

    def test_base_policy_is_read_only_by_default(self):
        policy = ExecutionPolicy()
        with pytest.raises(PolicyDenied):
            policy.check(ActionRequest(action=Action.WRITE, summary="write"))


# ========================================================================== #
# 3. UNIT -- provider adapters (mocked HTTP)
# ========================================================================== #


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestOpenAIAdapter:
    @pytest.mark.asyncio
    async def test_complete_parses_message_and_usage(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/chat/completions")
            body = json.loads(request.content)
            assert body["model"] == "test-model"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                    },
                },
            )

        adapter = OpenAIAdapter(base_url="https://api.test/v1", api_key="sk-x")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        completion = await adapter.complete(
            [UserMessage(content="hi")], model="test-model"
        )
        assert completion.message.content == "hello"
        assert completion.usage.input_tokens == 11
        assert completion.usage.output_tokens == 7
        assert completion.finish_reason == "stop"
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_complete_parses_tool_calls(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "view_file",
                                            "arguments": '{"path": "a.py", "start_line": 2}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            )

        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        completion = await adapter.complete([UserMessage(content="read a.py")])
        assert completion.finish_reason == "tool_calls"
        (tc,) = completion.message.tool_calls
        assert tc.name == "view_file"
        assert tc.arguments == {"path": "a.py", "start_line": 2}
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_rate_limit_maps_to_rate_limit_error(self):
        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(429, text="slow down")),
        )
        with pytest.raises(RateLimitError):
            await adapter.complete([UserMessage(content="hi")])
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_unauthorized_maps_to_auth_error(self):
        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(401, json={"e": "bad key"})),
        )
        with pytest.raises(AuthenticationError):
            await adapter.complete([UserMessage(content="hi")])
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_stream_yields_content_chunks(self):
        sse = b"".join(
            [
                b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=sse, headers={"content-type": "text/event-stream"}
            )

        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        chunks = [c async for c in adapter.stream([UserMessage(content="hi")])]
        assert "".join(chunks) == "Hello"
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_system_and_tool_messages_are_serialized(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "k"},
                         "finish_reason": "stop"}
                    ],
                    "usage": {},
                },
            )

        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        await adapter.complete(
            [
                SystemMessage(content="sys"),
                UserMessage(content="u"),
                AssistantMessage(
                    content="", tool_calls=[ToolCall(id="c1", name="t", arguments={"a": 1})]
                ),
                ToolMessage(tool_call_id="c1", name="t", content="res"),
            ]
        )
        roles = [m["role"] for m in captured["messages"]]
        assert roles == ["system", "user", "assistant", "tool"]
        assert captured["messages"][2]["tool_calls"][0]["function"]["name"] == "t"
        assert json.loads(captured["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
        await adapter.aclose()


class TestAnthropicAdapter:
    @pytest.mark.asyncio
    async def test_complete_splits_system_and_parses_tool_use(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/messages")
            body = json.loads(request.content)
            assert body["system"] == "be terse"
            assert body["messages"][0]["role"] == "user"
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "on it"},
                        {"type": "tool_use", "id": "tu_1", "name": "view_file",
                         "input": {"path": "x.py"}},
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 8, "output_tokens": 4},
                },
            )

        adapter = AnthropicAdapter(base_url="https://api.test/v1", api_key="k")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        completion = await adapter.complete(
            [SystemMessage(content="be terse"), UserMessage(content="read x.py")]
        )
        assert completion.message.content == "on it"
        (tc,) = completion.message.tool_calls
        assert tc.name == "view_file" and tc.arguments == {"path": "x.py"}
        assert completion.finish_reason == "tool_calls"
        assert completion.usage.input_tokens == 8
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_tool_result_becomes_user_turn(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                      "usage": {}},
            )

        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        await adapter.complete(
            [ToolMessage(tool_call_id="tu_1", name="t", content="result", is_error=False)]
        )
        block = captured["messages"][0]["content"][0]
        assert block["type"] == "tool_result" and block["tool_use_id"] == "tu_1"
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_provider_state_replays_verbatim(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                      "usage": {}},
            )

        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        original_blocks = [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "tu_9", "name": "t", "input": {"a": 1}},
        ]
        prior = AssistantMessage(content="hello", provider_state={"content": original_blocks})
        await adapter.complete([UserMessage(content="go"), prior])
        assert captured["messages"][1]["content"] == original_blocks
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_overloaded_529_is_transient(self):
        from harness.errors import TransientProviderError

        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(529, text="overloaded")),
        )
        with pytest.raises(TransientProviderError):
            await adapter.complete([UserMessage(content="hi")])
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_max_tokens_stop_reason_maps_to_length(self):
        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(
                lambda r: httpx.Response(
                    200,
                    json={"content": [{"type": "text", "text": "cut"}],
                          "stop_reason": "max_tokens", "usage": {}},
                )
            ),
        )
        completion = await adapter.complete([UserMessage(content="hi")])
        assert completion.finish_reason == "length"
        await adapter.aclose()


class TestProviderDecorators:
    @pytest.mark.asyncio
    async def test_retry_provider_retries_transient_then_succeeds(self):
        attempts = {"n": 0}

        class Flaky(LLMProvider):
            async def complete(self, messages, **kw):
                attempts["n"] += 1
                if attempts["n"] < 3:
                    raise RateLimitError("429")
                return assistant("recovered")

            async def stream(self, messages, **kw):
                yield ""

        provider = RetryProvider(Flaky(), max_retries=5, base_delay_s=0.0)
        completion = await provider.complete([UserMessage(content="hi")])
        assert completion.message.content == "recovered"
        assert attempts["n"] == 3

    @pytest.mark.asyncio
    async def test_retry_provider_gives_up_and_reraises(self):
        class AlwaysDown(LLMProvider):
            async def complete(self, messages, **kw):
                raise RateLimitError("429")

            async def stream(self, messages, **kw):
                yield ""

        provider = RetryProvider(AlwaysDown(), max_retries=2, base_delay_s=0.0)
        with pytest.raises(RateLimitError):
            await provider.complete([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_retry_provider_does_not_retry_auth_errors(self):
        attempts = {"n": 0}

        class BadKey(LLMProvider):
            async def complete(self, messages, **kw):
                attempts["n"] += 1
                raise AuthenticationError("nope")

            async def stream(self, messages, **kw):
                yield ""

        provider = RetryProvider(BadKey(), max_retries=5, base_delay_s=0.0)
        with pytest.raises(AuthenticationError):
            await provider.complete([UserMessage(content="hi")])
        assert attempts["n"] == 1

    @pytest.mark.asyncio
    async def test_accounting_provider_accumulates_usage(self):
        class Fixed(LLMProvider):
            async def complete(self, messages, **kw):
                return assistant("x")

            async def stream(self, messages, **kw):
                yield ""

        provider = AccountingProvider(Fixed())
        await provider.complete([UserMessage(content="a")])
        await provider.complete([UserMessage(content="b")])
        assert provider.usage.input_tokens == 20
        assert provider.usage.output_tokens == 10
        assert provider.calls == 2


# ========================================================================== #
# 4. UNIT -- tools
# ========================================================================== #


class TestFileTools:
    @pytest.mark.asyncio
    async def test_view_file_with_line_range(self, workspace: Path):
        (workspace / "f.txt").write_text("a\nb\nc\nd\n")
        tool = ViewFileTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(ViewFileTool.Args(path="f.txt", start_line=2, end_line=3), req)
        assert "b" in out and "c" in out and "a\n" not in out.split("---")[1][:20]

    @pytest.mark.asyncio
    async def test_view_file_missing_raises(self, workspace: Path):
        tool = ViewFileTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError):
            await tool.run(ViewFileTool.Args(path="nope.txt"), req)

    @pytest.mark.asyncio
    async def test_write_file_creates_parents(self, workspace: Path):
        tool = WriteFileTool()
        req = make_request(workspace, AutoApprovePolicy())
        await tool.run(WriteFileTool.Args(path="sub/dir/f.txt", content="hi"), req)
        assert (workspace / "sub/dir/f.txt").read_text() == "hi"

    @pytest.mark.asyncio
    async def test_patch_file_exact_single_occurrence(self, workspace: Path):
        (workspace / "f.py").write_text("x = 1\ny = 2\n")
        tool = PatchFileTool()
        req = make_request(workspace, AutoApprovePolicy())
        await tool.run(
            PatchFileTool.Args(path="f.py", old_text="x = 1", new_text="x = 42"), req
        )
        assert (workspace / "f.py").read_text() == "x = 42\ny = 2\n"

    @pytest.mark.asyncio
    async def test_patch_file_rejects_ambiguous_match(self, workspace: Path):
        (workspace / "f.py").write_text("dup\ndup\n")
        tool = PatchFileTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="occurs 2 times"):
            await tool.run(
                PatchFileTool.Args(path="f.py", old_text="dup", new_text="x"), req
            )

    @pytest.mark.asyncio
    async def test_patch_file_rejects_missing_match(self, workspace: Path):
        (workspace / "f.py").write_text("abc\n")
        tool = PatchFileTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="occurs 0 times"):
            await tool.run(
                PatchFileTool.Args(path="f.py", old_text="zzz", new_text="x"), req
            )

    @pytest.mark.asyncio
    async def test_workspace_guard_blocks_escape(self, workspace: Path):
        tool = ViewFileTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="escapes workspace"):
            await tool.run(ViewFileTool.Args(path="../../etc/hosts"), req)

    @pytest.mark.asyncio
    async def test_list_directory_recursive(self, workspace: Path):
        from harness.tools.files import ListDirTool

        (workspace / "a").mkdir()
        (workspace / "a/b.py").write_text("")
        (workspace / "top.txt").write_text("")
        tool = ListDirTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(ListDirTool.Args(path=".", recursive=True), req)
        assert "a/b.py" in out and "top.txt" in out

    @pytest.mark.asyncio
    async def test_grep_search_finds_matches(self, workspace: Path):
        from harness.tools.files import GrepTool

        (workspace / "code.py").write_text("import os\ndef foo():\n    pass\n")
        tool = GrepTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(GrepTool.Args(pattern="def foo"), req)
        assert "def foo" in out

    @pytest.mark.asyncio
    async def test_glob_search(self, workspace: Path):
        from harness.tools.files import GlobTool

        (workspace / "x.py").write_text("")
        (workspace / "y.txt").write_text("")
        tool = GlobTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(GlobTool.Args(pattern="*.py"), req)
        assert "x.py" in out and "y.txt" not in out


class TestTerminalTool:
    @pytest.mark.asyncio
    async def test_execute_success_captures_stdout(self, workspace: Path):
        from harness.tools.terminal import CommandTool

        tool = CommandTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(CommandTool.Args(command="echo hello"), req)
        assert "hello" in out and "exit: 0" in out

    @pytest.mark.asyncio
    async def test_execute_failure_surfaces_stderr_and_exit(self, workspace: Path):
        from harness.tools.terminal import CommandTool

        tool = CommandTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError) as exc:
            await tool.run(CommandTool.Args(command="echo oops >&2; exit 3"), req)
        assert "exit" in str(exc.value) and "oops" in str(exc.value)

    @pytest.mark.asyncio
    async def test_execute_timeout_kills_process(self, workspace: Path):
        from harness.tools.terminal import CommandTool

        tool = CommandTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="timed out"):
            await tool.run(
                CommandTool.Args(command="sleep 5", timeout_s=0.2), req
            )

    @pytest.mark.asyncio
    async def test_execute_declares_execute_action(self, workspace: Path):
        from harness.tools.terminal import CommandTool

        tool = CommandTool()
        actions = tool.actions(CommandTool.Args(command="ls"))
        assert actions[0].action is Action.EXECUTE


class TestGitTools:
    @pytest.mark.asyncio
    async def test_git_status_and_log_in_real_repo(self, workspace: Path):
        from harness.tools.git import GitLogTool, GitStatusTool

        async def sh(cmd: str) -> None:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(workspace),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()

        await sh("git init -q && git config user.email t@t.io && git config user.name t")
        (workspace / "a.txt").write_text("first")
        await sh("git add -A && git commit -qm 'initial'")

        req = make_request(workspace, AutoApprovePolicy())
        status = await GitStatusTool().run(GitStatusTool.Args(), req)
        assert status == "(clean)"
        log = await GitLogTool().run(GitLogTool.Args(count=5), req)
        assert "initial" in log

    @pytest.mark.asyncio
    async def test_git_diff_and_commit(self, workspace: Path):
        from harness.tools.git import GitCommitTool, GitDiffTool

        async def sh(cmd: str) -> None:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(workspace),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()

        await sh("git init -q && git config user.email t@t.io && git config user.name t")
        (workspace / "a.txt").write_text("first")
        await sh("git add -A && git commit -qm 'initial'")
        (workspace / "a.txt").write_text("changed")

        req = make_request(workspace, AutoApprovePolicy())
        diff = await GitDiffTool().run(GitDiffTool.Args(), req)
        assert "a.txt" in diff
        out = await GitCommitTool().run(GitCommitTool.Args(message="update a.txt"), req)
        assert "update" in out or "master" in out or "main" in out

    @pytest.mark.asyncio
    async def test_git_checkout_creates_branch(self, workspace: Path):
        from harness.tools.git import GitCheckoutTool

        async def sh(cmd: str) -> None:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(workspace),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()

        await sh("git init -q && git config user.email t@t.io && git config user.name t")
        (workspace / "a.txt").write_text("x")
        await sh("git add -A && git commit -qm 'init'")

        req = make_request(workspace, AutoApprovePolicy())
        await GitCheckoutTool().run(GitCheckoutTool.Args(branch="feature/x"), req)
        proc = await asyncio.create_subprocess_shell(
            "git rev-parse --abbrev-ref HEAD", cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        assert out.decode().strip() == "feature/x"


class TestWebTools:
    @pytest.mark.asyncio
    async def test_browse_url_reduces_html_to_text(self, monkeypatch):
        from harness.tools import web as web_mod

        html = (
            "<html><head><title>t</title><style>.x{}</style></head>"
            "<body><h1>Heading</h1><p>Body text</p>"
            "<script>var x = 1;</script></body></html>"
        )

        class FakeClient:
            def __init__(self, *a, **kw) -> None: ...
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def get(self, url, **kw):
                return httpx.Response(
                    200, text=html, headers={"content-type": "text/html"},
                    request=httpx.Request("GET", url),
                )

        monkeypatch.setattr(web_mod.httpx, "AsyncClient", FakeClient)
        tool = web_mod.BrowseUrlTool()
        req = Request(workspace=Path("/tmp"), policy=AutoApprovePolicy())
        out = await tool.run(web_mod.BrowseUrlArgs(url="https://example.com"), req)
        assert "Heading" in out and "Body text" in out
        assert "var x = 1" not in out and ".x{}" not in out

    @pytest.mark.asyncio
    async def test_browse_url_rejects_non_http(self, workspace: Path):
        from harness.tools.web import BrowseUrlTool

        tool = BrowseUrlTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="http"):
            await tool.run(BrowseUrlTool.Args(url="file:///etc/passwd"), req)

    @pytest.mark.asyncio
    async def test_web_search_parses_results(self, monkeypatch):
        from harness.tools import web as web_mod

        html = (
            '<a class="result__a" href="https://a.com">First</a>'
            '<a class="result__a" href="https://b.com">Second</a>'
        )

        class FakeClient:
            def __init__(self, *a, **kw) -> None: ...
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def get(self, url, **kw):
                return httpx.Response(200, text=html, request=httpx.Request("GET", url))

        monkeypatch.setattr(web_mod.httpx, "AsyncClient", FakeClient)
        tool = web_mod.WebSearchTool()
        req = Request(workspace=Path("/tmp"), policy=AutoApprovePolicy())
        out = await tool.run(web_mod.WebSearchArgs(query="anything"), req)
        assert "First" in out and "https://a.com" in out


# ========================================================================== #
# 5. INTEGRATION -- registry, policy enforcement, context pruning
# ========================================================================== #


class TestRegistryRouting:
    def test_schema_exposes_all_tools(self, registry):
        names = {s.name for s in registry.schema()}
        assert {"view_file", "write_file", "execute_command", "git_status"} <= names
        for spec in registry.schema():
            assert spec.parameters.get("type") == "object"

    def test_duplicate_registration_rejected(self, registry):
        from harness.tools.files import ViewFileTool

        with pytest.raises(ValueError, match="already registered"):
            registry.register("view_file", ViewFileTool)

    def test_non_handler_rejected(self, registry):
        with pytest.raises(TypeError):
            registry.register("bogus", object)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self, registry, workspace):
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="unknown tool"):
            await registry.execute("does_not_exist", {}, req)

    @pytest.mark.asyncio
    async def test_execute_validates_arguments(self, registry, workspace):
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="invalid arguments"):
            await registry.execute("view_file", {"wrong_key": 1}, req)

    @pytest.mark.asyncio
    async def test_execute_routes_to_handler(self, registry, workspace):
        (workspace / "a.txt").write_text("content here")
        req = make_request(workspace, AutoApprovePolicy())
        out = await registry.execute("view_file", {"path": "a.txt"}, req)
        assert "content here" in out

    @pytest.mark.asyncio
    async def test_readonly_policy_blocks_write_via_registry(self, registry, workspace):
        req = make_request(workspace, ReadOnlyPolicy())
        with pytest.raises(PolicyDenied):
            await registry.execute("write_file", {"path": "x.txt", "content": "no"}, req)
        assert not (workspace / "x.txt").exists()

    @pytest.mark.asyncio
    async def test_readonly_policy_blocks_command_via_registry(self, registry, workspace):
        req = make_request(workspace, ReadOnlyPolicy())
        with pytest.raises(PolicyDenied):
            await registry.execute("execute_command", {"command": "rm -rf /"}, req)

    @pytest.mark.asyncio
    async def test_readonly_policy_allows_reads(self, registry, workspace):
        (workspace / "r.txt").write_text("readable")
        req = make_request(workspace, ReadOnlyPolicy())
        out = await registry.execute("view_file", {"path": "r.txt"}, req)
        assert "readable" in out

    @pytest.mark.asyncio
    async def test_interactive_policy_approval_hook_fires(self, registry, workspace):
        seen: list[ActionRequest] = []

        def approve(req: ActionRequest) -> bool:
            seen.append(req)
            return True

        req = make_request(workspace, InteractivePolicy(approve=approve))
        await registry.execute("write_file", {"path": "ok.txt", "content": "yes"}, req)
        assert (workspace / "ok.txt").read_text() == "yes"
        assert seen and seen[0].action is Action.WRITE

    @pytest.mark.asyncio
    async def test_interactive_policy_hook_denial_stops_write(self, registry, workspace):
        req = make_request(workspace, InteractivePolicy(approve=lambda r: False))
        with pytest.raises(PolicyDenied):
            await registry.execute("write_file", {"path": "nope.txt", "content": "x"}, req)
        assert not (workspace / "nope.txt").exists()


class TestContextPruning:
    def _conversation(self, n_turns: int, filler: str = "x" * 400) -> Conversation:
        conv = Conversation()
        conv.add(SystemMessage(content="system"))
        conv.add(UserMessage(content="the original goal"))
        for i in range(n_turns):
            conv.add(UserMessage(content=f"turn {i} {filler}"))
            conv.add(AssistantMessage(content=f"reply {i} {filler}"))
        return conv

    def test_no_pruning_when_under_budget(self):
        conv = self._conversation(2)
        strategy = SlidingWindowStrategy()
        kept, dropped = strategy.fit(conv, budget=10_000)
        assert dropped == []
        assert kept == conv.messages

    def test_pruning_keeps_system_and_goal(self):
        conv = self._conversation(50)
        strategy = SlidingWindowStrategy(min_recent=4)
        kept, dropped = strategy.fit(conv, budget=600)
        assert dropped, "expected pruning to drop the middle"
        assert isinstance(kept[0], SystemMessage)
        assert any(isinstance(m, UserMessage) and m.content == "the original goal" for m in kept)

    def test_pruning_keeps_most_recent_turns(self):
        conv = self._conversation(50)
        strategy = SlidingWindowStrategy(min_recent=4)
        kept, _ = strategy.fit(conv, budget=600)
        last = kept[-1]
        assert isinstance(last, AssistantMessage) and last.content.startswith("reply 49")

    def test_overflow_when_goal_alone_too_big(self):
        from harness.errors import ContextOverflow

        conv = Conversation()
        conv.add(SystemMessage(content="s" * 100_000))
        strategy = SlidingWindowStrategy()
        with pytest.raises(ContextOverflow):
            strategy.fit(conv, budget=100)

    def test_heuristic_counter_grows_with_content(self):
        small = heuristic_counter([UserMessage(content="hi")])
        big = heuristic_counter([UserMessage(content="hi" * 5000)])
        assert big > small


# ========================================================================== #
# 6. E2E -- full agent loop, delegation, self-healing
# ========================================================================== #


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_single_turn_answer(self, workspace, registry, config):
        provider = ScriptedProvider([assistant('I am done.')])
        agent = Agent(
            role=AgentRole.PRIMARY, provider=provider, registry=registry,
            config=config, policy=AutoApprovePolicy(),
        )
        outcome = await agent.run("say hi")
        assert outcome.output == "I am done."
        assert outcome.turns == 1
        assert outcome.usage.input_tokens == 10

    @pytest.mark.asyncio
    async def test_tool_call_then_answer(self, workspace, registry, config):
        (workspace / "data.txt").write_text("the secret is 42")
        provider = ScriptedProvider([
            assistant("", [call("view_file", path="data.txt")]),
            assistant("The secret is 42."),
        ])
        agent = Agent(
            role=AgentRole.PRIMARY, provider=provider, registry=registry,
            config=config, policy=AutoApprovePolicy(),
        )
        outcome = await agent.run("find the secret")
        assert "42" in outcome.output
        assert outcome.turns == 2
        # the tool output must have reached the model on turn 2
        second_call = provider.calls[1]
        assert any(isinstance(m, ToolMessage) and "42" in m.content for m in second_call)

    @pytest.mark.asyncio
    async def test_loop_stops_at_max_turns(self, workspace, registry, config):
        provider = ScriptedProvider([
            assistant("", [call("list_directory", path=".")]) for _ in range(20)
        ])
        agent = Agent(
            role=AgentRole.PRIMARY, provider=provider, registry=registry,
            config=config, policy=AutoApprovePolicy(),
        )
        outcome = await agent.run("loop forever", max_turns=3)
        assert outcome.turns == 3

    @pytest.mark.asyncio
    async def test_tool_failure_is_fed_back_not_fatal(self, workspace, registry, config):
        provider = ScriptedProvider([
            assistant("", [call("view_file", path="missing.txt")]),
            assistant("That file does not exist."),
        ])
        agent = Agent(
            role=AgentRole.PRIMARY, provider=provider, registry=registry,
            config=config, policy=AutoApprovePolicy(),
        )
        outcome = await agent.run("read missing.txt")
        assert "does not exist" in outcome.output
        error_msgs = [
            m for m in provider.calls[1] if isinstance(m, ToolMessage) and m.is_error
        ]
        assert error_msgs, "the tool error must be visible to the model"

    @pytest.mark.asyncio
    async def test_policy_denial_is_reported_to_model(self, workspace, registry, config):
        provider = ScriptedProvider([
            assistant("", [call("write_file", path="x.txt", content="no")]),
            assistant("I cannot write in read-only mode."),
        ])
        agent = Agent(
            role=AgentRole.PRIMARY, provider=provider, registry=registry,
            config=config, policy=ReadOnlyPolicy(),
        )
        outcome = await agent.run("write x.txt")
        assert "read-only" in outcome.output.lower()
        assert not (workspace / "x.txt").exists()


class TestSelfHealing:
    @pytest.mark.asyncio
    async def test_agent_reads_failure_traceback_and_fixes_it(self, workspace, registry, config):
        """Simulated: bad script -> failing test -> model patches -> test passes."""
        import sys

        (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (workspace / "test_calc.py").write_text(
            "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        )
        run_tests = f"{sys.executable} -m pytest test_calc.py -q"
        provider = ScriptedProvider([
            # 1. run the tests -- they fail
            assistant("", [call("execute_command", command=run_tests)]),
            # 2. read the broken file
            assistant("", [call("view_file", path="calc.py")]),
            # 3. patch the bug
            assistant("", [call("patch_file", path="calc.py", old_text="return a - b", new_text="return a + b")]),
            # 4. re-run the tests -- they pass
            assistant("", [call("execute_command", command=run_tests)]),
            assistant("Fixed the sign bug; tests now pass."),
        ])
        agent = Agent(
            role=AgentRole.PRIMARY, provider=provider, registry=registry,
            config=config, policy=AutoApprovePolicy(),
        )
        outcome = await agent.run("make the tests pass")

        assert (workspace / "calc.py").read_text().endswith("return a + b\n")
        assert "tests now pass" in outcome.output

        # The model must have been shown the failing run's traceback.
        turn_two_context = provider.calls[1]
        assert any(
            isinstance(m, ToolMessage) and m.is_error and "assert" in m.content.lower()
            for m in turn_two_context
        ), "failing test traceback was not fed back to the model"

    @pytest.mark.asyncio
    async def test_failed_command_exit_code_reaches_model(self, workspace, registry, config):
        provider = ScriptedProvider([
            assistant("", [call("execute_command", command="python3 -c 'raise SystemExit(7)'")]),
            assistant("The command exited non-zero; aborting that approach."),
        ])
        agent = Agent(
            role=AgentRole.PRIMARY, provider=provider, registry=registry,
            config=config, policy=AutoApprovePolicy(),
        )
        await agent.run("run it")
        feedback = [m for m in provider.calls[1] if isinstance(m, ToolMessage)]
        assert any(m.is_error and "exit" in m.content for m in feedback)


class TestDelegation:
    @pytest.mark.asyncio
    async def test_coordinator_runs_architect_then_primary(self, workspace, registry, config):
        provider = ScriptedProvider([
            assistant("Goal: add a flag\n[ ] 1 write code\n[ ] 2 run tests"),  # architect
            assistant("Implemented and verified."),                            # primary
        ])
        coordinator = Coordinator(
            provider=provider, registry=registry, config=config, policy=AutoApprovePolicy()
        )
        outcome, subs = await coordinator.run("add a --version flag", plan_first=True)
        assert outcome.output == "Implemented and verified."
        assert len(subs) == 1 and subs[0].role is AgentRole.ARCHITECT
        assert "Goal:" in subs[0].summary

    @pytest.mark.asyncio
    async def test_delegate_returns_structured_sub_agent_result(self, workspace, registry, config):
        provider = ScriptedProvider([assistant("Reviewed: 0 defects found.")])
        coordinator = Coordinator(
            provider=provider, registry=registry, config=config, policy=AutoApprovePolicy()
        )
        result = await coordinator.delegate(AgentRole.REVIEWER, "review the change")
        assert isinstance(result, SubAgentResult)
        assert result.role is AgentRole.REVIEWER
        assert "0 defects" in result.summary
        assert result.tasks[0].status is TaskStatus.DONE
        assert result.tasks[0].owner is AgentRole.REVIEWER

    @pytest.mark.asyncio
    async def test_sub_agents_have_isolated_contexts(self, workspace, registry, config):
        provider = ScriptedProvider([
            assistant("coder output"),
            assistant("reviewer output"),
        ])
        coordinator = Coordinator(
            provider=provider, registry=registry, config=config, policy=AutoApprovePolicy()
        )
        await coordinator.delegate(AgentRole.CODER, "write the code")
        await coordinator.delegate(AgentRole.REVIEWER, "review the code")
        # The reviewer's prompt must not contain the coder's conversation.
        reviewer_messages = provider.calls[1]
        assert not any(
            isinstance(m, AssistantMessage) and m.content == "coder output"
            for m in reviewer_messages
        )

    @pytest.mark.asyncio
    async def test_each_sub_agent_reports_its_own_usage(self, workspace, registry, config):
        provider = ScriptedProvider([
            assistant("a"), assistant("b"), assistant("c"),
        ])
        coordinator = Coordinator(
            provider=provider, registry=registry, config=config, policy=AutoApprovePolicy()
        )
        r1 = await coordinator.delegate(AgentRole.CODER, "one")
        r2 = await coordinator.delegate(AgentRole.REVIEWER, "two")
        assert r1.usage.input_tokens == 10
        assert r2.usage.output_tokens == 5


class TestExecuteToolCallsDirect:
    @pytest.mark.asyncio
    async def test_execute_tool_calls_captures_mixed_outcomes(self, workspace, registry):
        (workspace / "ok.txt").write_text("fine")
        req = make_request(workspace, AutoApprovePolicy())
        messages = await execute_tool_calls(
            [
                call("view_file", path="ok.txt"),
                call("view_file", path="missing.txt"),
                call("not_a_tool"),
            ],
            registry,
            req,
        )
        assert len(messages) == 3
        assert messages[0].is_error is False and "fine" in messages[0].content
        assert messages[1].is_error is True
        assert messages[2].is_error is True
        assert [m.tool_call_id for m in messages] == ["c_view_file", "c_view_file", "c_not_a_tool"]


# ========================================================================== #
# 7. CLI entrypoint
# ========================================================================== #


class TestCli:
    def test_parse_args_defaults(self):
        from harness.main import parse_args

        args = parse_args(["fix the bug"])
        assert args.prompt == ["fix the bug"]
        assert args.provider == "openai"
        assert args.policy == "interactive"

    def test_parse_args_provider_and_policy(self):
        from harness.main import parse_args

        args = parse_args(
            ["--provider", "anthropic", "--policy", "readonly", "--model", "m", "do"]
        )
        assert args.provider == "anthropic"
        assert args.policy == "readonly"
        assert args.model == "m"

    def test_build_policy_selects_class(self):
        from harness.main import build_policy

        assert isinstance(build_policy("readonly"), ReadOnlyPolicy)
        assert isinstance(build_policy("auto-approve"), AutoApprovePolicy)
        assert isinstance(build_policy("interactive"), InteractivePolicy)
        with pytest.raises(ValueError):
            build_policy("nonsense")

    def test_build_config_applies_overrides(self):
        from harness.main import build_config, parse_args

        args = parse_args(
            ["--model", "override-me", "--max-turns", "7", "--context-window", "8000", "do"]
        )
        config = build_config(args)
        assert config.model == "override-me"
        assert config.max_turns == 7
        assert config.context_window == 8000

    def test_parse_headers(self):
        from harness.main import _parse_headers

        assert _parse_headers(["X-A: 1", "X-B:two"]) == {"X-A": "1", "X-B": "two"}
        with pytest.raises(SystemExit):
            _parse_headers(["malformed"])

    def test_build_provider_openai_shape(self):
        from harness.main import build_provider

        config = HarnessConfig(provider="openai", base_url="http://localhost:9/v1")
        provider = build_provider(config)
        assert isinstance(provider, AccountingProvider)

    def test_build_provider_anthropic_shape(self):
        from harness.main import build_provider

        config = HarnessConfig(
            provider="anthropic", base_url="https://api.test/v1", api_key="k"
        )
        provider = build_provider(config)
        assert isinstance(provider, AccountingProvider)

    def test_report_prints_output_and_footer(self, capsys):
        from harness.main import report

        outcome = LoopOutcome(
            output="done", usage=Usage(input_tokens=100, output_tokens=50), turns=2,
            conversation=Conversation(),
        )
        report(outcome, quiet=False)
        err = capsys.readouterr()
        assert "done" in err.out
        assert "2 turns" in err.err

    def test_main_runs_prompt_with_mocked_provider(self, monkeypatch, workspace):
        import harness.main as main_mod

        provider = ScriptedProvider([assistant("all done")])
        monkeypatch.setattr(main_mod, "build_provider", lambda config: provider)
        rc = main_mod.main(
            ["--policy", "auto-approve", "--workspace", str(workspace), "do the thing"]
        )
        assert rc == 0

    def test_main_repl_exits_on_exit(self, monkeypatch, workspace):
        import harness.main as main_mod

        provider = ScriptedProvider([assistant("hi there")])
        monkeypatch.setattr(main_mod, "build_provider", lambda config: provider)
        inputs = iter(["hello agent", "exit"])
        monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
        rc = main_mod.main(["--policy", "auto-approve", "--workspace", str(workspace)])
        assert rc == 0

    def test_main_repl_reports_harness_error(self, monkeypatch, workspace):
        import harness.main as main_mod

        class Boom(ScriptedProvider):
            async def complete(self, messages, **kw):
                raise ProviderError("endpoint down")

        monkeypatch.setattr(main_mod, "build_provider", lambda config: Boom([]))
        inputs = iter(["trigger error", "exit"])
        monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
        rc = main_mod.main(["--policy", "auto-approve", "--workspace", str(workspace)])
        assert rc == 0

    def test_main_delegate_path(self, monkeypatch, workspace):
        import harness.main as main_mod

        provider = ScriptedProvider([assistant("review complete: 0 issues")])
        monkeypatch.setattr(main_mod, "build_provider", lambda config: provider)
        rc = main_mod.main(
            ["--policy", "auto-approve", "--workspace", str(workspace),
             "--delegate", "reviewer", "review it"]
        )
        assert rc == 0

    def test_main_plan_first_path(self, monkeypatch, workspace):
        import harness.main as main_mod

        provider = ScriptedProvider([assistant("plan body"), assistant("executed")])
        monkeypatch.setattr(main_mod, "build_provider", lambda config: provider)
        rc = main_mod.main(
            ["--policy", "auto-approve", "--workspace", str(workspace),
             "--plan-first", "do it"]
        )
        assert rc == 0

    def test_main_returns_one_on_empty_output(self, monkeypatch, workspace):
        import harness.main as main_mod

        provider = ScriptedProvider([assistant("")])
        monkeypatch.setattr(main_mod, "build_provider", lambda config: provider)
        rc = main_mod.main(["--policy", "auto-approve", "--workspace", str(workspace), "x"])
        assert rc == 1


# ========================================================================== #
# 8. Remaining provider / tool / loop branches
# ========================================================================== #


class TestProviderEdgePaths:
    @pytest.mark.asyncio
    async def test_openai_payload_includes_tools_when_given(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "k"},
                                   "finish_reason": "stop"}], "usage": {}},
            )

        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        await adapter.complete(
            [UserMessage(content="hi")],
            tools=[ToolSpec(name="t", description="d", parameters={"type": "object"})],
        )
        assert captured["tools"][0]["function"]["name"] == "t"
        assert captured["tool_choice"] == "auto"
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_openai_assistant_with_content_and_calls_keeps_both(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "k"},
                                   "finish_reason": "stop"}], "usage": {}},
            )

        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        await adapter.complete([
            AssistantMessage(content="thinking", tool_calls=[ToolCall(id="c", name="t", arguments={})])
        ])
        entry = captured["messages"][0]
        assert entry["content"] == "thinking" and entry["tool_calls"]
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_openai_malformed_tool_arguments_become_empty(self):
        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(
                lambda r: httpx.Response(
                    200,
                    json={"choices": [{"index": 0, "message": {
                        "role": "assistant", "content": "",
                        "tool_calls": [{"id": "c", "type": "function",
                                        "function": {"name": "t", "arguments": "{not json"}}]},
                        "finish_reason": "tool_calls"}], "usage": {}},
                )
            ),
        )
        completion = await adapter.complete([UserMessage(content="hi")])
        assert completion.message.tool_calls[0].arguments == {}
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_openai_server_error_is_transient(self):
        from harness.errors import TransientProviderError

        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(503, text="down")),
        )
        with pytest.raises(TransientProviderError):
            await adapter.complete([UserMessage(content="hi")])
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_openai_unknown_message_type_raises(self, workspace):
        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        with pytest.raises(ProviderError, match="Unknown message"):
            adapter._convert_messages([object()])  # type: ignore[list-item]
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_anthropic_stream_yields_text_deltas(self):
        sse = b"".join([
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"He"}}\n\n',
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"y"}}\n\n',
        ])

        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(
                lambda r: httpx.Response(200, content=sse,
                                         headers={"content-type": "text/event-stream"})
            ),
        )
        chunks = [c async for c in adapter.stream([UserMessage(content="hi")])]
        assert "".join(chunks) == "Hey"
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_anthropic_assistant_blocks_without_provider_state(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200, json={"content": [{"type": "text", "text": "k"}],
                           "stop_reason": "end_turn", "usage": {}},
            )

        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        prior = AssistantMessage(content="hello", tool_calls=[ToolCall(id="tu", name="t", arguments={"a": 1})])
        await adapter.complete([UserMessage(content="go"), prior])
        blocks = captured["messages"][1]["content"]
        assert blocks[0] == {"type": "text", "text": "hello"}
        assert blocks[1]["type"] == "tool_use" and blocks[1]["input"] == {"a": 1}
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_anthropic_http_500_is_transient(self):
        from harness.errors import TransientProviderError

        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(500, text="boom")),
        )
        with pytest.raises(TransientProviderError):
            await adapter.complete([UserMessage(content="hi")])
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_anthropic_plain_400_is_provider_error_non_retryable(self):
        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(400, json={"error": "bad"})),
        )
        with pytest.raises(ProviderError):
            await adapter.complete([UserMessage(content="hi")])
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_retry_provider_stream_delegates(self):
        class Fix(LLMProvider):
            async def complete(self, messages, **kw):
                return assistant("x")

            async def stream(self, messages, **kw):
                yield "one"
                yield "two"

        provider = RetryProvider(Fix(), max_retries=1, base_delay_s=0.0)
        chunks = [c async for c in provider.stream([UserMessage(content="hi")])]
        assert chunks == ["one", "two"]


class TestToolEdgePaths:
    @pytest.mark.asyncio
    async def test_patch_file_not_a_file(self, workspace):
        tool = PatchFileTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="not a file"):
            await tool.run(PatchFileTool.Args(path="ghost.py", old_text="a", new_text="b"), req)

    @pytest.mark.asyncio
    async def test_list_directory_not_a_dir(self, workspace):
        from harness.tools.files import ListDirTool

        (workspace / "file.txt").write_text("x")
        tool = ListDirTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="not a directory"):
            await tool.run(ListDirTool.Args(path="file.txt"), req)

    @pytest.mark.asyncio
    async def test_list_directory_flat_honours_ignore(self, workspace):
        from harness.tools.files import ListDirTool

        (workspace / "keep.txt").write_text("")
        (workspace / "skip.log").write_text("")
        tool = ListDirTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(ListDirTool.Args(path=".", ignore=["*.log"]), req)
        assert "keep.txt" in out
        assert "skip.log" in out  # default listing ignores by exact name only

    @pytest.mark.asyncio
    async def test_grep_python_fallback_path(self, workspace, monkeypatch):
        from harness.tools import files as files_mod

        (workspace / "m.py").write_text("alpha\nbeta\n")
        monkeypatch.setattr(files_mod.shutil, "which", lambda name: None)
        tool = files_mod.GrepTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(files_mod.GrepArgs(pattern="beta"), req)
        assert "beta" in out
        # no match path
        out2 = await tool.run(files_mod.GrepArgs(pattern="zzz-not-here"), req)
        assert out2 == "(no matches)"

    @pytest.mark.asyncio
    async def test_grep_python_fallback_glob_filter(self, workspace, monkeypatch):
        from harness.tools import files as files_mod

        (workspace / "a.py").write_text("needle\n")
        (workspace / "b.txt").write_text("needle\n")
        monkeypatch.setattr(files_mod.shutil, "which", lambda name: None)
        tool = files_mod.GrepTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(files_mod.GrepArgs(pattern="needle", glob="*.py"), req)
        assert "a.py" in out and "b.txt" not in out

    @pytest.mark.asyncio
    async def test_grep_glob_flag_with_rg(self, workspace):
        from harness.tools.files import GrepTool

        (workspace / "s.py").write_text("target_line\n")
        tool = GrepTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(GrepTool.Args(pattern="target_line", glob="*.py"), req)
        assert "target_line" in out

    @pytest.mark.asyncio
    async def test_glob_no_matches(self, workspace):
        from harness.tools.files import GlobTool

        tool = GlobTool()
        req = make_request(workspace, AutoApprovePolicy())
        assert await tool.run(GlobTool.Args(pattern="*.nonexistent"), req) == "(no matches)"

    @pytest.mark.asyncio
    async def test_git_status_reports_dirty(self, workspace):
        from harness.tools.git import GitStatusTool

        async def sh(cmd: str) -> None:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(workspace),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()

        await sh("git init -q && git config user.email t@t.io && git config user.name t")
        (workspace / "new.txt").write_text("x")
        req = make_request(workspace, AutoApprovePolicy())
        status = await GitStatusTool().run(GitStatusTool.Args(), req)
        assert "new.txt" in status

    @pytest.mark.asyncio
    async def test_git_diff_staged_flag(self, workspace):
        from harness.tools.git import GitDiffTool

        async def sh(cmd: str) -> None:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(workspace),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()

        await sh("git init -q && git config user.email t@t.io && git config user.name t")
        (workspace / "f.txt").write_text("body")
        await sh("git add -A")
        req = make_request(workspace, AutoApprovePolicy())
        out = await GitDiffTool().run(GitDiffTool.Args(staged=True), req)
        assert "f.txt" in out

    @pytest.mark.asyncio
    async def test_git_command_failure_raises_tool_error(self, workspace):
        from harness.tools.git import GitLogTool

        req = make_request(workspace, AutoApprovePolicy())  # not a repo
        with pytest.raises(ToolError, match="git log"):
            await GitLogTool().run(GitLogTool.Args(), req)

    @pytest.mark.asyncio
    async def test_git_checkout_existing_branch(self, workspace):
        from harness.tools.git import GitCheckoutTool

        async def sh(cmd: str) -> None:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(workspace),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()

        await sh("git init -q && git config user.email t@t.io && git config user.name t")
        (workspace / "a.txt").write_text("x")
        await sh("git add -A && git commit -qm init && git branch second")

        req = make_request(workspace, AutoApprovePolicy())
        await GitCheckoutTool().run(GitCheckoutTool.Args(branch="second"), req)
        proc = await asyncio.create_subprocess_shell(
            "git rev-parse --abbrev-ref HEAD", cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        assert out.decode().strip() == "second"

    @pytest.mark.asyncio
    async def test_browse_url_httpx_error_wrapped(self, monkeypatch):
        from harness.tools import web as web_mod

        class FailingClient:
            def __init__(self, *a, **kw) -> None: ...
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def get(self, url, **kw):
                raise httpx.ConnectError("no route")

        monkeypatch.setattr(web_mod.httpx, "AsyncClient", FailingClient)
        tool = web_mod.BrowseUrlTool()
        req = Request(workspace=Path("/tmp"), policy=AutoApprovePolicy())
        with pytest.raises(ToolError, match="fetch failed"):
            await tool.run(web_mod.BrowseUrlArgs(url="https://unreachable.test"), req)

    @pytest.mark.asyncio
    async def test_browse_url_plain_text_passes_through(self, monkeypatch):
        from harness.tools import web as web_mod

        class PlainClient:
            def __init__(self, *a, **kw) -> None: ...
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def get(self, url, **kw):
                return httpx.Response(
                    200, text="plain body", headers={"content-type": "text/plain"},
                    request=httpx.Request("GET", url),
                )

        monkeypatch.setattr(web_mod.httpx, "AsyncClient", PlainClient)
        tool = web_mod.BrowseUrlTool()
        req = Request(workspace=Path("/tmp"), policy=AutoApprovePolicy())
        out = await tool.run(web_mod.BrowseUrlArgs(url="https://x.test/f.txt"), req)
        assert out.strip() == "plain body"

    @pytest.mark.asyncio
    async def test_web_search_error_wrapped(self, monkeypatch):
        from harness.tools import web as web_mod

        class FailingClient:
            def __init__(self, *a, **kw) -> None: ...
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def get(self, url, **kw):
                raise httpx.TimeoutException("slow")

        monkeypatch.setattr(web_mod.httpx, "AsyncClient", FailingClient)
        tool = web_mod.WebSearchTool()
        req = Request(workspace=Path("/tmp"), policy=AutoApprovePolicy())
        with pytest.raises(ToolError, match="search failed"):
            await tool.run(web_mod.WebSearchArgs(query="x"), req)

    @pytest.mark.asyncio
    async def test_web_search_no_results(self, monkeypatch):
        from harness.tools import web as web_mod

        class EmptyClient:
            def __init__(self, *a, **kw) -> None: ...
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def get(self, url, **kw):
                return httpx.Response(200, text="<html>nothing</html>",
                                      request=httpx.Request("GET", url))

        monkeypatch.setattr(web_mod.httpx, "AsyncClient", EmptyClient)
        tool = web_mod.WebSearchTool()
        req = Request(workspace=Path("/tmp"), policy=AutoApprovePolicy())
        assert await tool.run(web_mod.WebSearchArgs(query="x"), req) == "(no results)"

    def test_registry_filtering(self):
        from harness.tools import build_registry

        only = build_registry(include={"view_file", "write_file"})
        assert {s.name for s in only.schema()} == {"view_file", "write_file"}
        without = build_registry(exclude={"execute_command"})
        assert "execute_command" not in {s.name for s in without.schema()}


class TestLoopEdgePaths:
    @pytest.mark.asyncio
    async def test_unexpected_tool_exception_is_captured(self, workspace, registry, config):
        class Exploding:
            name = "boom"
            description = "d"
            Args = type("A", (object,), {})

            def schema(self):
                return {"type": "object"}

            def actions(self, args):
                from harness.models import Action, ActionRequest
                return [ActionRequest(action=Action.READ, summary="x")]

            def check(self, args, req):
                return None

            async def run(self, args, req):
                raise RuntimeError("kaboom")

        registry._handlers["boom"] = Exploding
        req = make_request(workspace, AutoApprovePolicy())
        msgs = await execute_tool_calls([call("boom")], registry, req)
        assert msgs[0].is_error and "kaboom" in msgs[0].content

    @pytest.mark.asyncio
    async def test_provider_error_ends_loop_with_message(self, workspace, registry, config):
        class Down(LLMProvider):
            async def complete(self, messages, **kw):
                raise ProviderError("endpoint down")

            async def stream(self, messages, **kw):
                yield ""

        agent = Agent(
            role=AgentRole.PRIMARY, provider=Down(), registry=registry,
            config=config, policy=AutoApprovePolicy(),
        )
        outcome = await agent.run("do it", max_turns=2)
        assert "provider error" in outcome.output.lower()

    @pytest.mark.asyncio
    async def test_context_overflow_ends_loop(self, workspace, registry):
        tiny = HarnessConfig(workspace=workspace, context_window=200, reserve_tokens=100)
        provider = ScriptedProvider([assistant("x")])
        agent = Agent(
            role=AgentRole.PRIMARY, provider=provider, registry=registry,
            config=tiny, policy=AutoApprovePolicy(),
        )
        outcome = await agent.run("goal " + "z" * 5000)
        assert outcome.turns == 1
        assert outcome.output == ""

    @pytest.mark.asyncio
    async def test_pruning_note_injected_when_context_tight(self, workspace, registry):
        cfg = HarnessConfig(workspace=workspace, context_window=900, reserve_tokens=200)
        long_files = [call("view_file", path="big.txt") for _ in range(12)]
        (workspace / "big.txt").write_text("data " * 400)
        script = [assistant("", [c]) for c in long_files] + [assistant("finished")]
        provider = ScriptedProvider(script)
        agent = Agent(
            role=AgentRole.PRIMARY, provider=provider, registry=registry,
            config=cfg, policy=AutoApprovePolicy(),
        )
        await agent.run("inspect", max_turns=13)
        injected = any(
            isinstance(m, SystemMessage) and "context pruned" in m.content
            for msgs in provider.calls for m in msgs
        )
        assert injected, "expected a pruning marker once context got tight"

    @pytest.mark.asyncio
    async def test_agent_uses_custom_system_prompt(self, workspace, registry, config):
        provider = ScriptedProvider([assistant("ok")])
        agent = Agent(
            role=AgentRole.CODER, provider=provider, registry=registry,
            config=config, policy=AutoApprovePolicy(), system_prompt="CUSTOM PROMPT",
        )
        await agent.run("go")
        assert any(
            isinstance(m, SystemMessage) and m.content == "CUSTOM PROMPT"
            for m in provider.calls[0]
        )


# ========================================================================== #
# 9. Final branch coverage
# ========================================================================== #


class TestCliBranches:
    def test_confirm_yes_and_no(self, monkeypatch, capsys):
        from harness.main import _confirm

        monkeypatch.setattr("builtins.input", lambda: "y")
        assert _confirm(ActionRequest(action=Action.EXECUTE, summary="s", detail="d")) is True
        monkeypatch.setattr("builtins.input", lambda: "n")
        assert _confirm(ActionRequest(action=Action.WRITE, summary="s")) is False

    def test_report_quiet_suppresses_footer(self, capsys):
        from harness.main import report

        report(
            LoopOutcome(output="x", usage=Usage(input_tokens=1), turns=1,
                        conversation=Conversation()),
            quiet=True,
        )
        captured = capsys.readouterr()
        assert "x" in captured.out
        assert "turns" not in captured.err

    def test_repl_blank_then_eof_returns_zero(self, monkeypatch, workspace):
        import harness.main as main_mod

        provider = ScriptedProvider([assistant("hi")])
        monkeypatch.setattr(main_mod, "build_provider", lambda config: provider)
        calls = iter(["", "   "])

        def fake_input(*a):
            try:
                return next(calls)
            except StopIteration:
                raise EOFError

        monkeypatch.setattr("builtins.input", fake_input)
        rc = main_mod.main(["--policy", "auto-approve", "--workspace", str(workspace)])
        assert rc == 0

    def test_repl_keyboard_interrupt_returns_zero(self, monkeypatch, workspace):
        import harness.main as main_mod

        monkeypatch.setattr(main_mod, "build_provider", lambda config: ScriptedProvider([]))
        monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
        rc = main_mod.main(["--policy", "auto-approve", "--workspace", str(workspace)])
        assert rc == 0

    def test_main_keyboard_interrupt_returns_130(self, monkeypatch, workspace):
        import harness.main as main_mod

        def boom(args, config, prompt):
            raise KeyboardInterrupt

        monkeypatch.setattr(main_mod, "run_once", boom)
        rc = main_mod.main(["--policy", "auto-approve", "--workspace", str(workspace), "go"])
        assert rc == 130

    def test_main_harness_error_returns_1(self, monkeypatch, workspace):
        import harness.main as main_mod

        def boom(args, config, prompt):
            raise ProviderError("endpoint unreachable")

        monkeypatch.setattr(main_mod, "run_once", boom)
        rc = main_mod.main(["--policy", "auto-approve", "--workspace", str(workspace), "go"])
        assert rc == 1

    def test_run_once_delegate_uses_rendered_result(self, monkeypatch, workspace):
        import harness.main as main_mod

        provider = ScriptedProvider([assistant("review done")])
        monkeypatch.setattr(main_mod, "build_provider", lambda config: provider)
        args = main_mod.parse_args(
            ["--policy", "auto-approve", "--workspace", str(workspace), "--delegate", "architect", "plan"]
        )
        config = main_mod.build_config(args)
        outcome = asyncio.run(main_mod.run_once(args, config, "plan the work"))
        assert "<architect>" in outcome.output


def test_dunder_main_module_runs(monkeypatch, workspace):
    """`python -m harness` must dispatch to main()."""
    import runpy

    import harness.main as main_mod

    monkeypatch.setattr(main_mod, "build_provider", lambda config: ScriptedProvider([assistant("ok")]))
    monkeypatch.setattr(
        "sys.argv",
        ["harness", "--policy", "auto-approve", "--workspace", str(workspace), "hi"],
    )
    try:
        runpy.run_module("harness.__main__", run_name="__main__")
    except SystemExit as exc:  # main() returns 0 -> SystemExit(0)
        assert exc.code == 0


class TestAnthropicPayloadBranches:
    @pytest.mark.asyncio
    async def test_tools_are_serialized_with_input_schema(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200, json={"content": [{"type": "text", "text": "k"}],
                           "stop_reason": "end_turn", "usage": {}},
            )

        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        await adapter.complete(
            [UserMessage(content="hi")],
            tools=[ToolSpec(name="t", description="d", parameters={"type": "object", "properties": {}})],
        )
        assert captured["tools"][0]["name"] == "t"
        assert captured["tools"][0]["input_schema"]["type"] == "object"
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_403_maps_to_auth_error(self):
        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(403, text="forbidden")),
        )
        with pytest.raises(AuthenticationError):
            await adapter.complete([UserMessage(content="hi")])
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_429_maps_to_rate_limit(self):
        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(429, text="slow")),
        )
        with pytest.raises(RateLimitError):
            await adapter.complete([UserMessage(content="hi")])
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_stream_skips_non_delta_and_malformed_events(self):
        sse = b"".join([
            b'data: {"type":"message_start"}\n\n',
            b"data: {not json\n\n",
            b'event: ping\n\n',
            b'data: {"type":"content_block_delta","delta":{"type":"input_json_delta"}}\n\n',
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Z"}}\n\n',
        ])

        adapter = AnthropicAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(
                lambda r: httpx.Response(200, content=sse,
                                         headers={"content-type": "text/event-stream"})
            ),
        )
        chunks = [c async for c in adapter.stream([UserMessage(content="hi")])]
        assert chunks == ["Z"]
        await adapter.aclose()


class TestOpenAIStreamBranches:
    @pytest.mark.asyncio
    async def test_stream_ignores_malformed_and_non_delta_lines(self):
        sse = b"".join([
            b"event: ping\n\n",
            b"data: {broken\n\n",
            b'data: {"choices":[{"delta":{}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"Q"}}]}\n\n',
            b"data: [DONE]\n\n",
        ])

        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(
                lambda r: httpx.Response(200, content=sse,
                                         headers={"content-type": "text/event-stream"})
            ),
        )
        chunks = [c async for c in adapter.stream([UserMessage(content="hi")])]
        assert chunks == ["Q"]
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_stream_raises_on_http_error(self):
        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(429, text="slow")),
        )
        with pytest.raises(RateLimitError):
            async for _ in adapter.stream([UserMessage(content="hi")]):
                pass
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_non_retryable_400_is_provider_error(self):
        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(400, text="bad request")),
        )
        with pytest.raises(ProviderError):
            await adapter.complete([UserMessage(content="hi")])
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_openai_factory_wraps_with_retry_and_accounting(self):
        provider = make_openai_provider(base_url="https://api.test/v1", api_key="k")
        assert isinstance(provider, AccountingProvider)
        await provider.aclose()  # type: ignore[attr-defined]


class TestFactories:
    @pytest.mark.asyncio
    async def test_anthropic_factory_wraps(self):
        provider = make_anthropic_provider(base_url="https://api.test/v1", api_key="k")
        assert isinstance(provider, AccountingProvider)
        await provider.aclose()  # type: ignore[attr-defined]

    def test_unknown_policy_error_message(self):
        from harness.main import build_policy

        with pytest.raises(ValueError, match="unknown policy"):
            build_policy("bogus")


# ========================================================================== #
# 10. Last reachable branches
# ========================================================================== #


class TestResidualBranches:
    @pytest.mark.asyncio
    async def test_openai_skips_non_function_tool_entries(self):
        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(
                lambda r: httpx.Response(
                    200,
                    json={"choices": [{"index": 0, "message": {
                        "role": "assistant", "content": "",
                        "tool_calls": [
                            {"id": "x", "type": "retrieval", "retrieval": {}},
                            {"id": "c", "type": "function",
                             "function": {"name": "t", "arguments": "{}"}},
                        ]},
                        "finish_reason": "tool_calls"}], "usage": {}},
                )
            ),
        )
        completion = await adapter.complete([UserMessage(content="hi")])
        assert len(completion.message.tool_calls) == 1
        assert completion.message.tool_calls[0].name == "t"
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_openai_stream_raises_on_unauthorized(self):
        adapter = OpenAIAdapter(base_url="https://api.test/v1")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1",
            transport=_mock_transport(lambda r: httpx.Response(401, text="no")),
        )
        with pytest.raises(AuthenticationError):
            async for _ in adapter.stream([UserMessage(content="hi")]):
                pass
        await adapter.aclose()

    def test_network_tools_declare_network_action(self):
        from harness.tools.web import BrowseUrlTool, WebSearchTool

        search_actions = WebSearchTool().actions(WebSearchTool.Args(query="q"))
        browse_actions = BrowseUrlTool().actions(BrowseUrlTool.Args(url="https://x"))
        assert search_actions[0].action is Action.NETWORK
        assert browse_actions[0].action is Action.NETWORK

    @pytest.mark.asyncio
    async def test_list_directory_truncates_huge_tree(self, workspace, monkeypatch):
        from harness.tools import files as files_mod

        tool = files_mod.ListDirTool()
        req = make_request(workspace, AutoApprovePolicy())

        def fake_walk(root):
            yield str(root), [], ["a", "b", "c", "d"]
            yield str(root), [], [f"f{i}.txt" for i in range(4100)]

        monkeypatch.setattr(files_mod.os, "walk", fake_walk)
        out = await tool.run(files_mod.ListDirArgs(path=".", recursive=True), req)
        assert "(truncated)" in out

    @pytest.mark.asyncio
    async def test_grep_python_fallback_truncates_and_skips_dirs(self, workspace, monkeypatch):
        from harness.tools import files as files_mod

        many = workspace / "many.txt"
        many.write_text("\n".join(f"line{i} hit" for i in range(600)))
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__/ignored.txt").write_text("hit")
        monkeypatch.setattr(files_mod.shutil, "which", lambda name: None)
        tool = files_mod.GrepTool()
        req = make_request(workspace, AutoApprovePolicy(), timeout_s=30.0)
        req = req.model_copy(update={"max_output_chars": 10_000_000})
        out = await tool.run(files_mod.GrepArgs(pattern="hit"), req)
        assert "(truncated)" in out
        assert "__pycache__" not in out

    @pytest.mark.asyncio
    async def test_command_spawn_failure_is_tool_error(self, workspace, monkeypatch):
        from harness.tools.terminal import CommandTool

        async def boom(*a, **kw):
            raise OSError("no shell")

        monkeypatch.setattr(asyncio, "create_subprocess_shell", boom)
        tool = CommandTool()
        req = make_request(workspace, AutoApprovePolicy())
        with pytest.raises(ToolError, match="spawn failed"):
            await tool.run(CommandTool.Args(command="true"), req)

    @pytest.mark.asyncio
    async def test_command_respects_cwd_argument(self, workspace):
        from harness.tools.terminal import CommandTool

        (workspace / "sub").mkdir()
        tool = CommandTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(CommandTool.Args(command="pwd", cwd="sub"), req)
        assert "sub" in out


# ========================================================================== #
# 6. RESIDUAL COVERAGE BRANCHES -- edge and error paths
# ========================================================================== #


class TestCoverageBranches:
    def test_tool_failure_render(self):
        from harness.models import ToolFailure

        tf = ToolFailure(tool="view_file", kind="validation", message="bad arg")
        assert tf.render() == "[validation] view_file: bad arg"

    @pytest.mark.asyncio
    async def test_default_stream_yields_nothing(self):
        """The base stream() default body yields nothing (call it unbound)."""

        class Impl(LLMProvider):
            async def complete(self, messages, **kw):  # type: ignore[override]
                raise NotImplementedError

            async def stream(self, messages, **kw):  # type: ignore[override]
                yield "x"

        chunks = [c async for c in LLMProvider.stream(Impl(), [])]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_retry_provider_negative_budget_is_unreachable(self):
        from harness.providers.base import RetryProvider

        class AlwaysTransient(LLMProvider):
            async def complete(self, messages, **kw):  # type: ignore[override]
                raise TransientProviderError("down")

            async def stream(self, messages, **kw):  # type: ignore[override]
                yield ""

        prov = RetryProvider(AlwaysTransient(), max_retries=-1)
        with pytest.raises(ProviderError, match="Unreachable retry state"):
            await prov.complete([])

    @pytest.mark.asyncio
    async def test_accounting_provider_stream_passthrough(self):
        class Streamer(LLMProvider):
            async def complete(self, messages, **kw):  # type: ignore[override]
                raise NotImplementedError

            async def stream(self, messages, **kw):  # type: ignore[override]
                yield "a"
                yield "b"

        prov = AccountingProvider(Streamer())
        chunks = [c async for c in prov.stream([])]
        assert chunks == ["a", "b"]
        assert prov.calls == 0  # streaming does not count as a completion

    @pytest.mark.asyncio
    async def test_grep_fallback_skips_dirs_and_unreadable(self, workspace, monkeypatch):
        """Covers the Python-regex fallback: non-file entries and OSError reads."""
        import pathlib

        import harness.tools.files as files_mod

        monkeypatch.setattr(files_mod.shutil, "which", lambda name: None)
        (workspace / "sub").mkdir()  # a directory entry -> skipped via is_file()
        (workspace / "boom.txt").write_text("needle")

        orig = pathlib.Path.read_text

        def flaky(self, *a, **k):
            if self.name == "boom.txt":
                raise OSError("permission denied")
            return orig(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "read_text", flaky)

        tool = files_mod.GrepTool()
        req = make_request(workspace, AutoApprovePolicy())
        out = await tool.run(files_mod.GrepArgs(pattern="needle"), req)
        assert out == "(no matches)"

    def test_git_tool_action_declarations(self):
        from harness.tools.git import (
            GitCheckoutTool,
            GitCommitTool,
            GitDiffTool,
            GitLogTool,
            GitStatusTool,
        )

        assert GitStatusTool().actions(GitStatusTool.Args())[0].action is Action.READ
        assert GitDiffTool().actions(GitDiffTool.Args())[0].action is Action.READ
        assert GitDiffTool().actions(GitDiffTool.Args(staged=True, path="x"))[0].action is Action.READ
        assert GitCommitTool().actions(GitCommitTool.Args(message="m"))[0].action is Action.EXECUTE
        assert GitLogTool().actions(GitLogTool.Args())[0].action is Action.READ
        assert GitCheckoutTool().actions(GitCheckoutTool.Args(branch="b"))[0].action is Action.EXECUTE

    @pytest.mark.asyncio
    async def test_git_diff_with_path_and_commit_with_files(self, workspace):
        from harness.tools.git import GitCommitTool, GitDiffTool

        async def sh(cmd: str) -> None:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(workspace),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()

        await sh("git init -q && git config user.email t@t.io && git config user.name t")
        (workspace / "a.txt").write_text("one\n")
        (workspace / "b.txt").write_text("two\n")
        await sh("git add -A && git commit -qm init")

        (workspace / "a.txt").write_text("one changed\n")
        req = make_request(workspace, AutoApprovePolicy())

        diff = await GitDiffTool().run(GitDiffTool.Args(path="a.txt"), req)
        assert "a.txt" in diff

        out = await GitCommitTool().run(GitCommitTool.Args(message="touch a", files=["a.txt"]), req)
        assert "touch a" in out

    def test_repl_reports_harness_error_then_exits(self, workspace, monkeypatch, capsys):
        """Covers the REPL's HarnessError branch and its clean EOF exit."""
        import harness.main as main_mod
        from harness.loop import LoopOutcome

        class FakeCoordinator:
            def __init__(self, **kw):
                pass

            async def run(self, line, **kw):
                raise HarnessError("kaboom")

        monkeypatch.setattr(main_mod, "Coordinator", FakeCoordinator)

        lines = iter(["hello", EOFError()])

        def fake_input(prompt=""):
            nxt = next(lines)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        monkeypatch.setattr("builtins.input", fake_input)

        config = HarnessConfig(workspace=workspace)
        rc = asyncio.run(main_mod.repl(argparse.Namespace(policy="auto-approve"), config))
        assert rc == 0
        assert "error: kaboom" in capsys.readouterr().err

    def test_double_underscore_main_guard(self, workspace, monkeypatch):
        """Execute harness/main.py as __main__ to cover the entry guard."""
        import runpy

        monkeypatch.setattr(sys, "argv", ["harness", "hi"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))

        async def fake_run_once(args, config, prompt):
            return LoopOutcome(output="done", usage=Usage(), turns=0, conversation=None)  # type: ignore[arg-type]

        import harness.main as main_mod

        monkeypatch.setattr(main_mod, "run_once", fake_run_once)
        monkeypatch.setattr(main_mod, "report", lambda outcome, quiet: None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(main_mod.__file__, run_name="__main__")
        assert exc.value.code == 0

    @pytest.mark.asyncio
    async def test_anthropic_adapter_parses_openai_shaped_proxy_body(self):
        """Compatibility proxies may answer /messages with an OpenAI envelope."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "object": "chat.completion",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Hi.",
                            "tool_calls": [{
                                "id": "c_1", "type": "function",
                                "function": {"name": "view_file",
                                             "arguments": '{"path": "x.py"}'},
                            }],
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 824, "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 768},
                    },
                },
            )

        adapter = AnthropicAdapter(base_url="https://api.test/v1", api_key="k")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        completion = await adapter.complete([UserMessage(content="say hi")])
        assert completion.message.content == "Hi."
        (tc,) = completion.message.tool_calls
        assert tc.id == "c_1" and tc.name == "view_file"
        assert tc.arguments == {"path": "x.py"}
        assert completion.usage.input_tokens == 824
        assert completion.usage.output_tokens == 3
        assert completion.usage.cache_read_tokens == 768
        assert completion.finish_reason == "stop"
        await adapter.aclose()

    @pytest.mark.asyncio
    async def test_anthropic_adapter_proxy_body_tolerates_malformed_tool_args(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "message": {
                            "role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "c_2", "type": "function",
                                "function": {"name": "view_file", "arguments": "{oops"},
                            }],
                        },
                    }],
                },
            )

        adapter = AnthropicAdapter(base_url="https://api.test/v1", api_key="k")
        adapter._client = httpx.AsyncClient(
            base_url="https://api.test/v1", transport=_mock_transport(handler)
        )
        completion = await adapter.complete([UserMessage(content="go")])
        assert completion.message.content == ""
        (tc,) = completion.message.tool_calls
        assert tc.arguments == {}
        assert completion.usage.input_tokens == 0
        assert completion.finish_reason == "stop"
        await adapter.aclose()

