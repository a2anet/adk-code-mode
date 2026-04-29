# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types

from adk_code_mode.tools import namespacing
from adk_code_mode.tools.dispatcher import Dispatcher
from adk_code_mode.tools.normaliser import ResolvedTool


@dataclass
class _FakeAgent:
    canonical_before_tool_callbacks: list[Any] = field(default_factory=list)
    canonical_after_tool_callbacks: list[Any] = field(default_factory=list)
    canonical_on_tool_error_callbacks: list[Any] = field(default_factory=list)


@dataclass
class _FakeInvCtx:
    agent: Any
    plugin_manager: PluginManager
    artifact_service: InMemoryArtifactService
    session: Session
    app_name: str = "a"
    user_id: str = "u"


def _make_ctx(agent: Any = None) -> _FakeInvCtx:
    return _FakeInvCtx(
        agent=agent or _FakeAgent(),
        plugin_manager=PluginManager(),
        artifact_service=InMemoryArtifactService(),
        session=Session(
            id="s", app_name="a", user_id="u", state={}, events=[], last_update_time=0.0
        ),
    )


class _EchoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echo the args back.")
        self.seen_args: dict[str, Any] | None = None

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        self.seen_args = args
        return {"echoed": args}


class _RaisesTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="boom", description="Always raises.")

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        raise RuntimeError("kaboom")


class _ConfirmationTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="confirm", description="Requests confirmation.")

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        tool_context.request_confirmation(hint="approve me")
        return {"needs_confirmation": True}


class _StateAndArtifactTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="stateful", description="Writes state and artifact.")

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        tool_context.state["counter"] = 1
        await tool_context.save_artifact(
            "note.txt",
            genai_types.Part(inline_data=genai_types.Blob(data=b"hello", mime_type="text/plain")),
        )
        return {"ok": True}


class _SleepTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="sleepy", description="Sleeps for a bit.")

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        await asyncio.sleep(float(args["delay"]))
        return {"slept": args["delay"]}


class _LongRunningPendingTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="pending", description="Long-running, returns no immediate response.")
        self.is_long_running = True

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        return None


class _LongRunningEagerTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="eager_lr", description="Long-running, returns an immediate response."
        )
        self.is_long_running = True

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        return {"ok": True}


class _NestedWriteTool(BaseTool):
    def __init__(self, name: str, key: str) -> None:
        super().__init__(name=name, description=f"Writes scratch.{key}.")
        self._key = key

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        scratch = dict(tool_context.state.get("scratch", {}))
        scratch[self._key] = args["value"]
        tool_context.state["scratch"] = scratch
        return {"set": self._key}


def _registry(tool: BaseTool) -> namespacing.Registry:
    ns = namespacing.build([ResolvedTool(tool=tool, toolset=None)])
    return namespacing.Registry(ns)


async def test_dispatch_success_propagates_value() -> None:
    tool = _EchoTool()
    reg = _registry(tool)
    ctx = _make_ctx()
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("echo", {"x": 1})
    assert result.ok is True
    assert result.value == {"echoed": {"x": 1}}


async def test_dispatch_unknown_tool_returns_error() -> None:
    reg = _registry(_EchoTool())
    ctx = _make_ctx()
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("does_not_exist", {})
    assert result.ok is False
    assert result.error_type == "ToolNotFound"


async def test_dispatch_tool_error_bubbles() -> None:
    reg = _registry(_RaisesTool())
    ctx = _make_ctx()
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("boom", {})
    assert result.ok is False
    assert result.error_type == "RuntimeError"
    assert result.error_message == "kaboom"


async def test_before_callback_short_circuits_tool() -> None:
    tool = _EchoTool()
    seen: dict[str, Any] = {}

    def before(tool: BaseTool, args: dict[str, Any], tool_context: Any) -> dict[str, Any]:
        seen["called"] = True
        return {"replaced": True}

    agent = _FakeAgent(canonical_before_tool_callbacks=[before])
    reg = _registry(tool)
    ctx = _make_ctx(agent)
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("echo", {"x": 1})
    assert result.ok is True
    assert result.value == {"replaced": True}
    assert tool.seen_args is None
    assert seen == {"called": True}


async def test_after_callback_rewrites_response() -> None:
    def after(tool: BaseTool, args: dict[str, Any], tool_context: Any, tool_response: Any) -> Any:
        return {"wrapped": tool_response}

    agent = _FakeAgent(canonical_after_tool_callbacks=[after])
    reg = _registry(_EchoTool())
    ctx = _make_ctx(agent)
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("echo", {"x": 1})
    assert result.ok is True
    assert result.value == {"wrapped": {"echoed": {"x": 1}}}


async def test_on_error_callback_can_recover() -> None:
    def on_error(tool: BaseTool, args: dict[str, Any], tool_context: Any, error: Exception) -> Any:
        return {"recovered": str(error)}

    agent = _FakeAgent(canonical_on_tool_error_callbacks=[on_error])
    reg = _registry(_RaisesTool())
    ctx = _make_ctx(agent)
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("boom", {})
    assert result.ok is True
    assert result.value == {"recovered": "kaboom"}


async def test_dispatch_rejects_unsupported_tool_actions() -> None:
    reg = _registry(_ConfirmationTool())
    ctx = _make_ctx()
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("confirm", {})
    assert result.ok is False
    assert result.error_type == "UnsupportedToolActionError"
    assert "tool confirmations" in (result.error_message or "")


async def test_dispatch_preserves_state_and_artifact_side_effects() -> None:
    reg = _registry(_StateAndArtifactTool())
    ctx = _make_ctx()
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("stateful", {})
    assert result.ok is True
    assert result.value == {"ok": True}
    assert ctx.session.state["counter"] == 1
    keys = await ctx.artifact_service.list_artifact_keys(app_name="a", user_id="u", session_id="s")
    assert "note.txt" in keys


async def test_dispatch_runs_tool_calls_concurrently() -> None:
    reg = _registry(_SleepTool())
    ctx = _make_ctx()
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]

    start = asyncio.get_running_loop().time()
    first, second = await asyncio.gather(
        d.dispatch("sleepy", {"delay": 0.3}),
        d.dispatch("sleepy", {"delay": 0.3}),
    )
    elapsed = asyncio.get_running_loop().time() - start

    assert first.ok is True
    assert second.ok is True
    assert elapsed < 0.55


async def test_dispatch_rejects_long_running_tools_with_no_response() -> None:
    reg = _registry(_LongRunningPendingTool())
    ctx = _make_ctx()
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("pending", {})
    assert result.ok is False
    assert result.error_type == "UnsupportedToolActionError"
    assert "long-running" in (result.error_message or "")


async def test_dispatch_allows_long_running_tools_with_immediate_response() -> None:
    reg = _registry(_LongRunningEagerTool())
    ctx = _make_ctx()
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]
    result = await d.dispatch("eager_lr", {})
    assert result.ok is True
    assert result.value == {"ok": True}


async def test_dispatch_clamps_per_call_timeout_to_host_cap() -> None:
    # Sandbox code can call _rpc_client.call with any timeout, but the host
    # configured cap (per_tool_timeout_seconds) must remain a true ceiling.
    reg = _registry(_SleepTool())
    ctx = _make_ctx()
    d = Dispatcher(
        invocation_context=ctx,  # type: ignore[arg-type]
        registry=reg,
        execution_id="e1",
        per_tool_timeout_seconds=0.05,
    )
    result = await d.dispatch("sleepy", {"delay": 0.5}, timeout=999.0)
    assert result.ok is False
    assert result.error_type == "TimeoutError"
    assert "0.05" in (result.error_message or "")


async def test_dispatch_uses_per_call_timeout_when_below_host_cap() -> None:
    reg = _registry(_SleepTool())
    ctx = _make_ctx()
    d = Dispatcher(
        invocation_context=ctx,  # type: ignore[arg-type]
        registry=reg,
        execution_id="e1",
        per_tool_timeout_seconds=10.0,
    )
    result = await d.dispatch("sleepy", {"delay": 0.5}, timeout=0.05)
    assert result.ok is False
    assert result.error_type == "TimeoutError"
    assert "0.05" in (result.error_message or "")


async def test_concurrent_nested_state_writes_are_deep_merged() -> None:
    a = _NestedWriteTool(name="set_a", key="a")
    b = _NestedWriteTool(name="set_b", key="b")
    ns = namespacing.build([ResolvedTool(tool=a, toolset=None), ResolvedTool(tool=b, toolset=None)])
    reg = namespacing.Registry(ns)
    ctx = _make_ctx()
    d = Dispatcher(invocation_context=ctx, registry=reg, execution_id="e1")  # type: ignore[arg-type]

    await asyncio.gather(
        d.dispatch("set_a", {"value": 1}),
        d.dispatch("set_b", {"value": 2}),
    )

    assert ctx.session.state["scratch"] == {"a": 1, "b": 2}
