# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""In-process end-to-end tests using ``FakeRuntime`` (no Docker)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from adk_code_mode.runtime.base import SandboxConnectionError, SandboxResult, SandboxSession
from adk_code_mode.runtime.protocol import PROTOCOL_VERSION, DoneFrame, Frame, ReadyFrame
from adk_code_mode.tool import ExecuteCodeTool, ProtocolVersionMismatchError

from ._fake_runtime import FakeRuntime


class _EchoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echo back args.")

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        return {"echoed": args.get("message", "")}


class _OptionalArgTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="optional_lookup", description="Optional-arg tool.")

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": [],
            },
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        return {"has_record_id": "record_id" in args, "record_id": args.get("record_id")}


class _SlowTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="slow", description="Sleep for a while.")

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={"type": "object"},
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        await asyncio.sleep(2)
        return {"ok": True}


class _DynamicTool(BaseTool):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, description=f"dynamic-{name}")

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={"type": "object"},
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        return self.name


class _DynamicToolset(BaseToolset):
    async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[BaseTool]:
        which = "alpha"
        if readonly_context is not None:
            which = str(readonly_context.session.state.get("which", which))
        return [_DynamicTool(which)]


def _make_invocation_context(
    artifact_service: InMemoryArtifactService,
    session: Session,
) -> InvocationContext:
    agent = MagicMock()
    agent.canonical_before_tool_callbacks = []
    agent.canonical_after_tool_callbacks = []
    agent.canonical_on_tool_error_callbacks = []

    ctx = MagicMock(spec=InvocationContext)
    ctx.agent = agent
    ctx.plugin_manager = PluginManager()
    ctx.artifact_service = artifact_service
    ctx.session = session
    ctx.app_name = "test-app"
    ctx.user_id = "u1"
    ctx.credential_service = None
    # Tests call ``run_async`` directly without a real model turn, so there's no
    # cached resolution to consume; the tool falls back to a fresh resolve each
    # call (which lets dynamic toolsets see updated state).
    ctx.invocation_id = "test-inv-1"
    return ctx


def _tool_context(ctx: InvocationContext, *, call_id: str = "call-1") -> ToolContext:
    """Wrap a (fake) ``InvocationContext`` the way ADK wraps a real tool call.

    ``call_id`` doubles as the execution id used for stdout/stderr overflow
    artifact naming, mirroring the old ``CodeExecutionInput.execution_id``.
    """
    return ToolContext(invocation_context=ctx, function_call_id=call_id)


async def _run(tool: ExecuteCodeTool, ctx: InvocationContext, code: str, *, call_id: str) -> Any:
    return await tool.run_async(
        args={"code": code}, tool_context=_tool_context(ctx, call_id=call_id)
    )


@pytest.mark.asyncio
async def test_sandbox_echoes_and_runs_tool() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s1",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(
        tools=[_EchoTool()],
        backend=FakeRuntime(),
        max_output_chars=10_000,
    )
    code = "from tools import echo\nr = echo(message='hello from sandbox')\nprint('ECHO:', r)\n"

    result = await _run(tool, ctx, code, call_id="run-1")
    assert "ECHO: {'echoed': 'hello from sandbox'}" in result["stdout"]
    assert result["stderr"].strip() == ""


@pytest.mark.asyncio
async def test_artifact_helpers_save_list_and_load_json() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s2",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)
    code = (
        "import json\n"
        "from tools import save_artifact, list_artifacts, load_artifact\n"
        "save_artifact(\n"
        "    filename='report.json',\n"
        "    content=json.dumps({'ok': True}),\n"
        "    mime_type='application/json',\n"
        ")\n"
        "print('items', list_artifacts())\n"
        "loaded = load_artifact(filename='report.json')\n"
        "print('loaded', json.loads(loaded['data']))\n"
    )

    result = await _run(tool, ctx, code, call_id="run-ws")
    assert "report.json" in result["stdout"]
    assert "loaded {'ok': True}" in result["stdout"]
    keys = await artifact_service.list_artifact_keys(
        app_name="test-app", user_id="u1", session_id="s2"
    )
    assert "report.json" in keys


@pytest.mark.asyncio
async def test_workspace_outputs_are_collected_and_saved_as_artifacts() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s2-workspace",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)
    code = (
        "import os\n"
        "print('cwd', os.getcwd())\n"
        "with open('result.txt', 'w') as fh:\n"
        "    fh.write('created')\n"
    )

    result = await _run(tool, ctx, code, call_id="run-workspace")
    assert "cwd" in result["stdout"]
    assert result["output_files"] == ["result.txt"]
    keys = await artifact_service.list_artifact_keys(
        app_name="test-app", user_id="u1", session_id="s2-workspace"
    )
    assert "result.txt" in keys


@pytest.mark.asyncio
async def test_artifact_helpers_support_binary_content() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s2-binary",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)

    save = (
        "import base64\n"
        "from tools import save_artifact\n"
        "save_artifact(\n"
        "    filename='blob.bin',\n"
        "    content=base64.b64encode(b'\\x00\\x01\\x02').decode('ascii'),\n"
        "    mime_type='application/octet-stream',\n"
        ")\n"
    )
    await _run(tool, ctx, save, call_id="run-bin-1")

    load = (
        "import base64\n"
        "from tools import load_artifact\n"
        "blob = load_artifact(filename='blob.bin')\n"
        "print(blob['kind'], base64.b64decode(blob['data']), blob['mime_type'])\n"
    )
    result = await _run(tool, ctx, load, call_id="run-bin-2")

    assert "bytes b'\\x00\\x01\\x02' application/octet-stream" in result["stdout"]


@pytest.mark.asyncio
async def test_workspace_is_fresh_each_turn_while_artifact_helpers_persist() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s2-fresh",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)

    first = (
        "from tools import save_artifact\n"
        "with open('temp.txt', 'w') as fh:\n"
        "    fh.write('workspace only')\n"
        "save_artifact(\n"
        "    filename='persist.txt',\n"
        "    content='artifact persisted',\n"
        "    mime_type='text/plain',\n"
        ")\n"
    )
    # Distinct invocation_ids model two separate turns: the working directory
    # resets between turns, while Artifacts persist across them.
    ctx.invocation_id = "inv-fresh-1"
    await _run(tool, ctx, first, call_id="run-fresh-1")

    second = (
        "import os\n"
        "from tools import load_artifact\n"
        "print('workspace_has_temp', os.path.exists('temp.txt'))\n"
        "print('artifact_value', load_artifact(filename='persist.txt')['data'])\n"
    )
    ctx.invocation_id = "inv-fresh-2"
    result = await _run(tool, ctx, second, call_id="run-fresh-2")

    assert "workspace_has_temp False" in result["stdout"]
    assert "artifact_value artifact persisted" in result["stdout"]


@pytest.mark.asyncio
async def test_oversize_stdout_is_truncated_and_spilled() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s3",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=500)
    code = "print('x' * 5000)\n"

    result = await _run(tool, ctx, code, call_id="run-big")
    assert "Output exceeded 500 characters" in result["stdout"]
    assert "Full stdout was saved as an artifact" in result["stdout"]
    assert "load_artifact(filename='code_mode/stdout/run-big.txt')" in result["stdout"]
    keys = await artifact_service.list_artifact_keys(
        app_name="test-app", user_id="u1", session_id="s3"
    )
    assert "code_mode/stdout/run-big.txt" in keys


@pytest.mark.asyncio
async def test_timeout_terminates_hung_sandbox() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s4",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=500, timeout_seconds=1)

    result = await _run(tool, ctx, "while True:\n    pass\n", call_id="run-timeout")
    assert result["stdout"] == ""
    assert "Execution exceeded timeout of 1s" in result["stderr"]


@pytest.mark.asyncio
async def test_timeout_does_not_wait_for_in_flight_tool_call() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s4-tool-timeout",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(
        tools=[_SlowTool()], backend=FakeRuntime(), max_output_chars=500, timeout_seconds=1
    )

    start = time.monotonic()
    result = await _run(
        tool, ctx, "from tools import slow\nprint(slow())\n", call_id="run-tool-timeout"
    )
    elapsed = time.monotonic() - start

    assert elapsed < 1.8
    assert result["stdout"] == ""
    assert "Execution exceeded timeout of 1s" in result["stderr"]
    assert "Process exited with code" not in result["stderr"]


@pytest.mark.asyncio
async def test_nonzero_exit_code_is_reported() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s4-exit",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)

    result = await _run(tool, ctx, "import sys\nsys.exit(2)\n", call_id="run-exit")

    assert result["stdout"] == ""
    assert result["stderr"] == "Process exited with code 2."


@pytest.mark.asyncio
async def test_traceback_also_includes_exit_code_marker() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s4-traceback",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)

    result = await _run(tool, ctx, "raise ValueError('bad')\n", call_id="run-traceback")

    assert "ValueError: bad" in result["stderr"]
    assert result["stderr"].rstrip().endswith("Process exited with code 1.")


@pytest.mark.asyncio
async def test_syntax_error_also_includes_exit_code_marker() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s4-syntax",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)

    result = await _run(tool, ctx, "if True print('bad')\n", call_id="run-syntax")

    assert "SyntaxError" in result["stderr"]
    assert result["stderr"].rstrip().endswith("Process exited with code 1.")


@pytest.mark.asyncio
async def test_get_or_resolve_tools_re_resolves_dynamic_toolsets_per_invocation() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s5",
        app_name="test-app",
        user_id="u1",
        state={"which": "alpha"},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(tools=[_DynamicToolset()], backend=FakeRuntime())

    ctx.invocation_id = "test-inv-alpha"
    first_ns = await tool._get_or_resolve_tools(_tool_context(ctx))
    first = tool._prepare_tool_surface(first_ns)
    session.state["which"] = "beta"
    ctx.invocation_id = "test-inv-beta"
    second_ns = await tool._get_or_resolve_tools(_tool_context(ctx))
    second = tool._prepare_tool_surface(second_ns)

    assert "dynamic/alpha.py" in first.tools_map
    assert "dynamic/beta.py" in second.tools_map
    assert first is not second


@pytest.mark.asyncio
async def test_optional_stub_args_are_omitted_when_unspecified() -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="s6",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)

    tool = ExecuteCodeTool(
        tools=[_OptionalArgTool()], backend=FakeRuntime(), max_output_chars=10_000
    )
    code = "from tools import optional_lookup\nprint(optional_lookup())\n"

    result = await _run(tool, ctx, code, call_id="run-opt")
    assert "'has_record_id': False" in result["stdout"]
    assert result["stderr"].strip() == ""


# --- Turn-scoped session tests -------------------------------------------------


class _CountingBackend:
    """Wraps ``FakeRuntime`` and counts how many containers it starts."""

    def __init__(self) -> None:
        self._inner = FakeRuntime()
        self.starts = 0

    async def start(
        self, *, tools_files: Mapping[str, str], workdir_path: str, timeout_seconds: int | None
    ) -> SandboxSession:
        self.starts += 1
        return await self._inner.start(
            tools_files=tools_files, workdir_path=workdir_path, timeout_seconds=timeout_seconds
        )


class _RaisingSession:
    """A session whose first block drops the connection (drives reconnect)."""

    async def begin_block(self, input_paths: Sequence[str]) -> None:
        return None

    async def send(self, frame: Frame) -> None:
        raise SandboxConnectionError("simulated mid-turn connection loss")

    async def frames(self) -> AsyncIterator[Frame]:
        return
        yield  # pragma: no cover  (makes this an async generator)

    async def wait(self) -> SandboxResult:
        raise SandboxConnectionError("simulated mid-turn connection loss")

    async def close(self) -> None:
        return None


class _ConnectionLostOnceBackend:
    """First ``start`` yields a dead session; later starts use a real ``FakeRuntime``."""

    def __init__(self) -> None:
        self._inner = FakeRuntime()
        self.starts = 0

    async def start(
        self, *, tools_files: Mapping[str, str], workdir_path: str, timeout_seconds: int | None
    ) -> SandboxSession:
        self.starts += 1
        if self.starts == 1:
            return _RaisingSession()
        return await self._inner.start(
            tools_files=tools_files, workdir_path=workdir_path, timeout_seconds=timeout_seconds
        )


def _fresh_ctx(session_id: str, invocation_id: str) -> InvocationContext:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id=session_id,
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_invocation_context(artifact_service, session)
    ctx.invocation_id = invocation_id
    return ctx


@pytest.mark.asyncio
async def test_variables_persist_across_blocks_within_a_turn() -> None:
    ctx = _fresh_ctx("s-persist-vars", "inv-persist-vars")
    tool = ExecuteCodeTool(tools=[], backend=_CountingBackend(), max_output_chars=10_000)

    await _run(tool, ctx, "x = 41\n", call_id="b1")
    result = await _run(tool, ctx, "print('x+1', x + 1)\n", call_id="b2")

    assert "x+1 42" in result["stdout"]
    assert result["stderr"].strip() == ""
    # Both blocks reused a single container.
    assert tool._backend.starts == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_workspace_file_persists_across_blocks_within_a_turn() -> None:
    ctx = _fresh_ctx("s-persist-ws", "inv-persist-ws")
    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)

    await _run(tool, ctx, "open('carry.txt', 'w').write('kept')\n", call_id="b1")
    result = await _run(tool, ctx, "print('carry', open('carry.txt').read())\n", call_id="b2")

    assert "carry kept" in result["stdout"]


@pytest.mark.asyncio
async def test_distinct_invocations_get_distinct_sessions_and_no_state_leak() -> None:
    backend = _CountingBackend()
    tool = ExecuteCodeTool(tools=[], backend=backend, max_output_chars=10_000)

    ctx_a = _fresh_ctx("s-distinct-a", "inv-distinct-a")
    await _run(tool, ctx_a, "secret = 7\n", call_id="a1")

    ctx_b = _fresh_ctx("s-distinct-b", "inv-distinct-b")
    result = await _run(tool, ctx_b, "print('has_secret', 'secret' in dir())\n", call_id="b1")

    assert backend.starts == 2
    assert "has_secret False" in result["stdout"]
    assert len(tool._turns) == 2


@pytest.mark.asyncio
async def test_release_invocation_closes_the_turn_session() -> None:
    ctx = _fresh_ctx("s-release", "inv-release")
    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)

    await _run(tool, ctx, "print('hi')\n", call_id="b1")
    turn = tool._turns["inv-release"]

    await tool.release_invocation("inv-release")
    assert "inv-release" not in tool._turns
    assert turn.session._closed is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_idle_reaper_closes_stale_sessions() -> None:
    ctx = _fresh_ctx("s-reap", "inv-reap")
    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)
    tool._session_idle_timeout_seconds = 0.01

    await _run(tool, ctx, "print('hi')\n", call_id="b1")
    turn = tool._turns["inv-reap"]
    turn.last_used = time.monotonic() - 100.0

    await tool._reap_once()

    assert "inv-reap" not in tool._turns
    assert turn.session._closed is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_per_block_output_files_reflect_only_that_blocks_changes() -> None:
    ctx = _fresh_ctx("s-outputs", "inv-outputs")
    tool = ExecuteCodeTool(tools=[], backend=FakeRuntime(), max_output_chars=10_000)

    first = await _run(tool, ctx, "open('a.txt', 'w').write('A')\n", call_id="b1")
    second = await _run(tool, ctx, "open('b.txt', 'w').write('B')\n", call_id="b2")

    assert set(first["output_files"]) == {"a.txt"}
    # a.txt is unchanged in block 2, so only b.txt is reported.
    assert set(second["output_files"]) == {"b.txt"}


@pytest.mark.asyncio
async def test_tool_surface_stays_consistent_across_blocks() -> None:
    ctx = _fresh_ctx("s-surface", "inv-surface")
    tool = ExecuteCodeTool(tools=[_EchoTool()], backend=_CountingBackend(), max_output_chars=10_000)

    call = "from tools import echo\nprint('E', echo(message='hi'))\n"
    await _run(tool, ctx, call, call_id="b1")
    prepared_after_first = tool._turns["inv-surface"].prepared
    result = await _run(tool, ctx, call, call_id="b2")

    assert "E {'echoed': 'hi'}" in result["stdout"]
    # The turn reuses one prepared surface (registry + tools tree) across blocks.
    assert tool._turns["inv-surface"].prepared is prepared_after_first
    assert tool._backend.starts == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_reconnects_once_after_mid_turn_connection_loss() -> None:
    ctx = _fresh_ctx("s-reconnect", "inv-reconnect")
    backend = _ConnectionLostOnceBackend()
    tool = ExecuteCodeTool(tools=[], backend=backend, max_output_chars=10_000)

    result = await _run(tool, ctx, "print('after reconnect')\n", call_id="b1")

    assert backend.starts == 2  # dead session dropped, reconnected once
    assert "after reconnect" in result["stdout"]


class _DropAtWaitSession:
    """RunFrame is accepted, but the connection drops before the OutputFrame.

    ``done_frames`` controls whether a ``DoneFrame`` arrives first: with one the
    tool can tell the code ran (``RAN``); with none it cannot (``UNKNOWN``).
    """

    def __init__(self, *, done: bool) -> None:
        self._done = done

    async def begin_block(self, input_paths: Sequence[str]) -> None:
        return None

    async def send(self, frame: Frame) -> None:
        return None  # RunFrame delivered before the drop

    async def frames(self) -> AsyncIterator[Frame]:
        if self._done:
            yield DoneFrame(exit_code=0)

    async def wait(self) -> SandboxResult:
        raise SandboxConnectionError("connection dropped before OutputFrame")

    async def close(self) -> None:
        return None


class _MismatchedReadySession:
    def __init__(self) -> None:
        self.closed = False

    async def begin_block(self, input_paths: Sequence[str]) -> None:
        return None

    async def send(self, frame: Frame) -> None:
        return None

    async def frames(self) -> AsyncIterator[Frame]:
        yield ReadyFrame(protocol_version=PROTOCOL_VERSION + 1)

    async def wait(self) -> SandboxResult:
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def close(self) -> None:
        self.closed = True


class _SingleSessionBackend:
    """Hands out one prebuilt session and counts starts (to prove no reconnect)."""

    def __init__(self, session: SandboxSession) -> None:
        self._session = session
        self.starts = 0

    async def start(
        self, *, tools_files: Mapping[str, str], workdir_path: str, timeout_seconds: int | None
    ) -> SandboxSession:
        self.starts += 1
        return self._session


@pytest.mark.asyncio
async def test_unknown_execution_is_reported_and_not_retried() -> None:
    ctx = _fresh_ctx("s-unknown", "inv-unknown")
    backend = _SingleSessionBackend(_DropAtWaitSession(done=False))
    tool = ExecuteCodeTool(tools=[], backend=backend, max_output_chars=10_000)

    result = await _run(tool, ctx, "side_effect()\n", call_id="b1")

    # Code may have run, so we must not re-run it: one start, a message, no output.
    assert backend.starts == 1
    assert "unknown whether it executed" in result["stderr"]
    assert result["stdout"] == ""


@pytest.mark.asyncio
async def test_completed_execution_with_lost_output_is_reported_and_not_retried() -> None:
    ctx = _fresh_ctx("s-ran", "inv-ran")
    backend = _SingleSessionBackend(_DropAtWaitSession(done=True))
    tool = ExecuteCodeTool(tools=[], backend=backend, max_output_chars=10_000)

    result = await _run(tool, ctx, "side_effect()\n", call_id="b1")

    # DoneFrame arrived, so we know it ran: one start, a message, no re-run.
    assert backend.starts == 1
    assert "already executed" in result["stderr"]
    assert result["stdout"] == ""


@pytest.mark.asyncio
async def test_protocol_mismatch_discards_and_closes_cached_turn() -> None:
    ctx = _fresh_ctx("s-protocol", "inv-protocol")
    session = _MismatchedReadySession()
    backend = _SingleSessionBackend(session)
    tool = ExecuteCodeTool(tools=[], backend=backend, max_output_chars=10_000)

    with pytest.raises(ProtocolVersionMismatchError):
        await _run(tool, ctx, "print('hi')\n", call_id="b1")

    assert "inv-protocol" not in tool._turns
    assert session.closed is True
