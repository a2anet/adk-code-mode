# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""In-process end-to-end tests using ``FakeRuntime`` (no Docker)."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.code_executors.code_execution_utils import CodeExecutionInput, File
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.genai import types as genai_types

from adk_code_mode.executor import CodeModeExecutor

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
    # Tests run ``_aexecute`` directly without firing the model-callback, so
    # there's no cached resolution to consume; the executor falls back to a
    # fresh resolve each call (which lets dynamic toolsets see updated state).
    ctx.invocation_id = "test-inv-1"
    return ctx


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

    executor = CodeModeExecutor(
        tools=[_EchoTool()],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )
    code = "from tools import echo\nr = echo(message='hello from sandbox')\nprint('ECHO:', r)\n"

    result = await executor._aexecute(ctx, CodeExecutionInput(code=code, execution_id="run-1"))
    assert "ECHO: {'echoed': 'hello from sandbox'}" in result.stdout
    assert result.stderr.strip() == ""


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

    executor = CodeModeExecutor(
        tools=[],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )
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

    result = await executor._aexecute(ctx, CodeExecutionInput(code=code, execution_id="run-ws"))
    assert "report.json" in result.stdout
    assert "loaded {'ok': True}" in result.stdout
    keys = await artifact_service.list_artifact_keys(
        app_name="test-app", user_id="u1", session_id="s2"
    )
    assert "report.json" in keys


@pytest.mark.asyncio
async def test_workspace_inputs_are_staged_and_outputs_are_collected() -> None:
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

    executor = CodeModeExecutor(
        tools=[],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )
    code = (
        "import os\n"
        "print('cwd', os.getcwd())\n"
        "print('input', open('input.txt').read())\n"
        "with open('input.txt', 'w') as fh:\n"
        "    fh.write('changed')\n"
        "with open('result.txt', 'w') as fh:\n"
        "    fh.write('created')\n"
    )

    result = await executor._aexecute(
        ctx,
        CodeExecutionInput(
            code=code,
            execution_id="run-workspace",
            input_files=[File(name="input.txt", content=b"hello", mime_type="text/plain")],
        ),
    )
    assert "input hello" in result.stdout
    assert {file.name for file in result.output_files} == {"input.txt", "result.txt"}


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

    executor = CodeModeExecutor(
        tools=[],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )

    save = (
        "import base64\n"
        "from tools import save_artifact\n"
        "save_artifact(\n"
        "    filename='blob.bin',\n"
        "    content=base64.b64encode(b'\\x00\\x01\\x02').decode('ascii'),\n"
        "    mime_type='application/octet-stream',\n"
        ")\n"
    )
    await executor._aexecute(ctx, CodeExecutionInput(code=save, execution_id="run-bin-1"))

    load = (
        "import base64\n"
        "from tools import load_artifact\n"
        "blob = load_artifact(filename='blob.bin')\n"
        "print(blob['kind'], base64.b64decode(blob['data']), blob['mime_type'])\n"
    )
    result = await executor._aexecute(ctx, CodeExecutionInput(code=load, execution_id="run-bin-2"))

    assert "bytes b'\\x00\\x01\\x02' application/octet-stream" in result.stdout


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

    executor = CodeModeExecutor(
        tools=[],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )

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
    await executor._aexecute(ctx, CodeExecutionInput(code=first, execution_id="run-fresh-1"))

    second = (
        "import os\n"
        "from tools import load_artifact\n"
        "print('workspace_has_temp', os.path.exists('temp.txt'))\n"
        "print('artifact_value', load_artifact(filename='persist.txt')['data'])\n"
    )
    result = await executor._aexecute(
        ctx, CodeExecutionInput(code=second, execution_id="run-fresh-2")
    )

    assert "workspace_has_temp False" in result.stdout
    assert "artifact_value artifact persisted" in result.stdout


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

    executor = CodeModeExecutor(
        tools=[],
        runtime=FakeRuntime(),
        max_output_chars=500,
    )
    code = "print('x' * 5000)\n"

    result = await executor._aexecute(ctx, CodeExecutionInput(code=code, execution_id="run-big"))
    assert "Output exceeded 500 characters" in result.stdout
    assert "Full stdout saved as artifact" in result.stdout
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

    executor = CodeModeExecutor(
        tools=[],
        runtime=FakeRuntime(),
        max_output_chars=500,
        timeout_seconds=1,
    )

    result = await executor._aexecute(
        ctx, CodeExecutionInput(code="while True:\n    pass\n", execution_id="run-timeout")
    )
    assert result.stdout == ""
    assert "Execution exceeded timeout of 1s" in result.stderr


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

    executor = CodeModeExecutor(
        tools=[_SlowTool()],
        runtime=FakeRuntime(),
        max_output_chars=500,
        timeout_seconds=1,
    )

    start = time.monotonic()
    result = await executor._aexecute(
        ctx,
        CodeExecutionInput(
            code="from tools import slow\nprint(slow())\n",
            execution_id="run-tool-timeout",
        ),
    )
    elapsed = time.monotonic() - start

    assert elapsed < 1.8
    assert result.stdout == ""
    assert "Execution exceeded timeout of 1s" in result.stderr
    assert "Process exited with code" not in result.stderr


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

    executor = CodeModeExecutor(
        tools=[],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )

    result = await executor._aexecute(
        ctx,
        CodeExecutionInput(code="import sys\nsys.exit(2)\n", execution_id="run-exit"),
    )

    assert result.stdout == ""
    assert result.stderr == "Process exited with code 2."


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

    executor = CodeModeExecutor(
        tools=[],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )

    result = await executor._aexecute(
        ctx,
        CodeExecutionInput(code="raise ValueError('bad')\n", execution_id="run-traceback"),
    )

    assert "ValueError: bad" in result.stderr
    assert result.stderr.rstrip().endswith("Process exited with code 1.")


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

    executor = CodeModeExecutor(
        tools=[],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )

    result = await executor._aexecute(
        ctx,
        CodeExecutionInput(code="if True print('bad')\n", execution_id="run-syntax"),
    )

    assert "SyntaxError" in result.stderr
    assert result.stderr.rstrip().endswith("Process exited with code 1.")


@pytest.mark.asyncio
async def test_prepare_tool_surface_re_resolves_dynamic_toolsets_without_invocation_cache() -> None:
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

    executor = CodeModeExecutor(
        tools=[_DynamicToolset()],
        runtime=FakeRuntime(),
    )

    ctx.invocation_id = "test-inv-alpha"
    first = await executor._prepare_tool_surface(ctx)
    session.state["which"] = "beta"
    ctx.invocation_id = "test-inv-beta"
    second = await executor._prepare_tool_surface(ctx)

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

    executor = CodeModeExecutor(
        tools=[_OptionalArgTool()],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )
    code = "from tools import optional_lookup\nprint(optional_lookup())\n"

    result = await executor._aexecute(ctx, CodeExecutionInput(code=code, execution_id="run-opt"))
    assert "'has_record_id': False" in result.stdout
    assert result.stderr.strip() == ""
