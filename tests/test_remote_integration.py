# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for ``RemoteBackend`` ↔ sandbox HTTP server.

Starts the sandbox HTTP server as a subprocess (no Docker required), then
exercises the full execution path through ``RemoteBackend``.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import websockets.exceptions
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from adk_code_mode import ExecuteCodeTool, RemoteBackend

_SANDBOX_SRC = Path(__file__).resolve().parent.parent / "sandbox-wheel" / "src"


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


def _make_ctx(artifact_service: InMemoryArtifactService, session: Session) -> InvocationContext:
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
    ctx.invocation_id = "remote-inv-1"
    return ctx


def _tool_context(ctx: InvocationContext, *, call_id: str) -> ToolContext:
    return ToolContext(invocation_context=ctx, function_call_id=call_id)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def http_server(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Start a single-use sandbox HTTP server subprocess.

    Each test gets its own process because the server exits after handling one
    WebSocket connection.
    """
    port = _free_port()
    env = os.environ.copy()
    env["ADK_CODE_MODE_CONTROL_HTTP"] = "1"
    env["PORT"] = str(port)
    # The server extracts tools into TOOLS_DIR (default /tools, created in the
    # image). Run outside a container as a non-root user, it can't mkdir /tools
    # at the fs root, so point it at a writable temp dir — same as _fake_runtime.
    env["ADK_CODE_MODE_TOOLS_DIR"] = str(tmp_path / "tools")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(_SANDBOX_SRC), env.get("PYTHONPATH", "")])
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "adk_code_mode_sandbox"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for the server to be ready by polling the port
    deadline = time.monotonic() + 10.0
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            if time.monotonic() > deadline or proc.poll() is not None:
                proc.kill()
                out, err = proc.communicate(timeout=5)
                raise RuntimeError(
                    f"HTTP server failed to start:\nstdout={out.decode()}\nstderr={err.decode()}"
                )
            time.sleep(0.1)

    proc._port = port  # type: ignore[attr-defined]
    yield proc
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


def _server_url(server: Any) -> str:
    return f"ws://127.0.0.1:{server._port}"


@pytest.mark.asyncio
async def test_remote_tool_call(http_server: subprocess.Popen[bytes]) -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="remote-s1",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_ctx(artifact_service, session)

    tool = ExecuteCodeTool(
        tools=[_EchoTool()],
        backend=RemoteBackend(url=_server_url(http_server)),
        max_output_chars=10_000,
    )
    code = "from tools import echo\nr = echo(message='hello from remote')\nprint('ECHO:', r)\n"
    result = await tool.run_async(
        args={"code": code}, tool_context=_tool_context(ctx, call_id="remote-run-1")
    )
    assert "ECHO: {'echoed': 'hello from remote'}" in result["stdout"]


@pytest.mark.asyncio
async def test_remote_plain_code(http_server: subprocess.Popen[bytes]) -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="remote-s2",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_ctx(artifact_service, session)

    tool = ExecuteCodeTool(
        tools=[],
        backend=RemoteBackend(url=_server_url(http_server)),
        max_output_chars=10_000,
    )
    result = await tool.run_async(
        args={"code": "print('hello world')"},
        tool_context=_tool_context(ctx, call_id="remote-run-2"),
    )
    assert "hello world" in result["stdout"]


@pytest.mark.asyncio
async def test_remote_workspace_files(http_server: subprocess.Popen[bytes]) -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="remote-s3",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_ctx(artifact_service, session)

    tool = ExecuteCodeTool(
        tools=[],
        backend=RemoteBackend(url=_server_url(http_server)),
        max_output_chars=10_000,
    )
    code = (
        "with open('output.txt', 'w') as f:\n"
        "    f.write('created in sandbox')\n"
        "print('wrote file')\n"
    )
    result = await tool.run_async(
        args={"code": code}, tool_context=_tool_context(ctx, call_id="remote-run-3")
    )
    assert "wrote file" in result["stdout"]
    assert "output.txt" in result["output_files"]


@pytest.mark.asyncio
async def test_remote_variables_and_workspace_persist_across_blocks(
    http_server: subprocess.Popen[bytes],
) -> None:
    """Two blocks of one turn share the same container (one connection)."""
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="remote-multi",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_ctx(artifact_service, session)  # shared invocation_id => one turn

    tool = ExecuteCodeTool(
        tools=[],
        backend=RemoteBackend(url=_server_url(http_server)),
        max_output_chars=10_000,
    )

    await tool.run_async(
        args={"code": "x = 10\nopen('kept.txt', 'w').write('carried')\n"},
        tool_context=_tool_context(ctx, call_id="remote-multi-1"),
    )
    result = await tool.run_async(
        args={"code": "print('x*4', x * 4)\nprint('file', open('kept.txt').read())\n"},
        tool_context=_tool_context(ctx, call_id="remote-multi-2"),
    )

    assert "x*4 40" in result["stdout"]
    assert "file carried" in result["stdout"]


@pytest.mark.asyncio
async def test_remote_workspace_deletions_update_host_snapshot(
    http_server: subprocess.Popen[bytes],
) -> None:
    """The downloaded workspace tar is an authoritative snapshot, including deletes."""
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="remote-delete",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_ctx(artifact_service, session)

    tool = ExecuteCodeTool(
        tools=[],
        backend=RemoteBackend(url=_server_url(http_server)),
        max_output_chars=10_000,
    )

    await tool.run_async(
        args={"code": "open('roundtrip.txt', 'w').write('same')\n"},
        tool_context=_tool_context(ctx, call_id="remote-delete-1"),
    )
    deleted = await tool.run_async(
        args={
            "code": (
                "import os\nos.remove('roundtrip.txt')\nprint(os.path.exists('roundtrip.txt'))\n"
            )
        },
        tool_context=_tool_context(ctx, call_id="remote-delete-2"),
    )
    recreated = await tool.run_async(
        args={"code": "open('roundtrip.txt', 'w').write('same')\n"},
        tool_context=_tool_context(ctx, call_id="remote-delete-3"),
    )

    assert "False" in deleted["stdout"]
    assert recreated["output_files"] == ["roundtrip.txt"]


@pytest.mark.asyncio
async def test_remote_second_connection_is_rejected(
    http_server: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    """While one turn holds the connection, a second connect gets rejected."""
    backend = RemoteBackend(url=_server_url(http_server), start_attempts=1)
    w1 = tmp_path / "w1"
    w1.mkdir()
    session1 = await backend.start(tools_files={}, workdir_path=str(w1), timeout_seconds=None)
    try:
        w2 = tmp_path / "w2"
        w2.mkdir()
        with pytest.raises(websockets.exceptions.WebSocketException):
            await backend.start(tools_files={}, workdir_path=str(w2), timeout_seconds=None)
    finally:
        await session1.close()


@pytest.mark.asyncio
async def test_remote_close_shuts_the_container_down(
    http_server: subprocess.Popen[bytes],
) -> None:
    """Releasing the turn (ShutdownFrame) makes the single-use server exit."""
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="remote-shutdown",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _make_ctx(artifact_service, session)

    tool = ExecuteCodeTool(
        tools=[],
        backend=RemoteBackend(url=_server_url(http_server)),
        max_output_chars=10_000,
    )
    await tool.run_async(
        args={"code": "print('ready')\n"},
        tool_context=_tool_context(ctx, call_id="remote-shutdown-1"),
    )

    assert http_server.poll() is None  # still serving the turn
    await tool.release_invocation(ctx.invocation_id)

    for _ in range(200):
        if http_server.poll() is not None:
            break
        await asyncio.sleep(0.02)
    assert http_server.poll() is not None  # container exited after ShutdownFrame
