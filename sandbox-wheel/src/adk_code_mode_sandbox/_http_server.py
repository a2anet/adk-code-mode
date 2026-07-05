# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""HTTP/WebSocket server mode for the sandbox (one connection per turn).

Activated by ``ADK_CODE_MODE_CONTROL_HTTP=1``. The container accepts exactly
one WebSocket connection and holds it open for the whole turn, executing one or
more code blocks in-process with a persistent globals dict and ``/workspace``,
then exits when the turn ends (``ShutdownFrame`` or disconnect). The hosting
platform (Cloud Run, Fargate, K8s, etc.) creates a new container per turn.

Files (tools and workspace) are transferred as tar archives over binary
WebSocket frames; control frames use the standard JSON-lines protocol
over text WebSocket frames.

Connection protocol::

    1. Client sends binary: tools tar.gz   (once)
    2. Server sends text:   ReadyFrame      (once)
    3. Per code block, in fixed frame order:
         a. Client sends binary: workspace tar.gz (this block's inputs; may be empty)
         b. Client sends text:   RunFrame
         c. Bidirectional text frames: JSON-lines tool-call protocol
         d. Server sends text:   DoneFrame
         e. Server sends text:   OutputFrame
         f. Server sends binary: workspace tar.gz (updated files)
    4. Client sends text: ShutdownFrame (or disconnects) to end the turn.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import queue as queue_mod
import shutil
import tarfile
import tempfile
from http import HTTPStatus
from typing import Any

import websockets
import websockets.asyncio.server

from adk_code_mode_sandbox._entry import (
    _make_globals,
    _prepare_sys_path,
    _prepare_workdir,
    _sanitize_environ,
    run_block,
)
from adk_code_mode_sandbox._rpc_client import RpcClient
from adk_code_mode_sandbox import _rpc_client
from adk_code_mode_sandbox.protocol import (
    DoneFrame,
    OutputFrame,
    ProtocolError,
    ReadyFrame,
    RunFrame,
    ShutdownFrame,
    decode,
    encode,
)

logger = logging.getLogger("adk_code_mode_sandbox.http_server")

CONTROL_HTTP_ENV = "ADK_CODE_MODE_CONTROL_HTTP"
AUTH_TOKEN_ENV = "ADK_CODE_MODE_AUTH_TOKEN"
MAX_UPLOAD_TOOLS_ENV = "ADK_CODE_MODE_MAX_UPLOAD_TOOLS"
MAX_UPLOAD_WORKSPACE_ENV = "ADK_CODE_MODE_MAX_UPLOAD_WORKSPACE"

_DEFAULT_MAX_TOOLS_BYTES = 100 * 1024 * 1024  # 100 MiB
_DEFAULT_MAX_WORKSPACE_BYTES = 100 * 1024 * 1024  # 100 MiB
_WS_MAX_SIZE = 256 * 1024 * 1024  # 256 MiB


class _WsBridgeReader:
    """Sync reader backed by a thread-safe queue.

    The async WS pump pushes incoming text frames; the ``RpcClient`` reader
    thread consumes them via ``readline()``.
    """

    def __init__(self) -> None:
        self._q: queue_mod.Queue[bytes] = queue_mod.Queue()

    def feed(self, data: bytes) -> None:
        self._q.put_nowait(data)

    def close(self) -> None:
        self._q.put_nowait(b"")

    def readline(self) -> bytes:
        return self._q.get()


class _WsBridgeWriter:
    """Sync writer that sends to a WebSocket via the event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, ws: Any) -> None:
        self._loop = loop
        self._ws = ws

    def write(self, data: bytes) -> None:
        text = data.decode("utf-8")
        future = asyncio.run_coroutine_threadsafe(self._ws.send(text), self._loop)
        future.result()

    def flush(self) -> None:
        pass


async def serve(port: int, token: str | None) -> None:
    """Start the WebSocket server on *port* with optional bearer *token* auth.

    Accepts exactly one WebSocket connection, runs user code, then returns.
    """
    stop = asyncio.Event()
    accepted = False

    def process_request(connection: Any, request: Any) -> websockets.http11.Response | None:
        nonlocal accepted
        if request.path == "/health":
            return connection.respond(HTTPStatus.OK, "OK\n")
        if accepted:
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "Busy\n")
        if token is not None:
            import hmac

            auth = (request.headers.get("Authorization") or "").strip()
            if not hmac.compare_digest(auth, f"Bearer {token}"):
                return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        accepted = True
        return None

    async def handler(ws: Any) -> None:
        try:
            await _handle_connection(ws)
        except websockets.ConnectionClosed:
            logger.debug("client disconnected")
        except Exception:
            logger.exception("error handling WebSocket connection")
        finally:
            stop.set()

    async with websockets.asyncio.server.serve(
        handler,
        "0.0.0.0",
        port,
        process_request=process_request,
        max_size=_WS_MAX_SIZE,
    ):
        logger.info("sandbox HTTP server listening on port %d", port)
        await stop.wait()


async def _handle_connection(ws: Any) -> None:
    max_tools = int(os.environ.get(MAX_UPLOAD_TOOLS_ENV, _DEFAULT_MAX_TOOLS_BYTES))
    max_workspace = int(os.environ.get(MAX_UPLOAD_WORKSPACE_ENV, _DEFAULT_MAX_WORKSPACE_BYTES))

    tools_base = tempfile.mkdtemp(prefix="adk-cm-tools-")
    tools_dir = os.path.join(tools_base, "tools")
    os.makedirs(tools_dir)
    workspace_dir = tempfile.mkdtemp(prefix="adk-cm-ws-")

    try:
        # 1. Receive and extract the tools tar (once for the whole connection).
        tools_data = await ws.recv()
        if not isinstance(tools_data, (bytes, bytearray)):
            raise ValueError("expected binary frame for tools tar")
        if len(tools_data) > max_tools:
            raise ValueError(
                f"tools archive ({len(tools_data):,} bytes) exceeds limit ({max_tools:,} bytes)"
            )
        _extract_tar(tools_data, tools_dir)

        # 2. Sanitize environment, then prepare sys.path + a persistent /workspace.
        _sanitize_environ()
        _prepare_sys_path(tools_dir)
        _prepare_workdir(workspace_dir)

        # 3. One persistent globals dict for the connection, so variables set in
        #    one block are visible to later blocks in the same turn.
        globs = _make_globals()
        loop = asyncio.get_running_loop()

        # 4. Announce readiness once (carries the protocol version).
        await ws.send(encode(ReadyFrame()).decode("utf-8"))

        # 5. Run code blocks until the client ends the turn or disconnects.
        while await _run_one_block(ws, loop, globs, workspace_dir, max_workspace):
            pass
    finally:
        shutil.rmtree(tools_base, ignore_errors=True)
        shutil.rmtree(workspace_dir, ignore_errors=True)


async def _run_one_block(
    ws: Any,
    loop: asyncio.AbstractEventLoop,
    globs: dict[str, Any],
    workspace_dir: str,
    max_workspace: int,
) -> bool:
    """Handle one code block. Returns ``False`` when the turn should end.

    The per-block frame order is fixed — the binary workspace tar first, then
    the text ``RunFrame`` — which is what keeps the WebSocket from desyncing
    across blocks.
    """
    # a. This block's workspace tar (binary), or a control frame ending the turn.
    try:
        first = await ws.recv()
    except websockets.ConnectionClosed:
        return False
    if isinstance(first, str):
        try:
            frame = decode(first)
        except ProtocolError:
            return True
        if isinstance(frame, ShutdownFrame):
            return False
        return True  # unexpected control frame between blocks; ignore
    if len(first) > max_workspace:
        raise ValueError(
            f"workspace archive ({len(first):,} bytes) exceeds limit ({max_workspace:,} bytes)"
        )
    _extract_tar(first, workspace_dir)

    # b. The RunFrame (text).
    run_msg = await ws.recv()
    if not isinstance(run_msg, str):
        raise ValueError("expected text frame for RunFrame")
    run_frame = decode(run_msg)
    if not isinstance(run_frame, RunFrame):
        raise ValueError(f"expected RunFrame, got {type(run_frame).__name__}")

    # c. Run user code on a worker thread while pumping incoming text frames
    #    (tool results) into the RPC bridge. A fresh bridge/client per block
    #    keeps the reader-thread lifecycle simple; the globals dict persists.
    bridge_reader = _WsBridgeReader()
    bridge_writer = _WsBridgeWriter(loop, ws)
    client = RpcClient(reader=bridge_reader, writer=bridge_writer)
    _rpc_client.install(client)
    pump_task = asyncio.create_task(_pump_ws_to_reader(ws, bridge_reader))
    try:
        stdout_text, stderr_text, exit_code = await loop.run_in_executor(
            None, run_block, run_frame.code, globs
        )
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass

    # d-f. DoneFrame, OutputFrame, then the updated workspace tar (binary).
    await ws.send(encode(DoneFrame(exit_code=exit_code)).decode("utf-8"))
    await ws.send(
        encode(OutputFrame(stdout=stdout_text, stderr=stderr_text, exit_code=exit_code)).decode(
            "utf-8"
        )
    )
    await ws.send(_create_tar(workspace_dir))
    return True


async def _pump_ws_to_reader(ws: Any, reader: _WsBridgeReader) -> None:
    """Forward incoming WS text frames (tool results) to the bridge reader queue.

    Runs only while a block executes. Uses an explicit ``recv()`` loop (rather
    than ``async for``) so it can be cancelled and a fresh pump started for the
    next block. Binary frames are not expected mid-block and are ignored.
    """
    try:
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = msg.encode("utf-8")
                if not data.endswith(b"\n"):
                    data += b"\n"
                reader.feed(data)
    except websockets.ConnectionClosed:
        pass
    finally:
        reader.close()


def _extract_tar(data: bytes | bytearray, dest: str) -> None:
    if not data:
        return
    dest_real = os.path.realpath(dest)
    buf = io.BytesIO(data)
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        members = []
        for member in tf.getmembers():
            if not (member.isfile() or member.isdir()):
                continue
            if member.linkname:
                continue
            normalized = os.path.normpath(member.name)
            if normalized.startswith("/") or normalized.startswith(".."):
                continue
            if any(part == ".." for part in normalized.split(os.sep)):
                continue
            resolved = os.path.realpath(os.path.join(dest, normalized))
            if not resolved.startswith(dest_real + os.sep) and resolved != dest_real:
                continue
            member.name = normalized
            members.append(member)
        tf.extractall(dest, members=members)


def _create_tar(source_dir: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for root, _dirs, files in os.walk(source_dir, followlinks=False):
            for fname in files:
                abs_path = os.path.join(root, fname)
                if os.path.islink(abs_path) or not os.path.isfile(abs_path):
                    continue
                arcname = os.path.relpath(abs_path, source_dir)
                tf.add(abs_path, arcname=arcname)
    return buf.getvalue()
