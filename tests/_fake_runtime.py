# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Test-only in-process runtime that runs the sandbox entry in a subprocess.

Does **not** ship as part of the package. Uses a Unix-domain socket pair so
we get identical wire semantics to the Docker runtime without needing Docker.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import tempfile
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from adk_code_mode.runtime.base import SandboxSession, SandboxResult, SandboxBackend
from adk_code_mode.runtime.protocol import Frame, ProtocolError, decode, encode

_SANDBOX_SRC = Path(__file__).resolve().parent.parent / "sandbox-wheel" / "src"


class FakeRuntime(SandboxBackend):
    """Runs the sandbox as a local subprocess for in-process integration tests."""

    async def start(
        self,
        *,
        tools_files: Mapping[str, str],
        workdir_path: str,
        timeout_seconds: int | None,
    ) -> SandboxSession:
        tools_root = tempfile.mkdtemp(prefix="fake-runtime-tools-")
        tools_dir = os.path.join(tools_root, "tools")
        os.makedirs(tools_dir, exist_ok=True)
        for rel, source in tools_files.items():
            dest = os.path.join(tools_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            Path(dest).write_text(source, encoding="utf-8")

        os.makedirs(workdir_path, exist_ok=True)

        control_dir = tempfile.mkdtemp(prefix="fake-runtime-ctrl-")
        sock_path = os.path.join(control_dir, "control.sock")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(sock_path)
        listener.listen(1)

        env = os.environ.copy()
        env["ADK_CODE_MODE_CONTROL_SOCKET"] = sock_path
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(_SANDBOX_SRC), env.get("PYTHONPATH", "")])
        )
        # Patch the sandbox /tools and /workspace directory expectations.
        env["ADK_CODE_MODE_TOOLS_DIR"] = tools_dir
        env["ADK_CODE_MODE_WORKDIR"] = workdir_path

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "adk_code_mode_sandbox",
            cwd=workdir_path,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )

        loop = asyncio.get_running_loop()
        listener.settimeout(10.0)
        try:
            conn, _ = await loop.run_in_executor(None, listener.accept)
        except Exception:
            proc.kill()
            raise
        conn.setblocking(False)

        return _FakeSession(
            conn=conn,
            listener=listener,
            proc=proc,
            tools_root=tools_root,
            control_dir=control_dir,
        )


class _FakeSession(SandboxSession):
    def __init__(
        self,
        *,
        conn: socket.socket,
        listener: socket.socket,
        proc: asyncio.subprocess.Process,
        tools_root: str,
        control_dir: str,
    ) -> None:
        self._sock = conn
        self._listener = listener
        self._proc = proc
        self._tools_root = tools_root
        self._control_dir = control_dir
        self._buf = b""
        self._closed = False
        self._stdout_bytes = bytearray()
        self._stderr_bytes = bytearray()
        self._stdout_task = asyncio.create_task(_drain(proc.stdout, self._stdout_bytes))
        self._stderr_task = asyncio.create_task(_drain(proc.stderr, self._stderr_bytes))

    async def send(self, frame: Frame) -> None:
        await asyncio.get_running_loop().sock_sendall(self._sock, encode(frame))

    async def frames(self) -> AsyncIterator[Frame]:
        loop = asyncio.get_running_loop()
        while True:
            while b"\n" not in self._buf:
                try:
                    chunk = await loop.sock_recv(self._sock, 65536)
                except (OSError, ConnectionError):
                    return
                if not chunk:
                    return
                self._buf += chunk
            line, self._buf = self._buf.split(b"\n", 1)
            if not line:
                continue
            try:
                yield decode(line)
            except ProtocolError:
                continue

    async def wait(self) -> SandboxResult:
        await self._proc.wait()
        await asyncio.gather(self._stdout_task, self._stderr_task, return_exceptions=True)
        return SandboxResult(
            stdout=bytes(self._stdout_bytes).decode("utf-8", errors="replace"),
            stderr=bytes(self._stderr_bytes).decode("utf-8", errors="replace"),
            exit_code=self._proc.returncode or 0,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            self._listener.close()
        except OSError:
            pass
        if self._proc.returncode is None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            await self._proc.wait()
        await asyncio.gather(self._stdout_task, self._stderr_task, return_exceptions=True)
        import shutil

        shutil.rmtree(self._tools_root, ignore_errors=True)
        shutil.rmtree(self._control_dir, ignore_errors=True)


async def _drain(stream: Any, buf: bytearray) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        buf.extend(chunk)
