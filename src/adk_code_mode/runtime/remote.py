# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Remote sandbox backend.

Connects to a sandbox HTTP/WebSocket server running the same Docker image
as :class:`UnsafeLocalDockerBackend` but in HTTP mode (activated by
``ADK_CODE_MODE_CONTROL_HTTP=1``).

Works with any platform that hosts Docker containers as HTTP services:
Cloud Run, Fargate, ACI, Kubernetes, Fly.io, etc.
"""

from __future__ import annotations

import io
import os
import tarfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Mapping

import websockets
import websockets.asyncio.client

from adk_code_mode.runtime.base import SandboxResult, SandboxSession
from adk_code_mode.runtime.protocol import Frame, OutputFrame, ProtocolError, decode, encode


@dataclass
class RemoteBackend:
    """Connect to a remote sandbox over WebSocket.

    Args:
        url: Base URL of the sandbox HTTP server, e.g.
            ``"https://sandbox-xyz.run.app"`` or ``"ws://localhost:8080"``.
        token: Optional bearer token for authentication.
        connect_timeout: Seconds to wait for the WebSocket handshake.
    """

    url: str
    token: str | None = None
    connect_timeout: float = 30.0
    max_message_size: int = 256 * 1024 * 1024
    max_upload_tools_bytes: int = 100 * 1024 * 1024
    max_upload_workspace_bytes: int = 100 * 1024 * 1024
    max_download_workspace_bytes: int = 100 * 1024 * 1024

    async def start(
        self,
        *,
        tools_files: Mapping[str, str],
        workdir_path: str,
        timeout_seconds: int | None,
    ) -> SandboxSession:
        ws_url = self.url
        if ws_url.startswith("https://"):
            ws_url = "wss://" + ws_url[len("https://") :]
        elif ws_url.startswith("http://"):
            ws_url = "ws://" + ws_url[len("http://") :]
        elif not ws_url.startswith(("ws://", "wss://")):
            ws_url = "wss://" + ws_url

        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        ws = await websockets.asyncio.client.connect(
            ws_url,
            additional_headers=headers,
            open_timeout=self.connect_timeout,
            max_size=self.max_message_size,
        )

        try:
            tools_tar = _create_tar_from_mapping(tools_files)
            if len(tools_tar) > self.max_upload_tools_bytes:
                raise ValueError(
                    f"tools archive ({len(tools_tar):,} bytes) exceeds "
                    f"max_upload_tools_bytes ({self.max_upload_tools_bytes:,})"
                )
            workspace_tar = _create_tar_from_dir(workdir_path)
            if len(workspace_tar) > self.max_upload_workspace_bytes:
                raise ValueError(
                    f"workspace archive ({len(workspace_tar):,} bytes) exceeds "
                    f"max_upload_workspace_bytes ({self.max_upload_workspace_bytes:,})"
                )
            await ws.send(tools_tar)
            await ws.send(workspace_tar)
        except BaseException:
            await ws.close()
            raise

        return _RemoteSandboxSession(
            ws=ws,
            workdir_path=workdir_path,
            max_download_workspace_bytes=self.max_download_workspace_bytes,
        )


class _RemoteSandboxSession:
    def __init__(self, *, ws: Any, workdir_path: str, max_download_workspace_bytes: int) -> None:
        self._ws = ws
        self._workdir_path = workdir_path
        self._max_download_workspace_bytes = max_download_workspace_bytes

    async def send(self, frame: Frame) -> None:
        payload = encode(frame)
        await self._ws.send(payload.decode("utf-8"))

    async def frames(self) -> AsyncIterator[Frame]:
        try:
            while True:
                msg = await self._ws.recv()
                if isinstance(msg, str):
                    try:
                        yield decode(msg)
                    except ProtocolError:
                        continue
        except websockets.ConnectionClosed:
            return

    async def wait(self) -> SandboxResult:
        msg = await self._ws.recv()
        if not isinstance(msg, str):
            raise RuntimeError("expected text frame for OutputFrame")
        frame = decode(msg)
        if not isinstance(frame, OutputFrame):
            raise RuntimeError(f"expected OutputFrame, got {type(frame).__name__}")

        tar_data = await self._ws.recv()
        if isinstance(tar_data, (bytes, bytearray)) and tar_data:
            if len(tar_data) > self._max_download_workspace_bytes:
                raise ValueError(
                    f"workspace response archive ({len(tar_data):,} bytes) exceeds "
                    f"max_download_workspace_bytes ({self._max_download_workspace_bytes:,})"
                )
            _extract_tar(tar_data, self._workdir_path)

        return SandboxResult(
            stdout=frame.stdout,
            stderr=frame.stderr,
            exit_code=frame.exit_code,
        )

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass


def _create_tar_from_mapping(files: Mapping[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel_path, source in files.items():
            data = source.encode("utf-8")
            info = tarfile.TarInfo(name=rel_path)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _create_tar_from_dir(path: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for fname in files:
                abs_path = os.path.join(root, fname)
                if os.path.islink(abs_path) or not os.path.isfile(abs_path):
                    continue
                arcname = os.path.relpath(abs_path, path)
                tf.add(abs_path, arcname=arcname)
    return buf.getvalue()


def _extract_tar(data: bytes | bytearray, dest: str) -> None:
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


__all__ = ["RemoteBackend"]
