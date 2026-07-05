# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Remote sandbox backend.

Connects to a sandbox HTTP/WebSocket server running the same Docker image
as :class:`UnsafeLocalDockerBackend` but in HTTP mode (activated by
``ADK_CODE_MODE_CONTROL_HTTP=1``).

Works with any platform that hosts Docker containers as HTTP services:
Cloud Run, Fargate, ACI, Kubernetes, Fly.io, etc.

One connection is held open for the whole turn: ``start`` connects and uploads
the tools once, each block uploads its own inputs and downloads the updated
``/workspace``, and ``close`` sends a ``ShutdownFrame`` to release the container.
"""

from __future__ import annotations

import asyncio
import io
import os
import random
import tarfile
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import websockets
import websockets.asyncio.client

from adk_code_mode.runtime.base import SandboxConnectionError, SandboxResult, SandboxSession
from adk_code_mode.runtime.protocol import (
    Frame,
    OutputFrame,
    ProtocolError,
    ShutdownFrame,
    decode,
    encode,
)


def _is_retryable_connect_error(exc: BaseException) -> bool:
    """Transient connect failures worth retrying (slow cold start, reset, etc.)."""
    if isinstance(exc, (ConnectionError, EOFError, OSError, TimeoutError)):
        return True
    return type(exc).__module__.startswith("websockets")


@dataclass
class RemoteBackend:
    """Connect to a remote sandbox over WebSocket.

    Args:
        url: Base URL of the sandbox HTTP server, e.g.
            ``"https://sandbox-xyz.run.app"`` or ``"ws://localhost:8080"``.
        token: Optional bearer token for authentication.
        connect_timeout: Seconds to wait for the WebSocket handshake. Defaults to
            ``10`` — fail a stalled connect fast and retry rather than blocking.
            Raise it for platforms with slow cold starts.
        start_attempts: How many times to attempt the initial connect (+ tools
            upload) before giving up. ``1`` disables retry.
        start_retry_delay_seconds / start_retry_jitter_seconds: Linear backoff
            between connect attempts — ``delay * attempt + uniform(0, jitter)``.
    """

    url: str
    token: str | None = None
    connect_timeout: float = 10.0
    start_attempts: int = 3
    start_retry_delay_seconds: float = 1.0
    start_retry_jitter_seconds: float = 0.25
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
        attempts = max(1, self.start_attempts)
        for attempt in range(1, attempts + 1):
            try:
                return await self._connect(tools_files=tools_files, workdir_path=workdir_path)
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_connect_error(exc):
                    raise
                jitter = random.uniform(0.0, self.start_retry_jitter_seconds)
                await asyncio.sleep(self.start_retry_delay_seconds * attempt + jitter)
        raise RuntimeError("unreachable remote backend retry state")

    async def _connect(
        self, *, tools_files: Mapping[str, str], workdir_path: str
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
            await ws.send(tools_tar)
        except BaseException:
            await ws.close()
            raise

        return _RemoteSandboxSession(
            ws=ws,
            workdir_path=workdir_path,
            max_upload_workspace_bytes=self.max_upload_workspace_bytes,
            max_download_workspace_bytes=self.max_download_workspace_bytes,
        )


class _RemoteSandboxSession:
    def __init__(
        self,
        *,
        ws: Any,
        workdir_path: str,
        max_upload_workspace_bytes: int,
        max_download_workspace_bytes: int,
    ) -> None:
        self._ws = ws
        self._workdir_path = workdir_path
        self._max_upload_workspace_bytes = max_upload_workspace_bytes
        self._max_download_workspace_bytes = max_download_workspace_bytes

    async def begin_block(self, input_paths: Sequence[str]) -> None:
        # Always send a (possibly empty) workspace tar so the per-block frame
        # order — binary tar, then RunFrame — stays fixed and the WS never desyncs.
        workspace_tar = _create_tar_from_paths(self._workdir_path, input_paths)
        if len(workspace_tar) > self._max_upload_workspace_bytes:
            raise ValueError(
                f"workspace archive ({len(workspace_tar):,} bytes) exceeds "
                f"max_upload_workspace_bytes ({self._max_upload_workspace_bytes:,})"
            )
        try:
            await self._ws.send(workspace_tar)
        except websockets.ConnectionClosed as exc:
            raise SandboxConnectionError("sandbox connection closed while staging inputs") from exc

    async def send(self, frame: Frame) -> None:
        payload = encode(frame)
        try:
            await self._ws.send(payload.decode("utf-8"))
        except websockets.ConnectionClosed as exc:
            raise SandboxConnectionError("sandbox connection closed while sending frame") from exc

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
        try:
            msg = await self._ws.recv()
        except websockets.ConnectionClosed as exc:
            raise SandboxConnectionError("sandbox connection closed before OutputFrame") from exc
        if not isinstance(msg, str):
            raise RuntimeError("expected text frame for OutputFrame")
        frame = decode(msg)
        if not isinstance(frame, OutputFrame):
            raise RuntimeError(f"expected OutputFrame, got {type(frame).__name__}")

        try:
            tar_data = await self._ws.recv()
        except websockets.ConnectionClosed as exc:
            raise SandboxConnectionError("sandbox connection closed before workspace tar") from exc
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
        # Best-effort graceful shutdown: a ShutdownFrame lets the container exit
        # cleanly; closing the WS is the backstop if the frame doesn't land.
        try:
            await self._ws.send(encode(ShutdownFrame()).decode("utf-8"))
        except Exception:
            pass
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


def _create_tar_from_paths(root: str, rel_paths: Sequence[str]) -> bytes:
    """Tar just the named files under ``root`` (this block's staged inputs)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in rel_paths:
            abs_path = os.path.join(root, rel)
            if os.path.islink(abs_path) or not os.path.isfile(abs_path):
                continue
            tf.add(abs_path, arcname=rel)
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
