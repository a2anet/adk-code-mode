# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Docker-backed sandbox runtime.

The host:

1. materialises the tools tree into a tempdir,
2. opens a TCP listener on the host (ephemeral port),
3. bind-mounts the tools tree and execution workspace into the container,
4. starts ``python -m adk_code_mode_sandbox`` with the host endpoint in the env,
5. the container reaches the listener via ``host.docker.internal`` (Docker
   Desktop on mac/Windows provides this automatically; on Linux we add it as
   an extra host alias of the host gateway),
6. accepts a single connection from the sandbox and streams frames.

TCP (not Unix domain sockets) is used because UDS bind-mounts are broken on
Docker Desktop macOS — a long-standing virtiofs limitation.

By default the container is given network access only to the host gateway
via the default bridge network. For stricter setups, pass a custom network
through ``run_kwargs``.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import secrets
import shutil
import socket
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Mapping

from adk_code_mode.runtime.base import SandboxSession, SandboxResult
from adk_code_mode.runtime.protocol import Frame, ProtocolError, decode, encode

_CONTAINER_TOOLS_MOUNT = "/tools"
_CONTAINER_WORKDIR_MOUNT = "/workspace"
_HOST_ALIAS = "host.docker.internal"
_TOKEN_ENV = "ADK_CODE_MODE_CONTROL_TOKEN"
_TOKEN_HANDSHAKE_TIMEOUT_S = 5.0


@dataclass
class UnsafeLocalDockerBackend:
    """Docker-backed :class:`SandboxBackend` for **local development only**.

    Args:
        image: Fully-qualified container image tag. Must have
            ``adk-code-mode-sandbox`` installed. Extend the published
            ``ghcr.io/a2anet/adk-code-mode`` base image to bake in additional
            Python packages.
        network_mode: Docker network mode. Defaults to ``None``, which
            means the kwarg is not passed to ``containers.run`` and
            Docker picks the default bridge network so the container can
            reach the host's TCP control listener. Override to use a
            custom network.
        mem_limit: Container memory limit. Defaults to ``"1g"``; pass
            ``None`` to remove the cap.
        cpu_period / cpu_quota: CPU throttle. Default to one full vCPU
            (period 100_000 µs, quota 100_000 µs); pass ``None`` to
            remove the cap.
        read_only: Whether to mount the container rootfs read-only. Default True.
        extra_env: Extra environment variables to set inside the container.
        run_kwargs: Free-form overrides passed to ``docker_client.containers.run``.
    """

    image: str
    network_mode: str | None = None
    mem_limit: str | None = "1g"
    cpu_period: int | None = 100_000
    cpu_quota: int | None = 100_000
    read_only: bool = True
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    extra_env: Mapping[str, str] = field(default_factory=dict)
    run_kwargs: Mapping[str, Any] = field(default_factory=dict)

    async def start(
        self,
        *,
        tools_files: Mapping[str, str],
        workdir_path: str,
        timeout_seconds: int | None,
    ) -> SandboxSession:
        import docker  # local import so the rest of the package works without it

        if self.network_mode == "none":
            raise ValueError(
                "UnsafeLocalDockerBackend requires container network access to reach the host control listener; "
                "network_mode='none' is unsupported"
            )

        loop = asyncio.get_running_loop()
        tools_dir = tempfile.mkdtemp(prefix="adk-code-mode-tools-")
        listener: socket.socket | None = None
        try:
            _materialise_tools(tools_files, tools_dir)

            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # The container connects via host.docker.internal / host-gateway, so
            # the listener must be reachable off-host-loopback. A per-start token
            # gates first-frame access — see _accept_connection.
            listener.bind(("0.0.0.0", 0))
            listener.listen(1)
            host_port = listener.getsockname()[1]
            control_token = secrets.token_urlsafe(32)

            env = {
                "ADK_CODE_MODE_CONTROL_TCP": f"{_HOST_ALIAS}:{host_port}",
                _TOKEN_ENV: control_token,
                "ADK_CODE_MODE_WORKDIR": _CONTAINER_WORKDIR_MOUNT,
                **dict(self.extra_env),
            }

            volumes = {
                tools_dir: {"bind": _CONTAINER_TOOLS_MOUNT, "mode": "ro"},
                workdir_path: {"bind": _CONTAINER_WORKDIR_MOUNT, "mode": "rw"},
            }

            def _run_container() -> Any:
                client = docker.from_env()
                kwargs: dict[str, Any] = {
                    "detach": True,
                    "mem_limit": self.mem_limit,
                    "cpu_period": self.cpu_period,
                    "cpu_quota": self.cpu_quota,
                    "read_only": self.read_only,
                    "cap_drop": self.cap_drop,
                    "environment": env,
                    "volumes": volumes,
                    "stdout": True,
                    "stderr": True,
                    "tty": False,
                    "stdin_open": False,
                    "working_dir": _CONTAINER_WORKDIR_MOUNT,
                    **self.run_kwargs,
                }
                if sys.platform.startswith("linux"):
                    kwargs["extra_hosts"] = {_HOST_ALIAS: "host-gateway"}
                if self.network_mode is not None:
                    kwargs["network_mode"] = self.network_mode
                return client.containers.run(
                    self.image,
                    command=["python", "-m", "adk_code_mode_sandbox"],
                    **kwargs,
                )

            container = await loop.run_in_executor(None, _run_container)

            try:
                conn = await loop.run_in_executor(
                    None, _accept_connection, listener, 30.0, control_token
                )
            except BaseException:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
                raise

            return _DockerSandboxSession(
                container=container,
                control_sock=conn,
                listener=listener,
                tools_dir=tools_dir,
                timeout_seconds=timeout_seconds,
            )
        except BaseException:
            if listener is not None:
                try:
                    listener.close()
                except OSError:
                    pass
            shutil.rmtree(tools_dir, ignore_errors=True)
            raise


def _materialise_tools(files: Mapping[str, str], root: str) -> None:
    for rel, source in files.items():
        dest = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(source)


def _accept_connection(
    listener: socket.socket, timeout: float, expected_token: str
) -> socket.socket:
    """Accept connections until one presents the expected token as its first line.

    Bad connections are dropped silently so a process racing the sandbox onto the
    listener cannot lock the real sandbox out — accept loops until ``timeout``.
    """
    deadline = time.monotonic() + timeout
    expected = expected_token.encode("ascii")
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining == 0.0:
            raise TimeoutError("timed out waiting for sandbox to authenticate")
        listener.settimeout(remaining)
        try:
            conn, _ = listener.accept()
        finally:
            listener.settimeout(None)
        try:
            conn.settimeout(_TOKEN_HANDSHAKE_TIMEOUT_S)
            line = _read_line(conn, max_bytes=512)
        except (OSError, TimeoutError):
            conn.close()
            continue
        if hmac.compare_digest(line, expected):
            conn.setblocking(False)
            return conn
        conn.close()


def _read_line(conn: socket.socket, *, max_bytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < max_bytes:
        chunk = conn.recv(1)
        if not chunk:
            raise OSError("peer closed before sending token")
        if chunk == b"\n":
            return bytes(buf)
        buf.extend(chunk)
    raise OSError("token line exceeded max length")


class _DockerSandboxSession(SandboxSession):
    def __init__(
        self,
        *,
        container: Any,
        control_sock: socket.socket,
        listener: socket.socket,
        tools_dir: str,
        timeout_seconds: int | None,
    ) -> None:
        self._container = container
        self._sock = control_sock
        self._listener = listener
        self._tools_dir = tools_dir
        self._timeout = timeout_seconds
        self._recv_buf = b""
        self._lock = asyncio.Lock()
        self._closed = False

    async def send(self, frame: Frame) -> None:
        payload = encode(frame)
        async with self._lock:
            await asyncio.get_running_loop().sock_sendall(self._sock, payload)

    async def frames(self) -> AsyncIterator[Frame]:
        loop = asyncio.get_running_loop()
        while True:
            while b"\n" not in self._recv_buf:
                chunk = await loop.sock_recv(self._sock, 65536)
                if not chunk:
                    return
                self._recv_buf += chunk
            line, self._recv_buf = self._recv_buf.split(b"\n", 1)
            if not line:
                continue
            try:
                yield decode(line)
            except ProtocolError:
                continue

    async def wait(self) -> SandboxResult:
        loop = asyncio.get_running_loop()

        def _wait() -> dict[str, Any]:
            return self._container.wait(timeout=self._timeout) or {}

        try:
            result = await loop.run_in_executor(None, _wait)
        except Exception:
            result = {"StatusCode": -1}
        exit_code = int(result.get("StatusCode", 0))

        # docker-py's Container.logs() does not support demux, so stdout and stderr
        # have to be drained separately.
        def _logs_stdout() -> bytes:
            return self._container.logs(stdout=True, stderr=False) or b""

        def _logs_stderr() -> bytes:
            return self._container.logs(stdout=False, stderr=True) or b""

        stdout_b = await loop.run_in_executor(None, _logs_stdout)
        stderr_b = await loop.run_in_executor(None, _logs_stderr)
        return SandboxResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=exit_code,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = asyncio.get_running_loop()
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            self._listener.close()
        except OSError:
            pass

        def _kill_and_remove() -> None:
            try:
                self._container.reload()
                if self._container.status == "running":
                    self._container.kill()
            except Exception:
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass

        await loop.run_in_executor(None, _kill_and_remove)
        shutil.rmtree(self._tools_dir, ignore_errors=True)


__all__ = ["UnsafeLocalDockerBackend"]
