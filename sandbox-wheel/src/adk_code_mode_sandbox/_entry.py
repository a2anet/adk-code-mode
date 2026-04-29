# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Sandbox entry point.

Invoked as ``python -m adk_code_mode_sandbox``. Three control-pipe transports
are supported, chosen by env var (checked in order):

- ``ADK_CODE_MODE_CONTROL_TCP`` → ``host:port`` TCP endpoint on the host
  (used by the Docker runtime; works on mac/Windows/Linux uniformly).
- ``ADK_CODE_MODE_CONTROL_SOCKET`` → Unix domain socket path (used by the
  in-process test ``FakeRuntime``; broken on Docker Desktop macOS so not
  used by ``DockerRuntime``).
- ``ADK_CODE_MODE_CONTROL_FD`` → inherited fd(s). Accepts one fd
  (bidirectional) or two comma-separated fds (read,write).

Installs the RPC client, announces readiness, then waits for a ``run`` frame.
On receipt, execs the user code with stdout/stderr on the normal streams and
any ``tools.*`` calls funnelling through the RPC client.
"""

from __future__ import annotations

import os
import socket
import sys
import traceback
from typing import Any

from adk_code_mode_sandbox import _rpc_client
from adk_code_mode_sandbox._rpc_client import RpcClient
from adk_code_mode_sandbox.protocol import (
    DoneFrame,
    ReadyFrame,
    RunFrame,
    ShutdownFrame,
)

CONTROL_FD_ENV = "ADK_CODE_MODE_CONTROL_FD"
CONTROL_SOCKET_ENV = "ADK_CODE_MODE_CONTROL_SOCKET"
CONTROL_TCP_ENV = "ADK_CODE_MODE_CONTROL_TCP"
CONTROL_TOKEN_ENV = "ADK_CODE_MODE_CONTROL_TOKEN"
TOOLS_DIR_ENV = "ADK_CODE_MODE_TOOLS_DIR"
WORKDIR_ENV = "ADK_CODE_MODE_WORKDIR"
TOOLS_DIR = os.environ.get(TOOLS_DIR_ENV, "/tools")
WORKDIR = os.environ.get(WORKDIR_ENV, "/workspace")


def _open_control_streams() -> tuple[Any, Any]:
    """Open the host control pipe as a (reader, writer) pair."""
    tcp_endpoint = os.environ.get(CONTROL_TCP_ENV)
    if tcp_endpoint:
        host, _, port_s = tcp_endpoint.rpartition(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port_s)))
        token = os.environ.get(CONTROL_TOKEN_ENV, "")
        sock.sendall(token.encode("ascii") + b"\n")
        return sock.makefile("rb", buffering=0), sock.makefile("wb", buffering=0)
    sock_path = os.environ.get(CONTROL_SOCKET_ENV)
    if sock_path:
        usock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        usock.connect(sock_path)
        return usock.makefile("rb", buffering=0), usock.makefile("wb", buffering=0)
    raw = os.environ.get(CONTROL_FD_ENV, "3")
    if "," in raw:
        read_fd_s, write_fd_s = raw.split(",", 1)
        read_fd, write_fd = int(read_fd_s), int(write_fd_s)
    else:
        read_fd = write_fd = int(raw)
    return (
        os.fdopen(read_fd, "rb", buffering=0, closefd=False),
        os.fdopen(write_fd, "wb", buffering=0, closefd=False),
    )


def _prepare_sys_path() -> None:
    """Make the generated tools directory importable from user code."""
    if os.path.isdir(TOOLS_DIR):
        # TOOLS_DIR is the package directory itself (normally /tools), not a
        # parent that contains tools/. Add the parent so `import tools` resolves
        # directly to /tools/__init__.py instead of requiring /tools/tools.
        package_parent = os.path.dirname(os.path.abspath(TOOLS_DIR)) or os.sep
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)


def _prepare_workdir() -> None:
    os.makedirs(WORKDIR, exist_ok=True)
    os.chdir(WORKDIR)


def _make_globals() -> dict[str, Any]:
    """Build the globals dict for ``exec``. Looks like a normal ``__main__``."""
    return {
        "__name__": "__main__",
        "__doc__": None,
        "__package__": None,
        "__loader__": None,
        "__spec__": None,
        "__builtins__": __builtins__,
    }


def _run_code(code: str) -> int:
    """Execute user code. Returns the exit code (0 ok, 1 on exception)."""
    globs = _make_globals()
    try:
        compiled = compile(code, "<code-mode>", "exec")
    except SyntaxError:
        traceback.print_exc()
        return 1
    try:
        exec(compiled, globs)
    except SystemExit as exc:
        code_val = exc.code
        if code_val is None:
            return 0
        if isinstance(code_val, int):
            return code_val
        print(code_val, file=sys.stderr)
        return 1
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


def main() -> int:
    reader, writer = _open_control_streams()
    client = RpcClient(reader=reader, writer=writer)
    _rpc_client.install(client)
    _prepare_sys_path()
    _prepare_workdir()

    client.send(ReadyFrame())

    while True:
        frame = client.recv()
        if isinstance(frame, ShutdownFrame):
            return 0
        if isinstance(frame, RunFrame):
            exit_code = _run_code(frame.code)
            sys.stdout.flush()
            sys.stderr.flush()
            client.send(DoneFrame(exit_code=exit_code))
            continue
        # Any other frame kind is a host protocol error; keep going but log.
        print(
            f"[adk-code-mode-sandbox] unexpected frame kind: {type(frame).__name__}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
