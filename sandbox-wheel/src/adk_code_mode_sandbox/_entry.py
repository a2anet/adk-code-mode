# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Sandbox entry point.

Invoked as ``python -m adk_code_mode_sandbox``. Three control-pipe transports
are supported, chosen by env var (checked in order):

- ``ADK_CODE_MODE_CONTROL_TCP`` → ``host:port`` TCP endpoint on the host
  (used by ``UnsafeLocalDockerBackend``; works on mac/Windows/Linux uniformly).
- ``ADK_CODE_MODE_CONTROL_SOCKET`` → Unix domain socket path (used by the
  in-process test ``FakeRuntime``; broken on Docker Desktop macOS so not
  used by ``UnsafeLocalDockerBackend``).
- ``ADK_CODE_MODE_CONTROL_FD`` → inherited fd(s). Accepts one fd
  (bidirectional) or two comma-separated fds (read,write).

Installs the RPC client, announces readiness once, then loops handling ``run``
frames until a ``shutdown`` frame (or disconnect) ends the turn. Each block
runs against one persistent globals dict (state carries across the turn) with
its stdout/stderr captured and shipped back in a per-block ``OutputFrame``; any
``tools.*`` calls funnel through the RPC client.
"""

from __future__ import annotations

import io
import os
import socket
import sys
import traceback
from typing import Any

from adk_code_mode_sandbox import _rpc_client
from adk_code_mode_sandbox._rpc_client import RpcClient
from adk_code_mode_sandbox.protocol import (
    DoneFrame,
    OutputFrame,
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


def _prepare_sys_path(tools_dir: str | None = None) -> None:
    """Make the generated tools directory importable from user code."""
    target = tools_dir or TOOLS_DIR
    if os.path.isdir(target):
        # target is the package directory itself (normally /tools), not a
        # parent that contains tools/. Add the parent so `import tools` resolves
        # directly to /tools/__init__.py instead of requiring /tools/tools.
        package_parent = os.path.dirname(os.path.abspath(target)) or os.sep
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)


def _prepare_workdir(workdir: str | None = None) -> None:
    target = workdir or WORKDIR
    os.makedirs(target, exist_ok=True)
    os.chdir(target)


_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LANGUAGE",
        "TERM",
        "TMPDIR",
        "TEMP",
        "TMP",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
    }
)


def _sanitize_environ() -> None:
    """Strip sensitive env vars before user code runs.

    Preserves only a known-safe allowlist plus LC_* locale vars. Everything
    else (control tokens, platform credentials, extra_env from the host) is
    removed so user code cannot read it.
    """
    for key in list(os.environ):
        if key not in _ENV_ALLOWLIST and not key.startswith("LC_"):
            del os.environ[key]


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


def _exec_into(code: str, globs: dict[str, Any]) -> int:
    """Execute user code into ``globs``. Returns the exit code (0 ok, 1 on exception).

    ``globs`` is the persistent per-connection globals dict, so names bound by
    one code block are visible to later blocks in the same turn.
    """
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


def run_block(code: str, globs: dict[str, Any]) -> tuple[str, str, int]:
    """Run one code block with its stdout/stderr captured.

    Redirects ``sys.stdout`` / ``sys.stderr`` to in-memory buffers for the
    duration of the block so the captured text can be shipped back in an
    ``OutputFrame`` (both the TCP and HTTP transports emit one per block).
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    capture_out = io.StringIO()
    capture_err = io.StringIO()
    sys.stdout = capture_out
    sys.stderr = capture_err
    try:
        exit_code = _exec_into(code, globs)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return capture_out.getvalue(), capture_err.getvalue(), exit_code


def main() -> int:
    if os.environ.get("ADK_CODE_MODE_CONTROL_HTTP"):
        import asyncio

        from adk_code_mode_sandbox._http_server import AUTH_TOKEN_ENV, serve

        port = int(os.environ.get("PORT", "8080"))
        token = os.environ.get(AUTH_TOKEN_ENV)
        asyncio.run(serve(port, token))
        return 0

    reader, writer = _open_control_streams()
    client = RpcClient(reader=reader, writer=writer)
    _rpc_client.install(client)
    _prepare_sys_path()
    _prepare_workdir()
    _sanitize_environ()

    # One persistent globals dict + one /workspace for the whole connection, so
    # variables and files carry across the turn's successive code blocks.
    globs = _make_globals()
    client.send(ReadyFrame())

    while True:
        frame = client.recv()
        if isinstance(frame, ShutdownFrame):
            return 0
        if isinstance(frame, RunFrame):
            stdout_text, stderr_text, exit_code = run_block(frame.code, globs)
            client.send(DoneFrame(exit_code=exit_code))
            client.send(OutputFrame(stdout=stdout_text, stderr=stderr_text, exit_code=exit_code))
            continue
        # Any other frame kind is a host protocol error; keep going but log.
        print(
            f"[adk-code-mode-sandbox] unexpected frame kind: {type(frame).__name__}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
