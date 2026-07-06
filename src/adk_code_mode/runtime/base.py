# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Sandbox runtime interfaces.

A runtime is responsible for:

- preparing an isolated environment with ``/tools`` and ``/workspace`` mounts
- launching ``python -m adk_code_mode_sandbox`` inside it
- exposing an async ``send(frame)`` / ``recv()`` control pipe
- capturing each block's stdout/stderr as an ``OutputFrame``
- tearing the environment down when the handle is closed

A session is held open for the duration of a **turn** (one ADK invocation) and
runs one or more code blocks against a persistent process + ``/workspace``. All
public backends satisfy the same ``SandboxBackend`` protocol so the executor can
stay backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import AsyncIterator, Mapping, Protocol, runtime_checkable

from adk_code_mode.runtime.protocol import Frame


class SandboxConnectionError(Exception):
    """The sandbox control connection dropped mid-turn.

    Raised by a session when the connection is lost while running a block (as
    opposed to a clean per-block ``OutputFrame``). The executor drops the cached
    session and reconnects with a fresh ``start()``; in-turn state is lost.
    """


@dataclass(frozen=True)
class SandboxResult:
    """Captured output after a run."""

    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True)
class SandboxFiles:
    """In-memory file tree to mount into the sandbox.

    Maps posix paths (relative to the mount point) to UTF-8 source bytes. The
    runtime materialises these before the sandbox starts.
    """

    files: Mapping[str, str]


class SandboxSession(Protocol):
    """Live session with a running sandbox that runs N code blocks per turn.

    Per-block contract, driven by the executor and implemented identically by
    every backend: ``begin_block`` (stage inputs) → ``send`` the ``RunFrame`` →
    drain ``frames`` until the ``DoneFrame`` → ``wait`` (read the block's
    ``OutputFrame``). ``close`` ends the turn.
    """

    async def begin_block(self, input_paths: Sequence[str]) -> None:
        """Stage this block's freshly-written inputs into the sandbox.

        Called immediately before the block's ``RunFrame``. ``input_paths`` are
        posix-relative paths under the turn workspace that were staged for this
        block. Tar-based backends (``RemoteBackend``) upload just those files;
        mount-based backends (Docker) are a no-op — the workspace is bind-mounted.
        """
        ...

    async def send(self, frame: Frame) -> None: ...

    def frames(self) -> AsyncIterator[Frame]:
        """Yield frames arriving on the control pipe until this block's ``DoneFrame``/EOF."""
        ...

    async def wait(self) -> SandboxResult:
        """Read this block's ``OutputFrame`` (tar backends also pull the updated workspace)."""
        ...

    async def close(self) -> None:
        """Send a ``ShutdownFrame`` and tear the sandbox down; release resources."""
        ...


@runtime_checkable
class SandboxBackend(Protocol):
    """Factory for ``SandboxSession`` instances."""

    async def start(
        self,
        *,
        tools_files: Mapping[str, str],
        workdir_path: str,
        timeout_seconds: int | None,
    ) -> SandboxSession:
        """Launch a sandbox and connect, sending the tools once.

        Called on the first code block of a turn; the returned session then runs
        the turn's subsequent blocks without reconnecting.

        Args:
            tools_files: posix-relative path → source, to be materialised into the
                sandbox's ``/tools`` directory once at connect time.
            workdir_path: absolute path on the host used as the turn's workspace —
                bind-mounted into the sandbox at ``/workspace`` (Docker) or synced
                as tar archives (``RemoteBackend``), and used as the sandbox's cwd.
            timeout_seconds: if not ``None``, the runtime should kill the
                sandbox after this many seconds.
        """
        ...


__all__ = [
    "SandboxBackend",
    "SandboxConnectionError",
    "SandboxFiles",
    "SandboxResult",
    "SandboxSession",
]
