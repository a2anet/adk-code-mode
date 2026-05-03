# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Sandbox runtime interfaces.

A runtime is responsible for:

- preparing an isolated environment with ``/tools`` and ``/workspace`` mounts
- launching ``python -m adk_code_mode_sandbox`` inside it
- exposing an async ``send(frame)`` / ``recv()`` control pipe
- draining stdout/stderr when the run ends
- tearing the environment down when the handle is closed

All public backends satisfy the same ``SandboxBackend`` protocol so the
executor can stay backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Mapping, Protocol, runtime_checkable

from adk_code_mode.runtime.protocol import Frame


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
    """Live session with a running sandbox."""

    async def send(self, frame: Frame) -> None: ...

    def frames(self) -> AsyncIterator[Frame]:
        """Yield frames arriving on the control pipe until EOF."""
        ...

    async def wait(self) -> SandboxResult:
        """Wait for the sandbox to exit and return captured output."""
        ...

    async def close(self) -> None:
        """Terminate the sandbox if still running; release resources."""
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
        """Launch a sandbox.

        Args:
            tools_files: posix-relative path → source, to be materialised into the
                sandbox's ``/tools`` directory.
            workdir_path: absolute path on the host to bind-mount into the
                sandbox at ``/workspace`` and use as the sandbox's cwd.
            timeout_seconds: if not ``None``, the runtime should kill the
                sandbox after this many seconds.
        """
        ...


__all__ = ["SandboxBackend", "SandboxFiles", "SandboxResult", "SandboxSession"]
