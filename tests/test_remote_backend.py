# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``RemoteBackend``'s built-in connect retry."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence

import pytest
from pytest import MonkeyPatch

from adk_code_mode import RemoteBackend
from adk_code_mode.runtime.base import SandboxResult, SandboxSession
from adk_code_mode.runtime.protocol import Frame


class _NoopSession:
    async def begin_block(self, input_paths: Sequence[str]) -> None:
        return None

    async def send(self, frame: Frame) -> None:
        return None

    async def frames(self) -> AsyncIterator[Frame]:
        return
        yield  # pragma: no cover  (makes this an async generator)

    async def wait(self) -> SandboxResult:
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_start_retries_transient_connect_error(monkeypatch: MonkeyPatch) -> None:
    session = _NoopSession()
    calls = 0
    delays: list[float] = []

    async def fake_connect(*, tools_files: Mapping[str, str], workdir_path: str) -> SandboxSession:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out during opening handshake")
        return session

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    backend = RemoteBackend(
        url="ws://sandbox.example.test",
        start_retry_delay_seconds=0.25,
        start_retry_jitter_seconds=0.0,
    )
    monkeypatch.setattr(backend, "_connect", fake_connect)
    monkeypatch.setattr("adk_code_mode.runtime.remote.asyncio.sleep", fake_sleep)

    result = await backend.start(tools_files={}, workdir_path="/tmp", timeout_seconds=None)

    assert result is session
    assert calls == 2
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_start_does_not_retry_non_retryable_error(monkeypatch: MonkeyPatch) -> None:
    calls = 0

    async def fake_connect(*, tools_files: Mapping[str, str], workdir_path: str) -> SandboxSession:
        nonlocal calls
        calls += 1
        raise ValueError("tools archive exceeds limit")

    backend = RemoteBackend(url="ws://sandbox.example.test", start_attempts=3)
    monkeypatch.setattr(backend, "_connect", fake_connect)

    with pytest.raises(ValueError, match="tools archive"):
        await backend.start(tools_files={}, workdir_path="/tmp", timeout_seconds=None)

    assert calls == 1
