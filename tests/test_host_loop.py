# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the host-side frame loop in ``executor``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from adk_code_mode.executor import (
    ProtocolVersionMismatchError,
    _handle_tool_call,
    _host_loop,
    _json_safe,
)
from adk_code_mode.runtime.base import SandboxResult
from adk_code_mode.runtime.protocol import (
    PROTOCOL_VERSION,
    DoneFrame,
    Frame,
    ReadyFrame,
    ToolCallFrame,
    ToolResultFrame,
)
from adk_code_mode.tools.dispatcher import DispatchResult


class _StubHandle:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames
        self.sent: list[Frame] = []

    async def send(self, frame: Frame) -> None:
        self.sent.append(frame)

    def frames(self) -> AsyncIterator[Frame]:
        async def _iter() -> AsyncIterator[Frame]:
            for f in self._frames:
                yield f

        return _iter()

    async def wait(self) -> SandboxResult:
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def close(self) -> None:
        return None


async def test_host_loop_accepts_matching_protocol_version() -> None:
    handle = _StubHandle([ReadyFrame(), DoneFrame()])
    await _host_loop(session=handle, dispatcher=None)  # type: ignore[arg-type]


async def test_host_loop_rejects_mismatched_protocol_version() -> None:
    handle = _StubHandle([ReadyFrame(protocol_version=PROTOCOL_VERSION + 1), DoneFrame()])
    with pytest.raises(ProtocolVersionMismatchError):
        await _host_loop(session=handle, dispatcher=None)  # type: ignore[arg-type]


def test_json_safe_handles_common_non_json_leaf_types() -> None:
    @dataclass
    class _Payload:
        blob: bytes
        day: date
        price: Decimal
        path: Path

    value = _json_safe(
        _Payload(
            blob=b"\x00\x01",
            day=date(2026, 4, 28),
            price=Decimal("12.34"),
            path=Path("out/report.txt"),
        )
    )

    assert value == {
        "blob": {"kind": "bytes", "data": "AAE=", "encoding": "base64"},
        "day": "2026-04-28",
        "price": "12.34",
        "path": "out/report.txt",
    }


class _FailFirstSendHandle:
    def __init__(self) -> None:
        self.sent: list[Frame] = []
        self._failed = False

    async def send(self, frame: Frame) -> None:
        if not self._failed:
            self._failed = True
            raise TypeError("not json serialisable")
        self.sent.append(frame)

    def frames(self) -> AsyncIterator[Frame]:
        async def _iter() -> AsyncIterator[Frame]:
            if False:
                yield ReadyFrame()

        return _iter()

    async def wait(self) -> SandboxResult:
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def close(self) -> None:
        return None


class _FakeDispatcher:
    async def dispatch(
        self, name: str, args: dict[str, Any], timeout: float | None = None
    ) -> DispatchResult:
        return DispatchResult(ok=True, value={"ok": True})


async def test_handle_tool_call_sends_fallback_error_when_primary_send_fails() -> None:
    handle = _FailFirstSendHandle()

    await _handle_tool_call(
        handle,
        _FakeDispatcher(),  # type: ignore[arg-type]
        ToolCallFrame(id="call-1", name="tool", args={}),
    )

    assert len(handle.sent) == 1
    fallback = handle.sent[0]
    assert isinstance(fallback, ToolResultFrame)
    assert fallback.id == "call-1"
    assert fallback.ok is False
    assert fallback.error is not None
    assert "failed to serialise or send tool result" in fallback.error.message
