# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from adk_code_mode.runtime.protocol import (
    DoneFrame,
    ProtocolError,
    ReadyFrame,
    RunFrame,
    ToolCallFrame,
    ToolErrorPayload,
    ToolResultFrame,
    decode,
    encode,
)


def _roundtrip(frame: object) -> object:
    return decode(encode(frame))  # type: ignore[arg-type]


def test_ready_roundtrip() -> None:
    assert _roundtrip(ReadyFrame()) == ReadyFrame()


def test_run_roundtrip() -> None:
    assert _roundtrip(RunFrame(code="print('hi')")) == RunFrame(code="print('hi')")


def test_tool_call_roundtrip() -> None:
    src = ToolCallFrame(id="a1", name="slack.send", args={"x": 1}, timeout=5.0)
    assert _roundtrip(src) == src


def test_tool_result_ok_roundtrip() -> None:
    src = ToolResultFrame(id="a1", ok=True, value={"ok": True, "id": "m123"})
    assert _roundtrip(src) == src


def test_tool_result_error_roundtrip() -> None:
    src = ToolResultFrame(
        id="a1",
        ok=False,
        error=ToolErrorPayload(type="RuntimeError", message="boom", trace=None),
    )
    decoded = _roundtrip(src)
    assert isinstance(decoded, ToolResultFrame)
    assert decoded.ok is False
    assert decoded.error is not None
    assert decoded.error.type == "RuntimeError"
    assert decoded.error.message == "boom"


def test_done_roundtrip() -> None:
    assert _roundtrip(DoneFrame(exit_code=0)) == DoneFrame(exit_code=0)


def test_decode_rejects_unknown_kind() -> None:
    with pytest.raises(ProtocolError):
        decode(b'{"kind":"???"}\n')


def test_decode_rejects_empty() -> None:
    with pytest.raises(ProtocolError):
        decode(b"")


def test_decode_rejects_non_object() -> None:
    with pytest.raises(ProtocolError):
        decode(b"[1,2,3]\n")


def test_host_and_sandbox_protocol_byte_identical() -> None:
    """The two copies of protocol.py must be byte-identical."""
    from pathlib import Path

    host = Path("src/adk_code_mode/runtime/protocol.py").read_bytes()
    sandbox = Path("sandbox-wheel/src/adk_code_mode_sandbox/protocol.py").read_bytes()
    assert host == sandbox
