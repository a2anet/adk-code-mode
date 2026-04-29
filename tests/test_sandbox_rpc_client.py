# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

import pytest

from adk_code_mode_sandbox._rpc_client import RpcClient  # type: ignore[import-not-found]
from adk_code_mode_sandbox.protocol import (  # type: ignore[import-not-found]
    RunFrame,
    ToolResultFrame,
    encode,
)


class _Reader:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self._lock = threading.Lock()

    def readline(self) -> bytes:
        with self._lock:
            if not self._lines:
                return b""
            return self._lines.pop(0)


class _Writer:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self._lock = threading.Lock()

    def write(self, data: bytes) -> None:
        with self._lock:
            self.writes.append(data)

    def flush(self) -> None:
        return None


def test_rpc_client_demultiplexes_tool_results_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter(["call-a", "call-b"])
    monkeypatch.setattr(
        "adk_code_mode_sandbox._rpc_client.uuid.uuid4",
        lambda: type("_Uuid", (), {"hex": next(ids)})(),
    )

    reader = _Reader(
        [
            encode(ToolResultFrame(id="call-b", ok=True, value=2)),
            encode(ToolResultFrame(id="call-a", ok=True, value=1)),
        ]
    )
    writer = _Writer()
    client = RpcClient(reader=reader, writer=writer)

    results: dict[str, int] = {}

    def _run(label: str) -> None:
        results[label] = client.call(label, {})

    first = threading.Thread(target=_run, args=("first",))
    second = threading.Thread(target=_run, args=("second",))
    first.start()
    second.start()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(results.values()) == [1, 2]
    assert len(writer.writes) == 2


def test_rpc_client_recv_still_surfaces_non_tool_frames() -> None:
    reader = _Reader([encode(RunFrame(code="print('ok')"))])
    client = RpcClient(reader=reader, writer=_Writer())
    frame = client.recv()
    assert isinstance(frame, RunFrame)
    assert frame.code == "print('ok')"
