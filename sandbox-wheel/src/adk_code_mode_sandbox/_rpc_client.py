# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Blocking RPC client used by generated tool stubs inside the sandbox.

All tool stubs funnel through ``call(name, args)``. The global client instance
is installed by ``_entry`` before user code runs. Stubs import it lazily:

.. code-block:: python

    from adk_code_mode_sandbox._rpc_client import call
    result = call("slack.send_message", {"channel": "C1", "text": "hi"})
"""

from __future__ import annotations

from collections import deque
import threading
import uuid
from typing import Any

from adk_code_mode_sandbox.protocol import (
    Frame,
    ProtocolError,
    ToolCallFrame,
    ToolResultFrame,
    decode,
    encode,
)


class ToolError(RuntimeError):
    """Raised inside the sandbox when a host-side tool call fails."""

    def __init__(self, error_type: str, message: str, trace: str | None = None) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.trace = trace


class RpcClient:
    """Synchronous, thread-safe RPC client over the sandbox control pipe.

    The control pipe is a pair of unbuffered streams: ``_out`` for frames going
    to the host and ``_in`` for frames coming back. Both are line-oriented.
    """

    def __init__(self, *, reader: Any, writer: Any) -> None:
        self._reader = reader
        self._writer = writer
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state_changed = threading.Condition(self._state_lock)
        self._general_frames: deque[Frame] = deque()
        self._pending_results: dict[str, ToolResultFrame] = {}
        self._closed_error: Exception | None = None
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def send(self, frame: Frame) -> None:
        payload = encode(frame)
        with self._write_lock:
            self._writer.write(payload)
            self._writer.flush()

    def recv(self) -> Frame:
        with self._state_changed:
            while not self._general_frames and self._closed_error is None:
                self._state_changed.wait()
            if self._general_frames:
                return self._general_frames.popleft()
            raise self._closed_error or ProtocolError("control pipe closed by host")

    def call(self, name: str, args: dict[str, Any], timeout: float | None = None) -> Any:
        call_id = uuid.uuid4().hex
        self.send(ToolCallFrame(id=call_id, name=name, args=args, timeout=timeout))
        with self._state_changed:
            while call_id not in self._pending_results and self._closed_error is None:
                self._state_changed.wait()
            frame = self._pending_results.pop(call_id, None)
            if frame is None:
                raise self._closed_error or ProtocolError("control pipe closed by host")
            if frame.ok:
                return frame.value
            err = frame.error
            if err is None:
                raise ToolError("Error", "tool call failed without error payload")
            raise ToolError(err.type, err.message, err.trace)

    def _read_loop(self) -> None:
        try:
            while True:
                line = self._reader.readline()
                if not line:
                    raise ProtocolError("control pipe closed by host")
                frame = decode(line)
                with self._state_changed:
                    if isinstance(frame, ToolResultFrame):
                        self._pending_results[frame.id] = frame
                    else:
                        self._general_frames.append(frame)
                    self._state_changed.notify_all()
        except Exception as exc:
            with self._state_changed:
                self._closed_error = exc
                self._state_changed.notify_all()


_CLIENT: RpcClient | None = None


def install(client: RpcClient) -> None:
    """Set the process-wide RPC client. Called by ``_entry`` at boot."""
    global _CLIENT
    _CLIENT = client


def get() -> RpcClient:
    if _CLIENT is None:
        raise RuntimeError("RPC client not installed; sandbox is not in a valid state")
    return _CLIENT


def call(name: str, args: dict[str, Any], timeout: float | None = None) -> Any:
    """Convenience: invoke a tool through the installed client."""
    return get().call(name, args, timeout)


__all__ = ["RpcClient", "ToolError", "call", "get", "install"]
