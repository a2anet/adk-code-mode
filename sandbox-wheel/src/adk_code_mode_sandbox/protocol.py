# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""JSON-lines wire protocol between the host and the sandbox.

This module is the shared kernel: a byte-identical copy lives in the
``adk_code_mode_sandbox`` wheel. A CI check enforces the two files are
identical on every release.

Design rules:

- Zero dependencies outside the Python standard library. The sandbox wheel
  must remain stdlib-only so the container image does not pull ADK / google-genai.
- Frames are JSON objects, one per newline-terminated UTF-8 line.
- Every frame carries a ``kind`` discriminator. Correlated pairs (e.g.
  ``tool_call`` / ``tool_result``) carry a string ``id``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PROTOCOL_VERSION = 1


FrameKind = Literal[
    "ready",
    "run",
    "tool_call",
    "tool_result",
    "log",
    "done",
    "shutdown",
]


@dataclass(frozen=True)
class ReadyFrame:
    """Sandbox → host. Sent once after boot."""

    kind: Literal["ready"] = "ready"
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class RunFrame:
    """Host → sandbox. Instructs the sandbox to execute the given code."""

    code: str = ""
    kind: Literal["run"] = "run"


@dataclass(frozen=True)
class ToolCallFrame:
    """Sandbox → host. Request to invoke a host-side tool."""

    id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = None
    kind: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True)
class ToolErrorPayload:
    """The structured error body inside a failing ``tool_result`` frame."""

    type: str
    message: str
    trace: str | None = None


@dataclass(frozen=True)
class ToolResultFrame:
    """Host → sandbox. Result for a previously issued ``tool_call``."""

    id: str = ""
    ok: bool = True
    value: Any = None
    error: ToolErrorPayload | None = None
    kind: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True)
class LogFrame:
    """Sandbox → host. Optional structured log line."""

    level: str = "info"
    msg: str = ""
    kind: Literal["log"] = "log"


@dataclass(frozen=True)
class DoneFrame:
    """Sandbox → host. User code finished. Host drains stdout/stderr next."""

    exit_code: int = 0
    kind: Literal["done"] = "done"


@dataclass(frozen=True)
class ShutdownFrame:
    """Host → sandbox. Graceful shutdown."""

    kind: Literal["shutdown"] = "shutdown"


Frame = (
    ReadyFrame | RunFrame | ToolCallFrame | ToolResultFrame | LogFrame | DoneFrame | ShutdownFrame
)


_KIND_TO_CLS: dict[str, type] = {
    "ready": ReadyFrame,
    "run": RunFrame,
    "tool_call": ToolCallFrame,
    "tool_result": ToolResultFrame,
    "log": LogFrame,
    "done": DoneFrame,
    "shutdown": ShutdownFrame,
}


class ProtocolError(Exception):
    """Raised when a frame cannot be decoded or is semantically invalid."""


def encode(frame: Frame) -> bytes:
    """Serialise a frame to a newline-terminated UTF-8 byte string."""
    payload = asdict(frame)
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes | str) -> Frame:
    """Deserialise a single frame line.

    Accepts either bytes or str; lines may or may not have a trailing newline.
    """
    if isinstance(line, (bytes, bytearray)):
        text = line.decode("utf-8")
    else:
        text = line
    text = text.rstrip("\r\n")
    if not text:
        raise ProtocolError("empty frame")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"frame must be a JSON object, got {type(obj).__name__}")
    kind = obj.get("kind")
    if not isinstance(kind, str):
        raise ProtocolError("frame missing string `kind`")
    cls = _KIND_TO_CLS.get(kind)
    if cls is None:
        raise ProtocolError(f"unknown frame kind: {kind!r}")
    if cls is ToolResultFrame and obj.get("error") is not None:
        err = obj["error"]
        if not isinstance(err, dict):
            raise ProtocolError("tool_result.error must be an object")
        obj = dict(obj)
        obj["error"] = ToolErrorPayload(
            type=str(err.get("type", "Error")),
            message=str(err.get("message", "")),
            trace=err.get("trace"),
        )
    valid_fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in obj.items() if k in valid_fields}
    try:
        return cls(**filtered)  # type: ignore[no-any-return]
    except TypeError as exc:
        raise ProtocolError(f"invalid frame body for kind {kind!r}: {exc}") from exc


__all__ = [
    "PROTOCOL_VERSION",
    "FrameKind",
    "Frame",
    "ReadyFrame",
    "RunFrame",
    "ToolCallFrame",
    "ToolResultFrame",
    "ToolErrorPayload",
    "LogFrame",
    "DoneFrame",
    "ShutdownFrame",
    "ProtocolError",
    "encode",
    "decode",
]
