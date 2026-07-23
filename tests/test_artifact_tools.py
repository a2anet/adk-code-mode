# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``save_artifact`` / ``load_artifact`` / ``list_artifacts``
``FunctionTool`` wrappers.

These exercise the codec logic (text vs binary, base64 round-trip, missing
artifact) against a stub ``tool_context`` that records and replays calls.
End-to-end coverage through the sandbox lives in ``test_integration.py``.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from google.genai import types as genai_types

from adk_code_mode._artifact_tools import (
    ARTIFACT_TOOLS,
    list_artifacts,
    load_artifact,
    save_artifact,
)


class _StubToolContext:
    def __init__(
        self,
        *,
        load_result: genai_types.Part | None = None,
        list_result: list[str] | None = None,
    ) -> None:
        self._load_result = load_result
        self._list_result = list_result or []
        self.saved: list[tuple[str, genai_types.Part, dict[str, Any] | None]] = []
        self.loaded: list[tuple[str, int | None]] = []
        self.listed = 0

    async def save_artifact(
        self,
        filename: str,
        part: genai_types.Part,
        custom_metadata: dict[str, Any] | None = None,
    ) -> int:
        self.saved.append((filename, part, custom_metadata))
        return len(self.saved)

    async def load_artifact(
        self, filename: str, *, version: int | None = None
    ) -> genai_types.Part | None:
        self.loaded.append((filename, version))
        return self._load_result

    async def list_artifacts(self) -> list[str]:
        self.listed += 1
        return self._list_result


@pytest.mark.asyncio
async def test_save_artifact_encodes_text_mime_as_utf8() -> None:
    ctx = _StubToolContext()
    version = await save_artifact(
        filename="report.json",
        content='{"ok": true}',
        mime_type="application/json",
        tool_context=ctx,  # type: ignore[arg-type]
    )

    assert version == 1
    assert len(ctx.saved) == 1
    name, part, _ = ctx.saved[0]
    assert name == "report.json"
    assert part.inline_data is not None
    assert part.inline_data.mime_type == "application/json"
    assert part.inline_data.data == b'{"ok": true}'


@pytest.mark.asyncio
async def test_save_artifact_treats_text_plus_subtype_as_text() -> None:
    ctx = _StubToolContext()
    await save_artifact(
        filename="payload.svg",
        content="<svg/>",
        mime_type="image/svg+xml",
        tool_context=ctx,  # type: ignore[arg-type]
    )

    _, part, _ = ctx.saved[0]
    assert part.inline_data is not None
    assert part.inline_data.data == b"<svg/>"


@pytest.mark.asyncio
async def test_save_artifact_decodes_base64_for_binary_mime() -> None:
    raw = b"\x00\x01\x02\xff"
    ctx = _StubToolContext()
    await save_artifact(
        filename="blob.bin",
        content=base64.b64encode(raw).decode("ascii"),
        mime_type="application/octet-stream",
        tool_context=ctx,  # type: ignore[arg-type]
    )

    _, part, _ = ctx.saved[0]
    assert part.inline_data is not None
    assert part.inline_data.data == raw
    assert part.inline_data.mime_type == "application/octet-stream"


@pytest.mark.asyncio
async def test_save_artifact_defaults_mime_when_none() -> None:
    ctx = _StubToolContext()
    raw = b"\x10\x20"
    await save_artifact(
        filename="anon.bin",
        content=base64.b64encode(raw).decode("ascii"),
        tool_context=ctx,  # type: ignore[arg-type]
    )

    _, part, _ = ctx.saved[0]
    assert part.inline_data is not None
    assert part.inline_data.mime_type == "application/octet-stream"
    assert part.inline_data.data == raw


@pytest.mark.asyncio
async def test_save_artifact_forwards_custom_metadata() -> None:
    ctx = _StubToolContext()
    await save_artifact(
        filename="report.md",
        content="# Report",
        mime_type="text/markdown",
        custom_metadata={"a2anet_agent_run.send_as_a2a_artifact": "true"},
        tool_context=ctx,  # type: ignore[arg-type]
    )

    _, _, custom_metadata = ctx.saved[0]
    assert custom_metadata == {"a2anet_agent_run.send_as_a2a_artifact": "true"}


@pytest.mark.asyncio
async def test_save_artifact_defaults_custom_metadata_to_none() -> None:
    ctx = _StubToolContext()
    await save_artifact(
        filename="scratch.txt",
        content="notes",
        mime_type="text/plain",
        tool_context=ctx,  # type: ignore[arg-type]
    )

    _, _, custom_metadata = ctx.saved[0]
    assert custom_metadata is None


@pytest.mark.asyncio
async def test_load_artifact_returns_text_kind_for_text_mime() -> None:
    part = genai_types.Part(
        inline_data=genai_types.Blob(data=b'{"ok": true}', mime_type="application/json")
    )
    ctx = _StubToolContext(load_result=part)
    result = await load_artifact(
        filename="report.json",
        tool_context=ctx,  # type: ignore[arg-type]
    )

    assert result == {
        "kind": "text",
        "data": '{"ok": true}',
        "mime_type": "application/json",
    }


@pytest.mark.asyncio
async def test_load_artifact_returns_bytes_kind_for_binary_mime() -> None:
    raw = b"\x00\x01\x02"
    part = genai_types.Part(
        inline_data=genai_types.Blob(data=raw, mime_type="application/octet-stream")
    )
    ctx = _StubToolContext(load_result=part)
    result = await load_artifact(
        filename="blob.bin",
        tool_context=ctx,  # type: ignore[arg-type]
    )

    assert result is not None
    assert result["kind"] == "bytes"
    assert result["mime_type"] == "application/octet-stream"
    assert base64.b64decode(result["data"]) == raw


@pytest.mark.asyncio
async def test_load_artifact_returns_none_when_missing() -> None:
    ctx = _StubToolContext(load_result=None)
    result = await load_artifact(
        filename="missing.txt",
        tool_context=ctx,  # type: ignore[arg-type]
    )
    assert result is None


@pytest.mark.asyncio
async def test_load_artifact_forwards_version_argument() -> None:
    ctx = _StubToolContext(load_result=None)
    await load_artifact(
        filename="report.json",
        version=3,
        tool_context=ctx,  # type: ignore[arg-type]
    )
    assert ctx.loaded == [("report.json", 3)]


@pytest.mark.asyncio
async def test_list_artifacts_forwards_to_tool_context() -> None:
    ctx = _StubToolContext(list_result=["a.txt", "b.txt"])
    result = await list_artifacts(tool_context=ctx)  # type: ignore[arg-type]
    assert result == ["a.txt", "b.txt"]
    assert ctx.listed == 1


def test_artifact_tools_constant_exposes_all_three() -> None:
    names = {t.name for t in ARTIFACT_TOOLS}
    assert names == {"save_artifact", "load_artifact", "list_artifacts"}


def test_artifact_tools_declarations_omit_tool_context_param() -> None:
    by_name = {t.name: t for t in ARTIFACT_TOOLS}
    save_decl = by_name["save_artifact"]._get_declaration()
    assert save_decl is not None
    properties: dict[str, Any] = save_decl.parameters.properties or {}  # type: ignore[union-attr]
    assert "tool_context" not in properties
    assert {"filename", "content", "custom_metadata"}.issubset(properties.keys())
