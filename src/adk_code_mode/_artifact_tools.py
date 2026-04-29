# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Built-in ``save_artifact`` / ``load_artifact`` / ``list_artifacts`` tools.

These are regular ``FunctionTool`` instances injected at the front of
``CodeModeExecutor.tools`` (set ``include_artifact_tools=False`` to opt out).
They appear in the rendered catalog as top-level tools; the model writes
``from tools import save_artifact, load_artifact, list_artifacts``.

The wire format is JSON. Binary content is base64 over the wire — the model
encodes it before calling ``save_artifact`` and decodes it after
``load_artifact``. Loaded artifacts return a flat ``{"kind", "data",
"mime_type"}`` dict so the model dispatches on ``kind`` without importing
host-side types.
"""

from __future__ import annotations

import base64
from typing import Any

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_TYPES = {"application/json", "application/xml", "application/javascript"}


def _is_text_mime(mime_type: str | None) -> bool:
    if not mime_type:
        return False
    if mime_type.startswith(_TEXT_MIME_PREFIXES):
        return True
    if mime_type in _TEXT_MIME_TYPES:
        return True
    return mime_type.endswith("+json") or mime_type.endswith("+xml")


async def save_artifact(
    *,
    filename: str,
    content: str,
    mime_type: str | None = None,
    tool_context: ToolContext,
) -> int:
    """Save an artifact to the session. Returns the new version number.

    For text or JSON, pass ``content`` as a string and set ``mime_type``:

        save_artifact(
            filename="report.json",
            content=json.dumps({"status": "ok"}),
            mime_type="application/json",
        )

    For binary, base64-encode the bytes and set the binary mime type:

        save_artifact(
            filename="image.png",
            content=base64.b64encode(image_bytes).decode("ascii"),
            mime_type="image/png",
        )
    """
    if _is_text_mime(mime_type):
        part = genai_types.Part(
            inline_data=genai_types.Blob(
                data=content.encode("utf-8"),
                mime_type=mime_type or "text/plain",
            )
        )
    else:
        part = genai_types.Part(
            inline_data=genai_types.Blob(
                data=base64.b64decode(content),
                mime_type=mime_type or "application/octet-stream",
            )
        )
    return await tool_context.save_artifact(filename, part)


async def load_artifact(
    *,
    filename: str,
    version: int | None = None,
    tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Load an artifact by filename. Returns ``None`` if not found.

    Returned dict shape:

        {
            "kind": "text" | "bytes",
            "data": <str>,                # text: the string; bytes: base64-encoded
            "mime_type": <str | None>,
        }

    Example:

        result = load_artifact(filename="report.json")
        if result is None:
            return
        if result["kind"] == "text" and result["mime_type"] == "application/json":
            payload = json.loads(result["data"])
        elif result["kind"] == "bytes":
            blob = base64.b64decode(result["data"])
    """
    part = await tool_context.load_artifact(filename, version=version)
    if part is None:
        return None
    inline = part.inline_data
    if inline is not None:
        mime_type = inline.mime_type
        data = inline.data or b""
        if isinstance(data, str):
            data = data.encode("utf-8")
        if _is_text_mime(mime_type):
            return {
                "kind": "text",
                "data": data.decode("utf-8"),
                "mime_type": mime_type,
            }
        return {
            "kind": "bytes",
            "data": base64.b64encode(data).decode("ascii"),
            "mime_type": mime_type,
        }
    if part.text is not None:
        return {"kind": "text", "data": part.text, "mime_type": None}
    return {"kind": "text", "data": "", "mime_type": None}


async def list_artifacts(*, tool_context: ToolContext) -> list[str]:
    """Return the filenames of artifacts visible to this session.

    Example:

        for filename in list_artifacts():
            print(filename)
    """
    return await tool_context.list_artifacts()


ARTIFACT_TOOLS: tuple[FunctionTool, ...] = (
    FunctionTool(save_artifact),
    FunctionTool(load_artifact),
    FunctionTool(list_artifacts),
)


__all__ = ["ARTIFACT_TOOLS", "list_artifacts", "load_artifact", "save_artifact"]
