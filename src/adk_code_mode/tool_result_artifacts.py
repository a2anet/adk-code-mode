# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Optionally save each tool's result as a session Artifact.

``ToolResultArtifactTool`` wraps a ``BaseTool`` so that every call's return value
is written to a session artifact (via ``ToolContext.save_artifact``) and tagged
with ``code_mode.tool_result = "true"``. Enable it with
``CodeModeCodeExecutor(save_tool_results_as_artifacts=True)`` (the default);
the executor applies the wrapper to the resolved tool surface.

Naming is transparent by default: the artifact filename is derived from the tool
name and call id. The model may optionally pass ``artifact_name`` /
``artifact_description`` — injected as optional parameters on every wrapped tool
— to name and describe the saved artifact. Those, plus the ``code_mode.*``
markers, land in the artifact's ``custom_metadata`` so a host (e.g. an A2A layer)
can decide whether and how to forward the result to the user.

Small results are returned to the model unchanged. Results whose serialised form
exceeds ``large_result_threshold`` are replaced with a short notice pointing at
the artifact, keeping large payloads out of the model's context while remaining
reloadable with ``load_artifact``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Transport-neutral metadata stamped on saved tool-result artifacts. Downstream
# consumers read these to decide whether/how to surface the artifact to the user.
TOOL_RESULT_METADATA_KEY = "code_mode.tool_result"
TOOL_RESULT_NAME_KEY = "code_mode.artifact_name"
TOOL_RESULT_DESCRIPTION_KEY = "code_mode.artifact_description"
TOOL_RESULT_FILENAME_KEY = "code_mode.artifact_filename"

DEFAULT_LARGE_RESULT_THRESHOLD = 50_000

_ARTIFACT_NAME_PARAM = "artifact_name"
_ARTIFACT_DESCRIPTION_PARAM = "artifact_description"


class ToolResultArtifactTool(BaseTool):
    """Wrap a tool so its result is saved as a session artifact.

    Preserves the wrapped tool's declaration (the model still sees the original
    parameters) and injects two optional parameters, ``artifact_name`` and
    ``artifact_description``, for naming the saved artifact.
    """

    def __init__(
        self,
        wrapped: BaseTool,
        *,
        large_result_threshold: int = DEFAULT_LARGE_RESULT_THRESHOLD,
    ) -> None:
        super().__init__(
            name=wrapped.name,
            description=wrapped.description,
            is_long_running=wrapped.is_long_running,
        )
        self._wrapped = wrapped
        self._large_result_threshold = large_result_threshold
        # Don't shadow a real tool parameter that happens to share the name.
        existing = _declared_param_names(wrapped)
        self._inject_name = _ARTIFACT_NAME_PARAM not in existing
        self._inject_description = _ARTIFACT_DESCRIPTION_PARAM not in existing

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        base = self._wrapped._get_declaration()
        if base is None:
            return None
        decl = base.model_copy(deep=True)
        if decl.parameters is None:
            decl.parameters = types.Schema(type=types.Type.OBJECT, properties={}, required=[])
        properties = dict(decl.parameters.properties or {})
        if self._inject_name:
            properties[_ARTIFACT_NAME_PARAM] = types.Schema(
                type=types.Type.STRING,
                description=(
                    "Optional snake_case name for the saved tool-result artifact "
                    "(no extension). Defaults to a name derived from the tool."
                ),
            )
        if self._inject_description:
            properties[_ARTIFACT_DESCRIPTION_PARAM] = types.Schema(
                type=types.Type.STRING,
                description="Optional human-readable description for the saved artifact.",
            )
        decl.parameters.properties = properties
        return decl

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        call_args = dict(args)
        raw_name = call_args.pop(_ARTIFACT_NAME_PARAM, None) if self._inject_name else None
        raw_description = (
            call_args.pop(_ARTIFACT_DESCRIPTION_PARAM, None) if self._inject_description else None
        )

        result = await self._wrapped.run_async(args=call_args, tool_context=tool_context)

        # No artifact service configured → behave as a transparent pass-through
        # rather than failing every tool call (mirrors ADK's own guard).
        if tool_context._invocation_context.artifact_service is None:
            return result

        artifact_name = _normalise_artifact_name(raw_name) if raw_name else None
        stem = artifact_name or _default_stem(self.name, tool_context)
        filename = f"{stem}.json"

        data, mime_type = _serialise_result(result)
        metadata: dict[str, Any] = {
            TOOL_RESULT_METADATA_KEY: "true",
            TOOL_RESULT_FILENAME_KEY: filename,
        }
        if artifact_name:
            metadata[TOOL_RESULT_NAME_KEY] = artifact_name
        if raw_description:
            metadata[TOOL_RESULT_DESCRIPTION_KEY] = str(raw_description)

        await tool_context.save_artifact(
            filename=filename,
            artifact=types.Part(
                inline_data=types.Blob(data=data, mime_type=mime_type, display_name=filename)
            ),
            custom_metadata=metadata,
        )

        if len(data) > self._large_result_threshold:
            return {
                "tool_result_artifact": filename,
                "note": (
                    f"The tool result ({len(data):,} bytes) was saved as artifact "
                    f"{filename!r} instead of returned inline. Reload it with "
                    f"load_artifact({filename!r})."
                ),
            }
        return result


def wrap_tool_result_as_artifact(
    tool: BaseTool, *, large_result_threshold: int = DEFAULT_LARGE_RESULT_THRESHOLD
) -> ToolResultArtifactTool:
    """Wrap ``tool`` so its results are saved as session artifacts."""
    return ToolResultArtifactTool(tool, large_result_threshold=large_result_threshold)


def _declared_param_names(tool: BaseTool) -> set[str]:
    declaration = tool._get_declaration()
    params = getattr(declaration, "parameters", None) if declaration else None
    properties = getattr(params, "properties", None) if params else None
    return set(properties or {})


def _default_stem(tool_name: str, tool_context: ToolContext) -> str:
    call_id = getattr(tool_context, "function_call_id", None) or "result"
    return _normalise_artifact_name(f"{tool_name}_{call_id}")


def _normalise_artifact_name(value: Any) -> str:
    name = str(value).removesuffix(".json")
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._-")
    return name or "tool_result"


def _serialise_result(result: Any) -> tuple[bytes, str]:
    if isinstance(result, (bytes, bytearray)):
        return bytes(result), "application/octet-stream"
    if isinstance(result, str):
        return result.encode("utf-8"), "text/plain; charset=utf-8"
    return json.dumps(result, default=str, ensure_ascii=False).encode("utf-8"), "application/json"


__all__ = [
    "DEFAULT_LARGE_RESULT_THRESHOLD",
    "TOOL_RESULT_DESCRIPTION_KEY",
    "TOOL_RESULT_FILENAME_KEY",
    "TOOL_RESULT_METADATA_KEY",
    "TOOL_RESULT_NAME_KEY",
    "ToolResultArtifactTool",
    "wrap_tool_result_as_artifact",
]
