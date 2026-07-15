# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from typing import Any, cast

import pytest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from adk_code_mode import (
    TOOL_RESULT_DESCRIPTION_KEY,
    TOOL_RESULT_FILENAME_KEY,
    TOOL_RESULT_METADATA_KEY,
    TOOL_RESULT_NAME_KEY,
    ToolResultArtifactTool,
)
from adk_code_mode.executor import CodeModeCodeExecutor
from adk_code_mode.tools import normaliser
from tests._fake_runtime import FakeRuntime


class _ResultTool(BaseTool):
    def __init__(self, result: Any, *, name: str = "search") -> None:
        super().__init__(name=name, description="Search")
        self._result = result

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description="Search",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"q": types.Schema(type=types.Type.STRING)},
                required=["q"],
            ),
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        return self._result


class _FakeToolContext:
    def __init__(self, *, has_artifact_service: bool = True, call_id: str = "call-1") -> None:
        self.function_call_id = call_id
        self._invocation_context = type(
            "Inv", (), {"artifact_service": object() if has_artifact_service else None}
        )()
        self.saved: list[dict[str, Any]] = []

    async def save_artifact(self, **kwargs: Any) -> int:
        self.saved.append(kwargs)
        return len(self.saved) - 1


def _ctx(**kwargs: Any) -> ToolContext:
    return cast(ToolContext, _FakeToolContext(**kwargs))


def test_declaration_injects_optional_naming_params() -> None:
    decl = ToolResultArtifactTool(_ResultTool({"ok": True}))._get_declaration()
    assert decl is not None and decl.parameters is not None
    assert set(decl.parameters.properties or {}) == {"q", "artifact_name", "artifact_description"}
    assert decl.parameters.required == ["q"]  # naming params stay optional


def test_declaration_does_not_shadow_existing_param() -> None:
    class _Named(_ResultTool):
        def _get_declaration(self) -> types.FunctionDeclaration:
            return types.FunctionDeclaration(
                name=self.name,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"artifact_name": types.Schema(type=types.Type.STRING)},
                ),
            )

    tool = ToolResultArtifactTool(_Named({"ok": True}))
    assert tool._inject_name is False
    assert tool._inject_description is True


@pytest.mark.asyncio
async def test_small_result_saved_and_returned_unchanged() -> None:
    ctx = cast(Any, _FakeToolContext())
    result = await ToolResultArtifactTool(_ResultTool({"ok": True})).run_async(
        args={"q": "hi"}, tool_context=cast(ToolContext, ctx)
    )
    assert result == {"ok": True}  # transparent for small results
    saved = ctx.saved[0]
    assert saved["custom_metadata"][TOOL_RESULT_METADATA_KEY] == "true"
    assert saved["filename"] == saved["custom_metadata"][TOOL_RESULT_FILENAME_KEY]
    assert json.loads(saved["artifact"].inline_data.data) == {"ok": True}
    assert saved["filename"].startswith("search_call-1")  # auto-derived, no model naming
    assert TOOL_RESULT_NAME_KEY not in saved["custom_metadata"]


@pytest.mark.asyncio
async def test_optional_name_and_description_used() -> None:
    ctx = cast(Any, _FakeToolContext())
    await ToolResultArtifactTool(_ResultTool({"ok": True})).run_async(
        args={"q": "hi", "artifact_name": "A2A Profiles!", "artifact_description": "desc"},
        tool_context=cast(ToolContext, ctx),
    )
    saved = ctx.saved[0]
    assert saved["filename"] == "A2A_Profiles.json"
    assert saved["custom_metadata"][TOOL_RESULT_NAME_KEY] == "A2A_Profiles"
    assert saved["custom_metadata"][TOOL_RESULT_DESCRIPTION_KEY] == "desc"


@pytest.mark.asyncio
async def test_large_result_elided_with_reload_tip() -> None:
    big = {"blob": "x" * 100}
    ctx = cast(Any, _FakeToolContext())
    result = await ToolResultArtifactTool(_ResultTool(big), large_result_threshold=10).run_async(
        args={"q": "hi"}, tool_context=cast(ToolContext, ctx)
    )
    assert set(result) == {"tool_result_artifact", "note"}
    assert result["tool_result_artifact"] == ctx.saved[0]["filename"]
    assert result["note"] == (
        "The tool result (112 bytes) was saved as an artifact "
        "'search_call-1.json' instead of being returned inline. Reload it with "
        "load_artifact(filename='search_call-1.json')."
    )
    assert json.loads(ctx.saved[0]["artifact"].inline_data.data) == big  # full payload persisted


@pytest.mark.asyncio
async def test_no_artifact_service_is_passthrough() -> None:
    ctx = cast(Any, _FakeToolContext(has_artifact_service=False))
    result = await ToolResultArtifactTool(_ResultTool({"ok": True})).run_async(
        args={"q": "hi"}, tool_context=cast(ToolContext, ctx)
    )
    assert result == {"ok": True}
    assert ctx.saved == []


def _resolved(tool: BaseTool) -> normaliser.ResolvedTool:
    return normaliser.ResolvedTool(tool=tool, toolset=None)


def test_executor_wraps_non_artifact_tools_by_default() -> None:
    executor = CodeModeCodeExecutor(tools=[_ResultTool({}, name="search")], backend=FakeRuntime())
    wrapped = executor.apply_tool_result_wrapping([_resolved(_ResultTool({}, name="search"))])
    assert isinstance(wrapped[0].tool, ToolResultArtifactTool)


def test_executor_skips_artifact_tools_and_avoids_double_wrap() -> None:
    executor = CodeModeCodeExecutor(tools=[], backend=FakeRuntime())
    resolved = [_resolved(t) for t in executor.tools if isinstance(t, BaseTool)]
    assert resolved  # the built-in artifact tools were injected
    wrapped = executor.apply_tool_result_wrapping(resolved)
    assert all(not isinstance(rt.tool, ToolResultArtifactTool) for rt in wrapped)

    once = executor.apply_tool_result_wrapping([_resolved(_ResultTool({}, name="x"))])
    twice = executor.apply_tool_result_wrapping(once)
    assert sum(isinstance(rt.tool, ToolResultArtifactTool) for rt in twice) == 1


def test_executor_flag_off_is_noop() -> None:
    executor = CodeModeCodeExecutor(
        tools=[], backend=FakeRuntime(), save_tool_results_as_artifacts=False
    )
    resolved = [_resolved(_ResultTool({}, name="search"))]
    assert executor.apply_tool_result_wrapping(resolved) is resolved


def test_catalog_reflects_injected_naming_params() -> None:
    """The model-facing catalog must show the same params the stubs expose."""
    from adk_code_mode.tools import namespacing
    from adk_code_mode.tools.catalog import render_catalog

    executor = CodeModeCodeExecutor(tools=[_ResultTool({}, name="search")], backend=FakeRuntime())
    ns_tools = namespacing.build(
        executor.apply_tool_result_wrapping([_resolved(_ResultTool({}, name="search"))])
    )
    catalog = render_catalog(ns_tools)
    assert "artifact_name" in catalog
    assert "artifact_description" in catalog
