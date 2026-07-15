# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ExecuteCodeTool`` itself: declaration and ``process_llm_request``.

Catalog-injection behavior here supersedes the old
``adk_code_mode.callback.code_mode_before_model_callback`` tests — that
standalone callback no longer exists; its behavior now lives in
``ExecuteCodeTool.process_llm_request``, gated by
``append_function_stubs_to_system_instruction``.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from adk_code_mode.tool import ExecuteCodeTool
from tests._fake_runtime import FakeRuntime


class _SchemaTool(BaseTool):
    def __init__(self, name: str, description: str = "Tool.") -> None:
        super().__init__(name=name, description=description)

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={"type": "object"},
        )


def _make_tool(tools: list[BaseTool], **kwargs: Any) -> ExecuteCodeTool:
    return ExecuteCodeTool(tools=tools, backend=FakeRuntime(), **kwargs)


def _tool_context(invocation_id: str = "inv-1") -> ToolContext:
    session = Session(id="s", app_name="a", user_id="u", state={}, events=[], last_update_time=0.0)
    ctx = MagicMock(spec=InvocationContext)
    ctx.invocation_id = invocation_id
    ctx.session = session
    ctx.artifact_service = None
    return ToolContext(invocation_context=ctx, function_call_id="call-1")


def _llm_request(system_instruction: str | None = None) -> LlmRequest:
    return LlmRequest(
        config=genai_types.GenerateContentConfig(system_instruction=system_instruction)
    )


def test_declaration_exposes_a_single_code_parameter() -> None:
    tool = _make_tool([])
    decl = tool._get_declaration()
    assert decl is not None and decl.parameters_json_schema is not None
    assert decl.name == "execute_code"
    assert set(decl.parameters_json_schema["properties"]) == {"code"}
    assert decl.parameters_json_schema["required"] == ["code"]


@pytest.mark.asyncio
async def test_process_llm_request_declares_the_tool() -> None:
    tool = _make_tool([])
    request = _llm_request()

    await tool.process_llm_request(tool_context=_tool_context(), llm_request=request)

    declared_names = {
        decl.name
        for t in (request.config.tools or [])
        if isinstance(t, genai_types.Tool)
        for decl in (t.function_declarations or [])
    }
    assert "execute_code" in declared_names


@pytest.mark.asyncio
async def test_catalog_appended_when_system_instruction_is_none() -> None:
    tool = _make_tool([_SchemaTool("ping", "Ping.")])
    request = _llm_request(system_instruction=None)

    await tool.process_llm_request(tool_context=_tool_context(), llm_request=request)

    instruction = request.config.system_instruction
    assert isinstance(instruction, str)
    assert "Reference catalog of the functions available" in instruction
    assert "<code-mode>" in instruction and instruction.rstrip().endswith("</code-mode>")
    assert "def ping" in instruction


@pytest.mark.asyncio
async def test_catalog_appended_to_existing_string_system_instruction() -> None:
    tool = _make_tool([_SchemaTool("ping")])
    request = _llm_request(system_instruction="You are an assistant.")

    await tool.process_llm_request(tool_context=_tool_context(), llm_request=request)

    instruction = request.config.system_instruction
    assert isinstance(instruction, str)
    assert instruction.startswith("You are an assistant.")
    assert "<code-mode>" in instruction
    assert instruction.rstrip().endswith("</code-mode>")


@pytest.mark.asyncio
async def test_nothing_appended_when_catalog_exceeds_max_catalog_chars() -> None:
    tool = _make_tool(
        [_SchemaTool(f"tool_{i}", f"Description {i}.") for i in range(20)],
        max_catalog_chars=200,
    )
    request = _llm_request(system_instruction=None)

    await tool.process_llm_request(tool_context=_tool_context(), llm_request=request)

    # No fallback message either — the model relies on progressive discovery
    # (list /tools/, read a stub's docstring) with nothing added upfront.
    assert request.config.system_instruction is None


@pytest.mark.asyncio
async def test_catalog_omitted_when_flag_disabled() -> None:
    tool = _make_tool([_SchemaTool("ping")], append_function_stubs_to_system_instruction=False)
    request = _llm_request(system_instruction=None)

    await tool.process_llm_request(tool_context=_tool_context(), llm_request=request)

    assert request.config.system_instruction is None


@pytest.mark.asyncio
async def test_get_or_resolve_tools_is_cached_per_invocation() -> None:
    tool = _make_tool([_SchemaTool("ping")])
    ctx = _tool_context("inv-cache")

    first = await tool._get_or_resolve_tools(ctx)
    second = await tool._get_or_resolve_tools(ctx)

    assert first is second


@pytest.mark.asyncio
async def test_resolved_tool_cache_releases_with_invocation_context() -> None:
    tool = _make_tool([_SchemaTool("ping")])
    ctx = _tool_context("inv-weak")
    context_ref = weakref.ref(ctx._invocation_context)

    await tool._get_or_resolve_tools(ctx)
    assert len(tool._resolved_tools) == 1

    del ctx
    for _ in range(3):
        gc.collect()

    assert context_ref() is None
    assert len(tool._resolved_tools) == 0
