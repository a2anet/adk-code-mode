# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types

from adk_code_mode import code_mode_before_model_callback
from adk_code_mode.executor import CodeModeCodeExecutor
from adk_code_mode.tools import normaliser
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


@dataclass
class _FakeConfig:
    system_instruction: Any = None


@dataclass
class _FakeRequest:
    config: _FakeConfig


class _FakeContext:
    def __init__(self, invocation_id: str = "inv-1") -> None:
        self.invocation_id = invocation_id


def _make_executor(tools: list[BaseTool], **kwargs: Any) -> CodeModeCodeExecutor:
    return CodeModeCodeExecutor(tools=tools, backend=FakeRuntime(), **kwargs)


@pytest.mark.asyncio
async def test_callback_appends_block_when_system_instruction_is_none() -> None:
    executor = _make_executor([_SchemaTool("ping", "Ping.")])
    callback = code_mode_before_model_callback(executor)
    request = _FakeRequest(config=_FakeConfig(system_instruction=None))

    await callback(_FakeContext(), request)

    instruction = request.config.system_instruction
    assert isinstance(instruction, str)
    assert instruction.startswith("<tools>")
    assert instruction.rstrip().endswith("</tools>")
    assert "def ping" in instruction


@pytest.mark.asyncio
async def test_callback_appends_block_to_string_system_instruction() -> None:
    executor = _make_executor([_SchemaTool("ping")])
    callback = code_mode_before_model_callback(executor)
    request = _FakeRequest(config=_FakeConfig(system_instruction="You are an assistant."))

    await callback(_FakeContext(), request)

    instruction = request.config.system_instruction
    assert isinstance(instruction, str)
    assert instruction.startswith("You are an assistant.")
    assert "<tools>" in instruction
    assert instruction.rstrip().endswith("</tools>")


@pytest.mark.asyncio
async def test_callback_appends_block_to_list_system_instruction() -> None:
    executor = _make_executor([_SchemaTool("ping")])
    callback = code_mode_before_model_callback(executor)
    request = _FakeRequest(config=_FakeConfig(system_instruction=["First part.", "Second part."]))

    await callback(_FakeContext(), request)

    instruction = request.config.system_instruction
    assert isinstance(instruction, list)
    assert instruction[:2] == ["First part.", "Second part."]
    assert instruction[-1].startswith("<tools>")
    assert instruction[-1].rstrip().endswith("</tools>")


@pytest.mark.asyncio
async def test_callback_uses_overflow_when_catalog_too_large() -> None:
    executor = _make_executor(
        [_SchemaTool(f"tool_{i}", f"Description {i}.") for i in range(20)],
        max_catalog_chars=200,
    )
    callback = code_mode_before_model_callback(executor)
    request = _FakeRequest(config=_FakeConfig(system_instruction=None))

    await callback(_FakeContext(), request)

    instruction = request.config.system_instruction
    assert "A `tools` package is available" in instruction
    assert "pathlib.Path('/tools').iterdir()" in instruction
    assert ".py` file (a top-level tool" in instruction
    assert "subdirectory (a namespace" in instruction
    assert "def tool_0" not in instruction


@pytest.mark.asyncio
async def test_callback_caches_resolved_tools_for_executor() -> None:
    executor = _make_executor([_SchemaTool("ping")], include_artifact_tools=False)
    callback = code_mode_before_model_callback(executor)
    request = _FakeRequest(config=_FakeConfig(system_instruction=None))

    await callback(_FakeContext("inv-42"), request)

    cached = executor._consume_resolved_tools("inv-42")
    assert cached is not None
    assert len(cached) == 1
    assert cached[0].tool.name == "ping"
    # Pop semantics — second consume returns nothing.
    assert executor._consume_resolved_tools("inv-42") is None


def test_resolution_cache_evicts_oldest_when_full() -> None:
    executor = _make_executor([])
    limit = executor._RESOLUTION_CACHE_LIMIT
    fake = [normaliser.ResolvedTool(tool=_SchemaTool("x"), toolset=None)]
    for i in range(limit + 5):
        executor._record_resolved_tools(f"inv-{i}", fake)
    # Oldest five entries got pushed out; the most recent ``limit`` remain.
    assert executor._consume_resolved_tools("inv-0") is None
    assert executor._consume_resolved_tools("inv-4") is None
    assert executor._consume_resolved_tools(f"inv-{limit + 4}") is not None


def test_consume_returns_none_when_callback_was_not_wired() -> None:
    """Graceful fallback path used by ``_prepare_tool_surface``: if the
    callback never recorded tools for this invocation, consume returns
    ``None`` and the executor resolves fresh."""
    executor = _make_executor([])
    assert executor._consume_resolved_tools("never-seen") is None
