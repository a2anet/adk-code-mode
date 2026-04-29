# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool

from adk_code_mode.tools import normaliser


class _Plain(BaseTool):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, description=f"desc-{name}")


class _FakeToolset(BaseToolset):
    def __init__(self, tools: list[BaseTool]) -> None:
        super().__init__()
        self._tools = tools

    async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[BaseTool]:
        return list(self._tools)

    async def close(self) -> None:
        return None


def _fn() -> str:
    """A sample docstring."""
    return "ok"


def _make_readonly_context(invocation_id: str = "test-inv") -> ReadonlyContext:
    invocation_context = MagicMock(spec=InvocationContext)
    invocation_context.invocation_id = invocation_id
    return ReadonlyContext(invocation_context)


async def test_resolve_mixed_inputs() -> None:
    tool_a = _Plain("tool_a")
    toolset = _FakeToolset([_Plain("slack_send"), _Plain("slack_list")])
    resolved = await normaliser.resolve([tool_a, toolset, _fn], _make_readonly_context())
    assert [r.tool.name for r in resolved] == ["tool_a", "slack_send", "slack_list", "_fn"]
    assert [r.toolset for r in resolved] == [None, toolset, toolset, None]
    assert isinstance(resolved[3].tool, FunctionTool)


async def test_resolve_rejects_garbage() -> None:
    with pytest.raises(TypeError):
        await normaliser.resolve([42], _make_readonly_context())  # type: ignore[list-item]
