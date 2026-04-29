# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Normalise heterogeneous tool inputs into a flat ``list[BaseTool]``.

Accepts any mix of:

- plain Python callables (wrapped as ``FunctionTool``)
- ``BaseTool`` instances (kept as-is)
- ``BaseToolset`` instances (expanded via ``get_tools_with_prefix``)

Toolsets are async. We collect their tools inside the caller's event loop.
Each resolved tool remembers the toolset it came from (if any) so namespacing
can group them later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool

ToolInput = BaseTool | BaseToolset | Callable[..., Any]


@dataclass(frozen=True)
class ResolvedTool:
    """A single tool with provenance for namespacing."""

    tool: BaseTool
    toolset: BaseToolset | None


async def resolve(
    inputs: Sequence[ToolInput],
    readonly_context: ReadonlyContext,
) -> list[ResolvedTool]:
    """Flatten the given inputs into a list of ``ResolvedTool``.

    Args:
        inputs: The mixed sequence passed to ``CodeModeCodeExecutor(tools=...)``.
        readonly_context: Forwarded to ``BaseToolset.get_tools_with_prefix``.

    Returns:
        A flat list preserving input order. Toolset tools appear in the order
        the toolset returned them.
    """
    resolved: list[ResolvedTool] = []
    for item in inputs:
        if isinstance(item, BaseTool):
            resolved.append(ResolvedTool(tool=item, toolset=None))
        elif isinstance(item, BaseToolset):
            tools = await item.get_tools_with_prefix(readonly_context)
            for tool in tools:
                resolved.append(ResolvedTool(tool=tool, toolset=item))
        elif callable(item):
            resolved.append(ResolvedTool(tool=FunctionTool(item), toolset=None))
        else:
            raise TypeError(
                f"Unsupported tool input {item!r}: expected BaseTool, BaseToolset, or callable"
            )
    return resolved


__all__ = ["ResolvedTool", "ToolInput", "resolve"]
