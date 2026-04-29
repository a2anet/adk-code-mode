# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


import pytest
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset

from adk_code_mode.tools import namespacing
from adk_code_mode.tools.normaliser import ResolvedTool


class _Tool(BaseTool):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, description=f"desc-{name}")


class _NamedToolset(BaseToolset):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[BaseTool]:
        return []

    async def close(self) -> None:
        return None


def test_bare_tools_live_at_top_level() -> None:
    tools = [
        ResolvedTool(tool=_Tool("send_message"), toolset=None),
        ResolvedTool(tool=_Tool("list-channels"), toolset=None),
    ]
    ns = namespacing.build(tools)
    assert ns[0].namespace is None
    assert ns[0].attribute == "send_message"
    assert ns[1].namespace is None
    assert ns[1].attribute == "list_channels"


def test_toolset_tools_share_namespace() -> None:
    slack = _NamedToolset("Slack")
    tools = [
        ResolvedTool(tool=_Tool("send_message"), toolset=slack),
        ResolvedTool(tool=_Tool("list_channels"), toolset=slack),
    ]
    ns = namespacing.build(tools)
    assert ns[0].namespace == "slack"
    assert ns[0].dotted_path == "slack.send_message"
    assert ns[1].dotted_path == "slack.list_channels"


def test_duplicate_raw_tool_names_raise() -> None:
    tools = [
        ResolvedTool(tool=_Tool("ping"), toolset=None),
        ResolvedTool(tool=_Tool("ping"), toolset=_NamedToolset("Slack")),
    ]
    with pytest.raises(namespacing.DuplicateToolNameError):
        namespacing.build(tools)


def test_python_identifier_collisions_raise() -> None:
    tools = [
        ResolvedTool(tool=_Tool("ping"), toolset=None),
        ResolvedTool(tool=_Tool("PING"), toolset=None),
    ]
    with pytest.raises(namespacing.PythonNameCollisionError):
        namespacing.build(tools)


def test_namespace_collisions_raise() -> None:
    tools = [
        ResolvedTool(tool=_Tool("send_message"), toolset=_NamedToolset("Slack")),
        ResolvedTool(tool=_Tool("list_channels"), toolset=_NamedToolset("slack")),
    ]
    with pytest.raises(namespacing.NamespaceCollisionError):
        namespacing.build(tools)


def test_namespace_and_top_level_name_conflict_raises() -> None:
    tools = [
        ResolvedTool(tool=_Tool("slack"), toolset=None),
        ResolvedTool(tool=_Tool("send_message"), toolset=_NamedToolset("Slack")),
    ]
    with pytest.raises(namespacing.PythonNameCollisionError):
        namespacing.build(tools)


def test_registry_bidirectional_lookup() -> None:
    slack = _NamedToolset("Slack")
    ns = namespacing.build([ResolvedTool(tool=_Tool("send_message"), toolset=slack)])
    reg = namespacing.Registry(ns)
    assert reg.resolve_call("slack.send_message") is ns[0]
    assert reg.resolve_call("send_message") is ns[0]
    with pytest.raises(KeyError):
        reg.resolve_call("nope")
