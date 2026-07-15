# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types

from adk_code_mode.tools import namespacing
from adk_code_mode.tools.catalog import render_catalog
from adk_code_mode.tools.normaliser import ResolvedTool


class _SchemaTool(BaseTool):
    def __init__(
        self,
        name: str,
        *,
        description: str,
        schema: dict[str, object] | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self._schema = schema or {}

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._schema,
        )


class _FakeToolset:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tool_name_prefix = ""


def _build(nts: list[tuple[BaseTool, object | None]]) -> list[namespacing.NamespacedTool]:
    resolved = [ResolvedTool(tool=t, toolset=ts) for t, ts in nts]  # type: ignore[arg-type]
    return namespacing.build(resolved)


def test_render_catalog_includes_namespace_section_with_import_line() -> None:
    slack = _FakeToolset("slack")
    send = _SchemaTool(
        "send_message",
        description="Send a message to a Slack channel.",
        schema={
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Channel ID like C123."},
                "text": {"type": "string", "description": "Message text."},
                "thread_ts": {"type": "string", "nullable": True, "description": "Thread ts."},
            },
            "required": ["channel", "text"],
        },
    )
    catalog = render_catalog(_build([(send, slack)]))
    assert "# tools.slack" in catalog
    assert "from tools.slack import send_message" in catalog
    assert (
        "def send_message(*, channel: str, text: str, thread_ts: str | None = ...) -> Any:"
        in catalog
    )
    assert '"""Send a message to a Slack channel.' in catalog
    assert "_MISSING" not in catalog


def test_render_catalog_includes_top_level_tools_section() -> None:
    echo = _SchemaTool(
        "echo",
        description="Echo a message.",
        schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )
    catalog = render_catalog(_build([(echo, None)]))
    assert "# tools" in catalog
    assert "from tools import echo" in catalog
    assert "def echo(*, message: str) -> Any:" in catalog


def test_render_catalog_groups_tools_by_namespace_and_sorts() -> None:
    slack = _FakeToolset("slack")
    gmail = _FakeToolset("gmail")
    a = _SchemaTool("send_email", description="Send an email.", schema={"type": "object"})
    b = _SchemaTool("send_message", description="Send a message.", schema={"type": "object"})
    catalog = render_catalog(_build([(b, slack), (a, gmail)]))
    gmail_idx = catalog.index("# tools.gmail")
    slack_idx = catalog.index("# tools.slack")
    assert gmail_idx < slack_idx, "namespaces sorted alphabetically"
