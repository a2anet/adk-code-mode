# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Errors reported as ordinary return values by ADK tools."""

from __future__ import annotations

import json
import re
from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool

# ADK's `RestApiTool` returns an HTTP error as `{"error": "... Status Code: 400, <body>"}`.
_REST_ERROR_STATUS = re.compile(r"Status Code: (\d{3}), (.*)\Z", re.DOTALL)


class HTTPStatusError(Exception):
    """Raised when a REST tool's API answers with an HTTP error status."""

    def __init__(self, tool_name: str, status_code: int, body: Any) -> None:
        text = body if isinstance(body, str) else json.dumps(body)
        super().__init__(f"Tool `{tool_name}` returned HTTP {status_code}: {text}")
        self.status_code = status_code
        self.body = body


class McpToolError(Exception):
    """Raised when an MCP tool reports that its call failed."""


def raise_for_failed_result(tool: BaseTool, result: Any) -> None:
    """Raise when ADK returns a tool failure as an ordinary value."""
    if not isinstance(result, dict):
        return
    if isinstance(tool, RestApiTool) and set(result) == {"error"}:
        match = _REST_ERROR_STATUS.search(str(result["error"]))
        if match is None:
            raise RuntimeError(f"Tool `{tool.name}` failed: {result['error']}")
        details = match.group(2)
        try:
            body: Any = json.loads(details)
        except ValueError:
            body = details
        raise HTTPStatusError(tool.name, int(match.group(1)), body)
    # An MCP `CallToolResult` with `isError` set, as `McpTool` returns it.
    if result.get("isError") is True:
        texts = [
            item["text"]
            for item in result.get("content") or []
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        raise McpToolError(f"Tool `{tool.name}` failed: {' '.join(texts)}")
