# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Test-only real ADK ``RestApiTool`` whose API answers with a canned response."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.openapi_tool.openapi_spec_parser import rest_api_tool
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

RULES_SPEC: dict[str, Any] = {
    "openapi": "3.0.0",
    "info": {"title": "Rules", "version": "1"},
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/rules": {
            "post": {
                "operationId": "createRule",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
                        }
                    }
                },
                "responses": {"201": {"description": "Created"}},
            }
        }
    },
}

VALIDATION_ERROR = {"errors": {"scope.zones": ["The zones field is required."]}}


async def rest_tool(monkeypatch: pytest.MonkeyPatch, status: int, body: Any) -> BaseTool:
    """Return ADK's ``create_rule`` tool with its API answering ``status`` and ``body``."""

    async def fake_request(**request_params: Any) -> httpx.Response:
        request = httpx.Request(request_params["method"], request_params["url"])
        return httpx.Response(status, json=body, request=request)

    monkeypatch.setattr(rest_api_tool, "_request", fake_request)
    [tool] = await OpenAPIToolset(spec_dict=RULES_SPEC).get_tools()
    return tool
