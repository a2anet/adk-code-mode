# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test against a real Docker container.

Marked ``@pytest.mark.docker`` — opt-in via ``uv run pytest -m docker``.
Skips cleanly if:

- Docker is not installed or the daemon is not reachable.

The test builds a throwaway ``python:3.13-slim``-based image pinned to the
freshly built local sandbox wheel, runs a single
``CodeModeCodeExecutor.execute_code`` round trip through ``UnsafeLocalDockerBackend``, and
asserts the sandbox's stdout echoed the tool result back.
"""

from __future__ import annotations
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types

from adk_code_mode import CodeModeCodeExecutor, UnsafeLocalDockerBackend

from ._docker_helpers import build_sandbox_image, build_sandbox_wheel, docker_ok

pytestmark = pytest.mark.docker

_IMAGE_TAG = "adk-code-mode-sandbox-test:local"


class _EchoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echo back a message.")

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        return {"echoed": args.get("message", "")}


def _fake_ctx(artifact_service: InMemoryArtifactService, session: Session) -> InvocationContext:
    agent = MagicMock()
    agent.canonical_before_tool_callbacks = []
    agent.canonical_after_tool_callbacks = []
    agent.canonical_on_tool_error_callbacks = []
    ctx = MagicMock(spec=InvocationContext)
    ctx.agent = agent
    ctx.plugin_manager = PluginManager()
    ctx.artifact_service = artifact_service
    ctx.session = session
    ctx.app_name = "test-app"
    ctx.user_id = "u1"
    ctx.credential_service = None
    ctx.invocation_id = "docker-inv-1"
    return ctx


@pytest.fixture(scope="module")
def docker_image() -> str:
    if not docker_ok():
        pytest.skip("docker daemon not reachable")
    wheel = build_sandbox_wheel()
    return build_sandbox_image(image_tag=_IMAGE_TAG, sandbox_wheel=wheel)


@pytest.mark.asyncio
async def test_docker_round_trip(docker_image: str) -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="docker-s1",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _fake_ctx(artifact_service, session)

    executor = CodeModeCodeExecutor(
        tools=[_EchoTool()],
        backend=UnsafeLocalDockerBackend(image=docker_image),
        max_output_chars=10_000,
        timeout_seconds=60,
    )
    code = "from tools import echo\nr = echo(message='hello from container')\nprint('ECHO:', r)\n"

    result = await executor._aexecute(
        ctx, CodeExecutionInput(code=code, execution_id="docker-run-1")
    )
    assert "ECHO: {'echoed': 'hello from container'}" in result.stdout


@pytest.mark.asyncio
async def test_docker_two_block_turn_preserves_state(docker_image: str) -> None:
    artifact_service = InMemoryArtifactService()
    session = Session(
        id="docker-s2",
        app_name="test-app",
        user_id="u1",
        state={},
        events=[],
        last_update_time=0.0,
    )
    ctx = _fake_ctx(artifact_service, session)  # one invocation_id => one turn

    executor = CodeModeCodeExecutor(
        tools=[],
        backend=UnsafeLocalDockerBackend(image=docker_image),
        max_output_chars=10_000,
        timeout_seconds=60,
    )

    # Two blocks of the same turn must run in the same container: the variable
    # and the /workspace file set in block 1 have to be visible in block 2.
    await executor._aexecute(
        ctx,
        CodeExecutionInput(
            code="turn_value = 99\nopen('turn.txt', 'w').write('kept')\n",
            execution_id="docker-run-2a",
        ),
    )
    result = await executor._aexecute(
        ctx,
        CodeExecutionInput(
            code="print('value', turn_value)\nprint('file', open('turn.txt').read())\n",
            execution_id="docker-run-2b",
        ),
    )

    assert "value 99" in result.stdout
    assert "file kept" in result.stdout

    executor.release_invocation(ctx.invocation_id)
