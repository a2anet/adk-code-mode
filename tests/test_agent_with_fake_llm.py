# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test driving an ``LlmAgent`` with a canned ``BaseLlm``.

No real LLM, no Docker. The fake LLM yields a ``function_call`` part for
``execute_code`` on the first call (structured tool-calling, exactly what a
real model does) and a plain text response on the second so the agent turn
terminates after one round of code execution.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk.agents import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types

from adk_code_mode import ExecuteCodeTool

from ._fake_runtime import FakeRuntime


class _EchoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echo back args.")

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


class _FakeLlm(BaseLlm):
    """Yields canned ``LlmResponse`` items and records every request it sees."""

    model: str = "fake-llm"
    _responses: list[LlmResponse]
    _calls: int
    _captured_requests: list[LlmRequest]

    def __init__(self, *, responses: list[LlmResponse]) -> None:
        super().__init__(model="fake-llm")
        # Pydantic ``BaseLlm`` lets us stash extra attrs because of
        # ``arbitrary_types_allowed=True``; use ``object.__setattr__`` so
        # we don't have to register them as model fields.
        object.__setattr__(self, "_responses", list(responses))
        object.__setattr__(self, "_calls", 0)
        object.__setattr__(self, "_captured_requests", [])

    @property
    def captured_requests(self) -> list[LlmRequest]:
        return self._captured_requests

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self._captured_requests.append(llm_request)
        idx = min(self._calls, len(self._responses) - 1)
        object.__setattr__(self, "_calls", self._calls + 1)
        yield self._responses[idx]


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)])
    )


def _execute_code_response(code: str, *, call_id: str = "call-1") -> LlmResponse:
    return LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id=call_id, name="execute_code", args={"code": code}
                    )
                )
            ],
        )
    )


def _build_runner(
    *,
    fake_llm: _FakeLlm,
    code_block: str,
    final_text: str,
) -> tuple[Runner, str, str]:
    execute_code = ExecuteCodeTool(
        tools=[_EchoTool()],
        backend=FakeRuntime(),
        max_output_chars=10_000,
    )
    object.__setattr__(
        fake_llm,
        "_responses",
        [_execute_code_response(code_block), _text_response(final_text)],
    )

    agent = LlmAgent(
        name="fake_agent",
        model=fake_llm,
        instruction="You are a helpful assistant.",
        tools=[execute_code],
    )

    runner = Runner(
        app_name="fake-app",
        agent=agent,
        session_service=InMemorySessionService(),  # type: ignore[no-untyped-call]
        artifact_service=InMemoryArtifactService(),
    )
    return runner, "fake-user", "fake-session"


async def _drive(runner: Runner, user_id: str, session_id: str) -> list[Any]:
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    new_message = genai_types.Content(role="user", parts=[genai_types.Part(text="go")])
    events = []
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=new_message
    ):
        events.append(event)
    return events


def _tool_response_for_events(events: list[Any], *, tool_name: str = "execute_code") -> Any:
    for event in events:
        content = getattr(event, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            response = getattr(part, "function_response", None)
            if response is not None and response.name == tool_name:
                return response.response
    return None


@pytest.mark.asyncio
async def test_agent_runs_code_through_execute_code_tool() -> None:
    fake_llm = _FakeLlm(responses=[])
    runner, user_id, session_id = _build_runner(
        fake_llm=fake_llm,
        code_block=("from tools import echo\nprint(echo(message='hello from fake-llm'))\n"),
        final_text="All done.",
    )

    events = await _drive(runner, user_id, session_id)

    result = _tool_response_for_events(events)
    assert result is not None
    assert "{'echoed': 'hello from fake-llm'}" in result["stdout"]

    assert fake_llm.captured_requests, "fake LLM was never called"
    first = fake_llm.captured_requests[0]

    # The model called a real structured tool, not a text-fenced code block.
    declared_names = {
        decl.name
        for t in (first.config.tools or [])
        if isinstance(t, genai_types.Tool)
        for decl in (t.function_declarations or [])
    }
    assert "execute_code" in declared_names

    # append_function_stubs_to_system_instruction defaults on: the full
    # catalog should already be in the first request's system instruction.
    system_instruction = first.config.system_instruction
    assert isinstance(system_instruction, str)
    assert "<code-mode>" in system_instruction and "</code-mode>" in system_instruction
    assert "def echo" in system_instruction


@pytest.mark.asyncio
async def test_agent_handles_code_with_no_tool_calls() -> None:
    fake_llm = _FakeLlm(responses=[])
    runner, user_id, session_id = _build_runner(
        fake_llm=fake_llm,
        code_block="print('hello')",
        final_text="Done.",
    )

    events = await _drive(runner, user_id, session_id)

    result = _tool_response_for_events(events)
    assert result is not None
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_agent_catalog_omitted_when_flag_disabled() -> None:
    execute_code = ExecuteCodeTool(
        tools=[_EchoTool()],
        backend=FakeRuntime(),
        max_output_chars=10_000,
        append_function_stubs_to_system_instruction=False,
    )
    fake_llm = _FakeLlm(
        responses=[
            _execute_code_response("from tools import echo\nprint(echo(message='hi'))\n"),
            _text_response("Done."),
        ]
    )
    agent = LlmAgent(
        name="fake_agent",
        model=fake_llm,
        instruction="You are a helpful assistant.",
        tools=[execute_code],
    )
    runner = Runner(
        app_name="fake-app",
        agent=agent,
        session_service=InMemorySessionService(),  # type: ignore[no-untyped-call]
        artifact_service=InMemoryArtifactService(),
    )

    await _drive(runner, "fake-user", "fake-session-no-catalog")

    first = fake_llm.captured_requests[0]
    system_instruction = first.config.system_instruction
    assert system_instruction is None or "<code-mode>" not in system_instruction
