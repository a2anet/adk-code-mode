# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test driving an ``LlmAgent`` with a canned ``BaseLlm``.

No real LLM, no Docker. The fake LLM yields a fenced ``python`` code block
on the first call and a plain text response on the second so the agent
turn terminates after one round of code execution.
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

from adk_code_mode import (
    CODE_MODE_SYSTEM_INSTRUCTION,
    CodeModeExecutor,
    code_mode_before_model_callback,
)

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


def _build_runner(
    *,
    fake_llm: _FakeLlm,
    code_block: str,
    final_text: str,
) -> tuple[Runner, str, str]:
    executor = CodeModeExecutor(
        tools=[_EchoTool()],
        runtime=FakeRuntime(),
        max_output_chars=10_000,
    )
    object.__setattr__(
        fake_llm,
        "_responses",
        [_text_response(f"```python\n{code_block}\n```"), _text_response(final_text)],
    )

    agent = LlmAgent(
        name="fake_agent",
        model=fake_llm,
        instruction=f"You are a helpful assistant.\n\n{CODE_MODE_SYSTEM_INSTRUCTION}",
        code_executor=executor,
        before_model_callback=code_mode_before_model_callback(executor),
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


def _stdout_for_events(events: list[Any]) -> str:
    chunks: list[str] = []
    for event in events:
        content = getattr(event, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            result = getattr(part, "code_execution_result", None)
            if result is not None and getattr(result, "output", None):
                chunks.append(result.output)
    return "\n".join(chunks)


@pytest.mark.asyncio
async def test_agent_runs_code_block_through_executor() -> None:
    fake_llm = _FakeLlm(responses=[])
    runner, user_id, session_id = _build_runner(
        fake_llm=fake_llm,
        code_block=("from tools import echo\nprint(echo(message='hello from fake-llm'))\n"),
        final_text="All done.",
    )

    events = await _drive(runner, user_id, session_id)

    output = _stdout_for_events(events)
    assert "{'echoed': 'hello from fake-llm'}" in output

    assert fake_llm.captured_requests, "fake LLM was never called"
    first = fake_llm.captured_requests[0]
    system_instruction = first.config.system_instruction
    if isinstance(system_instruction, str):
        rendered = system_instruction
    elif system_instruction is None:
        rendered = ""
    else:
        rendered = "\n".join(p for p in system_instruction if isinstance(p, str))
    assert CODE_MODE_SYSTEM_INSTRUCTION.split("\n", 1)[0] in rendered
    assert "<tools>" in rendered and "</tools>" in rendered
    assert "def echo" in rendered


@pytest.mark.asyncio
async def test_agent_handles_code_with_no_tool_calls() -> None:
    fake_llm = _FakeLlm(responses=[])
    runner, user_id, session_id = _build_runner(
        fake_llm=fake_llm,
        code_block="print('hello')",
        final_text="Done.",
    )

    events = await _drive(runner, user_id, session_id)

    output = _stdout_for_events(events)
    assert "hello" in output
