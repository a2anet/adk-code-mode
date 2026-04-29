# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.sessions.session import Session

from adk_code_mode import output


class _Ctx:
    def __init__(self, service: InMemoryArtifactService) -> None:
        self.artifact_service = service
        self.app_name = "a"
        self.user_id = "u"
        self.session = Session(
            id="s",
            app_name="a",
            user_id="u",
            state={},
            events=[],
            last_update_time=0.0,
        )


async def test_under_limit_passes_through() -> None:
    ctx = _Ctx(InMemoryArtifactService())
    text = "hello world"
    r = await output.truncate(
        text,
        limit=1000,
        stream_name="stdout",
        execution_id="e1",
        invocation_context=ctx,  # type: ignore[arg-type]
    )
    assert r.text == text
    assert r.artifact_filename is None


async def test_over_limit_truncates_and_spills() -> None:
    ctx = _Ctx(InMemoryArtifactService())
    text = "x" * 5_000
    r = await output.truncate(
        text,
        limit=1_000,
        stream_name="stdout",
        execution_id="e1",
        invocation_context=ctx,  # type: ignore[arg-type]
    )
    assert len(r.text) < len(text) + 300
    assert "Output exceeded" in r.text
    assert "[truncated]" in r.text
    assert r.artifact_filename == "code_mode/stdout/e1.txt"
    keys = await ctx.artifact_service.list_artifact_keys(app_name="a", user_id="u", session_id="s")
    assert "code_mode/stdout/e1.txt" in keys
