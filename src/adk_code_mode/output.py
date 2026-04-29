# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Bounded output surfacing with artifact spillover.

The model only ever sees a truncated view of the execution output; the full
stream is persisted as a session-scoped artifact so the user can still recover
it. When truncation kicks in, the message the model reads cites the artifact
name so it can ``load_artifact()`` the overflow if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from google.adk.agents.invocation_context import InvocationContext
from google.genai import types as genai_types

StreamName = Literal["stdout", "stderr"]

STDOUT_PREFIX = "code_mode/stdout"
STDERR_PREFIX = "code_mode/stderr"


def _overflow_filename(stream_name: StreamName, execution_id: str) -> str:
    prefix = STDOUT_PREFIX if stream_name == "stdout" else STDERR_PREFIX
    return f"{prefix}/{execution_id}.txt"


@dataclass(frozen=True)
class TruncationResult:
    text: str
    """The model-visible view (possibly truncated)."""
    artifact_filename: str | None
    """The artifact that stores the full content, if spillover happened."""
    artifact_version: int | None


async def truncate(
    text: str,
    *,
    limit: int,
    stream_name: StreamName,
    execution_id: str,
    invocation_context: InvocationContext,
) -> TruncationResult:
    """Cap ``text`` at ``limit`` characters; spill the full content to an artifact."""
    if len(text) <= limit:
        return TruncationResult(text=text, artifact_filename=None, artifact_version=None)

    if invocation_context.artifact_service is None:
        marker = f"\n---\nOutput exceeded {limit:,} characters. Try again with a smaller output."
        return TruncationResult(
            text=_head_tail(text, limit) + marker, artifact_filename=None, artifact_version=None
        )

    filename = _overflow_filename(stream_name, execution_id)
    part = genai_types.Part(
        inline_data=genai_types.Blob(
            data=text.encode("utf-8"),
            mime_type="text/plain",
        )
    )
    version = await invocation_context.artifact_service.save_artifact(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        filename=filename,
        artifact=part,
        session_id=invocation_context.session.id,
    )
    marker = (
        f"\n---\n"
        f"Output exceeded {limit:,} characters. Try again with a smaller output.\n"
        f"Full {stream_name} saved as artifact: {filename}"
    )
    return TruncationResult(
        text=_head_tail(text, limit) + marker,
        artifact_filename=filename,
        artifact_version=version,
    )


def _head_tail(text: str, limit: int) -> str:
    """Keep a head+tail window of roughly ``limit`` characters."""
    if limit <= 40:
        return text[:limit]
    head_len = int(limit * 0.7)
    tail_len = limit - head_len - 20  # leave room for the elision marker
    if tail_len < 0:
        tail_len = 0
    head = text[:head_len]
    tail = text[-tail_len:] if tail_len > 0 else ""
    return f"{head}\n... [truncated] ...\n{tail}"


__all__ = ["STDERR_PREFIX", "STDOUT_PREFIX", "TruncationResult", "truncate"]
