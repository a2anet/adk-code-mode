# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from adk_code_mode import tool as _tool


@pytest.fixture(autouse=True)
async def _release_turn_sessions() -> AsyncIterator[None]:
    """Close any turn-scoped sandbox sessions a test leaves open.

    Turn sessions now stay alive until ``release_invocation`` / the idle
    reaper. Tests that drive ``run_async`` directly never release, so close
    them here to avoid leaking sandbox subprocesses / connections between
    tests.
    """
    yield
    with _tool._LIVE_TURN_SESSIONS_LOCK:
        turns = list(_tool._LIVE_TURN_SESSIONS)
    for turn in turns:
        await turn.aclose()
