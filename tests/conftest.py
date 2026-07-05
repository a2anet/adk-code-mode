# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from adk_code_mode import executor as _executor


@pytest.fixture(autouse=True)
async def _release_turn_sessions() -> AsyncIterator[None]:
    """Close any turn-scoped sandbox sessions a test leaves open.

    Turn sessions now stay alive until ``release_invocation`` / the idle reaper /
    ``atexit``. Tests that drive ``_aexecute`` directly never release, so close
    them here to avoid leaking sandbox subprocesses / connections between tests.
    """
    yield
    with _executor._LIVE_TURN_SESSIONS_LOCK:
        turns = list(_executor._LIVE_TURN_SESSIONS)
    if not turns:
        return
    current = asyncio.get_running_loop()
    for turn in turns:
        if turn.loop is current:
            await turn.aclose()
        else:
            try:
                asyncio.run_coroutine_threadsafe(turn.aclose(), turn.loop)
            except RuntimeError:
                pass
