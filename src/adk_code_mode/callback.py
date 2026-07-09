# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""``code_mode_before_model_callback``: inject the tool catalog into the prompt.

Wire the result of :func:`code_mode_before_model_callback` as
``before_model_callback`` on an ``LlmAgent``. On every model turn, the
callback:

1. Resolves the tools (including any ``BaseToolset`` instances) for the
   current invocation, using the live ``CallbackContext``.
2. Renders the catalog — signatures + docstrings for every tool —
   falling back to a short overflow message when the rendered catalog
   would exceed ``max_catalog_chars``.
3. Appends ``\\n\\n<code-mode>\\n…\\n</code-mode>`` to
   ``llm_request.config.system_instruction``.
4. Caches the resolved tools on the executor so the follow-up
   ``execute_code`` call reuses them instead of re-resolving toolsets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from adk_code_mode.tools import namespacing, normaliser
from adk_code_mode.tools.catalog import render_catalog, render_overflow_catalog

if TYPE_CHECKING:
    from adk_code_mode.executor import CodeModeCodeExecutor

logger = logging.getLogger("adk_code_mode.callback")

_TOOLS_OPEN = "<code-mode>"
_TOOLS_CLOSE = "</code-mode>"
# Gemini (especially on Vertex) otherwise tries to invoke these catalog entries
# as native function calls, which fail with MALFORMED_FUNCTION_CALL because no
# function tools are declared. The nudge steers it to write Python instead.
_TOOLS_NUDGE = (
    "To call tools you must write Python code in a fenced Python block "
    "(i.e. ```python\\n...\\n```) that imports and runs them. The entries below "
    "are a Python library available in the sandbox, not callable functions — "
    "do not emit a function or tool call."
)


def code_mode_before_model_callback(
    executor: "CodeModeCodeExecutor",
) -> Callable[..., Awaitable[None]]:
    """Build a ``before_model_callback`` bound to ``executor``."""

    async def _callback(
        callback_context: Any,
        llm_request: Any,
        **_unused: Any,
    ) -> None:
        resolved = await normaliser.resolve(list(executor.tools), callback_context)

        executor._record_resolved_tools(callback_context.invocation_id, resolved)

        ns_tools = namespacing.build(resolved)
        catalog = render_catalog(ns_tools)
        if len(catalog) > executor.max_catalog_chars:
            catalog = render_overflow_catalog()
        block = f"{_TOOLS_OPEN}\n{_TOOLS_NUDGE}\n\n{catalog}\n{_TOOLS_CLOSE}"

        _append_system_instruction(llm_request.config, block)

    return _callback


def _append_system_instruction(config: Any, block: str) -> None:
    """Append ``block`` to ``config.system_instruction`` in place.

    Handles the two shapes ADK exposes: ``None`` or ``str``, and a list of
    items (typically strings or ``Part`` objects). The new block is always
    appended as a final string; non-string items in a list are preserved
    as-is.
    """
    existing = config.system_instruction
    if existing is None:
        config.system_instruction = block
        return
    if isinstance(existing, str):
        config.system_instruction = f"{existing}\n\n{block}"
        return
    config.system_instruction = [*existing, block]


__all__ = ["code_mode_before_model_callback"]
