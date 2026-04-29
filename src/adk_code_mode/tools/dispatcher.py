# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side tool dispatcher for code-mode calls.

Replicates the before/after/on-error callback chain from
``google.adk.flows.llm_flows.functions._execute_single_function_call_async``
using only public attributes on ``InvocationContext`` and ``LlmAgent``. This
insulates us from private-API drift.

Call sequence matches ADK's own flow:

1. ``plugin_manager.run_before_tool_callback``
2. ``agent.canonical_before_tool_callbacks`` (first non-``None`` wins)
3. ``tool.run_async(args=..., tool_context=...)``
4. On exception: ``plugin_manager.run_on_tool_error_callback`` →
   ``agent.canonical_on_tool_error_callbacks``. If no replacement, re-raise.
5. ``plugin_manager.run_after_tool_callback``
6. ``agent.canonical_after_tool_callbacks`` (first non-``None`` replaces result)
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import dataclass
from typing import Any

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event_actions import EventActions
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from adk_code_mode.tools.namespacing import Registry


def _deep_merge_dicts(d1: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``d2`` into ``d1``. Mirrors ADK's ``deep_merge_dicts``."""
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            d1[key] = _deep_merge_dicts(d1[key], value)
        else:
            d1[key] = value
    return d1


@dataclass(frozen=True)
class DispatchResult:
    """Result of a single tool call."""

    ok: bool
    value: Any = None
    error_type: str | None = None
    error_message: str | None = None


class UnsupportedToolActionError(RuntimeError):
    """Raised when a tool requests ADK actions code mode cannot surface."""


class Dispatcher:
    """Dispatches host-side tool calls on behalf of the sandbox."""

    def __init__(
        self,
        *,
        invocation_context: InvocationContext,
        registry: Registry,
        execution_id: str,
        per_tool_timeout_seconds: float | None = None,
    ) -> None:
        self._ctx = invocation_context
        self._registry = registry
        self._execution_id = execution_id
        self._counter = 0
        self._per_tool_timeout = per_tool_timeout_seconds
        self._merge_lock = asyncio.Lock()
        self._artifact_delta: dict[str, int] = {}

    @property
    def artifact_delta(self) -> dict[str, int]:
        return dict(self._artifact_delta)

    def _next_call_id(self) -> str:
        self._counter += 1
        return f"code_mode-{self._execution_id}-{self._counter}"

    def _agent(self, invocation_context: InvocationContext | Any) -> Any:
        """Return the invocation's agent if it carries ADK callback attributes.

        Duck-typed so custom agent subclasses and test doubles work without
        needing to inherit from ``LlmAgent``.
        """
        agent = invocation_context.agent
        if agent is None:
            return None
        if (
            hasattr(agent, "canonical_before_tool_callbacks")
            or hasattr(agent, "canonical_after_tool_callbacks")
            or hasattr(agent, "canonical_on_tool_error_callbacks")
        ):
            return agent
        return None

    def _new_tool_context(
        self, invocation_context: InvocationContext | Any, call_id: str
    ) -> ToolContext:
        synthetic_call = genai_types.FunctionCall(id=call_id, name="code_mode")
        return ToolContext(
            invocation_context=invocation_context,
            function_call_id=synthetic_call.id,
        )

    async def dispatch(
        self, name: str, args: dict[str, Any], timeout: float | None = None
    ) -> DispatchResult:
        """Resolve and run one tool call. Never raises."""
        try:
            nt = self._registry.resolve_call(name)
        except KeyError as exc:
            return DispatchResult(ok=False, error_type="ToolNotFound", error_message=str(exc))

        tool = nt.resolved.tool
        call_args = copy.deepcopy(args)
        # Clamp, don't override: the per-call wire timeout is attacker-controllable
        # (sandbox code can call _rpc_client.call directly with any value), so the
        # host's per_tool_timeout_seconds must remain a true ceiling.
        if timeout is None:
            effective_timeout = self._per_tool_timeout
        elif self._per_tool_timeout is None:
            effective_timeout = timeout
        else:
            effective_timeout = min(timeout, self._per_tool_timeout)
        try:
            call = self._run_and_merge(tool, call_args)
            if effective_timeout is not None:
                result = await asyncio.wait_for(call, timeout=effective_timeout)
            else:
                result = await call
            return DispatchResult(ok=True, value=result)
        except asyncio.TimeoutError:
            return DispatchResult(
                ok=False,
                error_type="TimeoutError",
                error_message=f"tool {tool.name!r} exceeded timeout of {effective_timeout}s",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return DispatchResult(
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    async def _run_and_merge(self, tool: BaseTool, args: dict[str, Any]) -> Any:
        # Mirrors ADK's handle_function_calls_async: tools share live session.state
        # for reads, write to per-call state_delta, then deltas deep-merge back in.
        response, tool_context = await self._run_with_callbacks(
            invocation_context=self._ctx,
            tool=tool,
            args=args,
        )
        async with self._merge_lock:
            _deep_merge_dicts(self._ctx.session.state, tool_context.actions.state_delta)
            self._artifact_delta.update(tool_context.actions.artifact_delta)
        return response

    async def _run_with_callbacks(
        self,
        *,
        invocation_context: InvocationContext | Any,
        tool: BaseTool,
        args: dict[str, Any],
    ) -> tuple[Any, ToolContext]:
        call_id = self._next_call_id()
        tool_context = self._new_tool_context(invocation_context, call_id)
        plugin_manager = invocation_context.plugin_manager
        agent = self._agent(invocation_context)

        response: Any = await plugin_manager.run_before_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_context
        )

        if response is None and agent is not None:
            for callback in getattr(agent, "canonical_before_tool_callbacks", []):
                response = callback(tool=tool, args=args, tool_context=tool_context)
                if inspect.isawaitable(response):
                    response = await response
                if response:
                    break

        if response is None:
            try:
                response = await tool.run_async(args=args, tool_context=tool_context)
            except BaseException as tool_error:
                override = await self._run_on_error(
                    invocation_context, tool, args, tool_context, tool_error
                )
                if override is not None:
                    response = override
                else:
                    raise

        altered = await plugin_manager.run_after_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_context, result=response
        )
        if altered is None and agent is not None:
            for callback in getattr(agent, "canonical_after_tool_callbacks", []):
                altered = callback(
                    tool=tool, args=args, tool_context=tool_context, tool_response=response
                )
                if inspect.isawaitable(altered):
                    altered = await altered
                if altered:
                    break
        if altered is not None:
            response = altered

        self._ensure_supported_actions(tool.name, tool_context.actions)

        if tool.is_long_running and not response:
            # ADK's normal flow keeps a pending function call open and resumes when the
            # async result arrives. Code mode has no such resume path — the sandbox is
            # blocked waiting for a synchronous tool result.
            raise UnsupportedToolActionError(
                f"tool {tool.name!r} is long-running and returned no immediate response; "
                "long-running tools that yield asynchronously are not supported in code mode"
            )
        return response, tool_context

    async def _run_on_error(
        self,
        invocation_context: InvocationContext | Any,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        error: BaseException,
    ) -> Any:
        plugin_manager = invocation_context.plugin_manager
        # Plugin hook only accepts ``Exception``; promote other ``BaseException``
        # subclasses (e.g. ``KeyboardInterrupt``) without trying to hand them off.
        if not isinstance(error, Exception):
            return None
        error_response = await plugin_manager.run_on_tool_error_callback(
            tool=tool, tool_args=args, tool_context=tool_context, error=error
        )
        if error_response is not None:
            return error_response
        agent = self._agent(invocation_context)
        if agent is None:
            return None
        for callback in getattr(agent, "canonical_on_tool_error_callbacks", []):
            error_response = callback(tool=tool, args=args, tool_context=tool_context, error=error)
            if inspect.isawaitable(error_response):
                error_response = await error_response
            if error_response is not None:
                return error_response
        return None

    def _ensure_supported_actions(self, tool_name: str, actions: EventActions) -> None:
        unsupported: list[str] = []
        if actions.requested_tool_confirmations:
            unsupported.append("tool confirmations")
        if actions.requested_auth_configs:
            unsupported.append("credential requests")
        if actions.render_ui_widgets:
            unsupported.append("UI widgets")
        if actions.transfer_to_agent:
            unsupported.append("agent transfer")
        if actions.escalate:
            unsupported.append("escalation")
        if actions.compaction is not None:
            unsupported.append("event compaction")
        if actions.end_of_agent:
            unsupported.append("end_of_agent")
        if actions.agent_state is not None:
            unsupported.append("agent state")
        if actions.rewind_before_invocation_id is not None:
            unsupported.append("rewind")
        if unsupported:
            joined = ", ".join(unsupported)
            raise UnsupportedToolActionError(
                f"tool {tool_name!r} requested unsupported ADK actions in code mode: {joined}"
            )


__all__ = ["Dispatcher", "DispatchResult", "UnsupportedToolActionError"]
