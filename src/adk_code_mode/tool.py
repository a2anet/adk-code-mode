# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""``ExecuteCodeTool`` — the user-facing ``BaseTool`` implementation.

Wires together the normaliser, namespacer, stub renderer, dispatcher,
workspace bridge, runtime, and output truncator behind a single structured
tool call: the model calls ``execute_code(code=...)`` like any other tool,
instead of writing a fenced code block in plain text that ADK regex-extracts.

A sandbox container is held open for the duration of a **turn** (one ADK
invocation): the first call connects and uploads the tools, later calls reuse
the live session with persistent globals + the working directory, and the
container is released when the turn ends (``release_invocation``, with an
idle reaper as a safety net).
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import datetime as dt
import enum
import logging
import mimetypes
import os
import shutil
import tempfile
import threading
import time
import weakref
from collections.abc import Awaitable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.code_execution_utils import File
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from adk_code_mode._artifact_tools import ARTIFACT_TOOLS
from adk_code_mode.output import truncate
from adk_code_mode.runtime.base import (
    SandboxBackend,
    SandboxConnectionError,
    SandboxResult,
    SandboxSession,
)
from adk_code_mode.runtime.protocol import (
    PROTOCOL_VERSION,
    DoneFrame,
    Frame,
    LogFrame,
    ReadyFrame,
    RunFrame,
    ToolCallFrame,
    ToolErrorPayload,
    ToolResultFrame,
)
from adk_code_mode.tool_result_artifacts import ToolResultArtifactTool
from adk_code_mode.tools import namespacing, normaliser, stubs
from adk_code_mode.tools.catalog import render_catalog
from adk_code_mode.tools.dispatcher import Dispatcher
from adk_code_mode.workspace.files import hash_file, walk_workspace

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest


class ProtocolVersionMismatchError(RuntimeError):
    """Sandbox image's wire protocol does not match the host's."""


class _BlockRunState(enum.Enum):
    """Whether a block's code could have run when the connection dropped mid-block.

    Only ``NOT_RUN`` is safe to re-execute; ``UNKNOWN`` / ``RAN`` are reported back
    to the model instead so a reconnect can't duplicate the block's side effects.
    """

    NOT_RUN = "not_run"  # RunFrame never delivered — code definitely did not run
    UNKNOWN = "unknown"  # RunFrame sent but no DoneFrame — code may have run
    RAN = "ran"  # DoneFrame received — code ran; only its output was lost


class _BlockConnectionLost(Exception):
    """Raised by ``_run_block`` when the connection drops, carrying ``_BlockRunState``."""

    def __init__(self, state: _BlockRunState) -> None:
        super().__init__(f"sandbox connection lost (code {state.value})")
        self.state = state


logger = logging.getLogger("adk_code_mode.tool")

_REAPER_MAX_POLL_SECONDS = 30.0

# One block runs at most twice: the original attempt plus a single reconnect, and
# only when the code never reached the sandbox (``_BlockRunState.NOT_RUN``).
_MAX_BLOCK_ATTEMPTS = 2

# Built-in artifact tools are never wrapped for tool-result saving — saving the
# result of ``save_artifact``/``load_artifact``/``list_artifacts`` is pointless.
_ARTIFACT_TOOL_NAMES = frozenset(tool.name for tool in ARTIFACT_TOOLS)

# Every open turn session is registered here (removed on ``aclose``) so the test
# harness (and any host that wants a shutdown hook) can release sandbox
# containers even if the owning ``ExecuteCodeTool`` is no longer referenced.
_LIVE_TURN_SESSIONS: "set[_TurnSession]" = set()
_LIVE_TURN_SESSIONS_LOCK = threading.Lock()

_TOOLS_OPEN = "<code-mode>"
_TOOLS_CLOSE = "</code-mode>"

_STATIC_DESCRIPTION = (
    "Execute Python code in a sandboxed container. The Python Standard "
    "Library and a set of custom tools are available as an importable "
    "`tools` package (e.g. `from tools.slack import send_message`) — list "
    "`/tools/` and read a file's docstring to see what's available, or "
    "check the system instruction for a full reference catalog if one was "
    "provided. Print anything you want to see; only stdout/stderr are "
    "returned. Variables and the working directory persist across calls "
    "within the same turn and reset on the next turn — use `save_artifact` "
    "/ `load_artifact` (imported the same way) to persist across turns. "
    "Files created or changed by the code are saved as artifacts "
    "automatically and returned by name."
)

_FUNCTION_STUBS_PREAMBLE = (
    "Reference catalog of the functions available inside execute_code's "
    "sandbox. These are not separate callable tools — they are Python "
    "functions to import and call from within the code you pass to "
    "execute_code (e.g. `from tools.slack import send_message`)."
)


ArtifactsSavedCallback = Callable[[InvocationContext, dict[str, int]], Awaitable[None]]

_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class _WeakContextEntry(Generic[_T]):
    ref: weakref.ReferenceType[InvocationContext]
    value: _T


class _WeakContextCache(Generic[_T]):
    """Weak identity map for unhashable ``InvocationContext`` objects."""

    def __init__(self) -> None:
        self._items: dict[int, _WeakContextEntry[_T]] = {}
        self._lock = threading.Lock()

    def get(self, context: InvocationContext) -> _T | None:
        key = id(context)
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if entry.ref() is context:
                return entry.value
            del self._items[key]
        return None

    def set(self, context: InvocationContext, value: _T) -> None:
        key = id(context)

        def remove(ref: weakref.ReferenceType[InvocationContext], key: int = key) -> None:
            with self._lock:
                entry = self._items.get(key)
                if entry is not None and entry.ref is ref:
                    del self._items[key]

        ref = weakref.ref(context, remove)
        with self._lock:
            self._items[key] = _WeakContextEntry(ref=ref, value=value)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class ExecuteCodeTool(BaseTool):
    """Execute model-written Python in a sandbox with access to ADK tools."""

    def __init__(
        self,
        *,
        tools: Sequence[BaseTool | BaseToolset | Callable[..., Any]] = (),
        backend: SandboxBackend,
        include_artifact_tools: bool = True,
        save_tool_results_as_artifacts: bool = True,
        append_function_stubs_to_system_instruction: bool = True,
        max_catalog_chars: int = 50_000,
        max_output_chars: int = 50_000,
        max_code_chars: int = 1_000_000,
        timeout_seconds: int | None = None,
        per_tool_timeout_seconds: float | None = None,
        session_idle_timeout_seconds: float = 600,
        on_artifacts_saved: ArtifactsSavedCallback | None = None,
    ) -> None:
        """Initializes the ExecuteCodeTool.

        Args:
          tools: Tools, toolsets, and/or plain callables made available inside
            the sandbox. Callables are wrapped as ``FunctionTool``.
          backend: The sandbox runtime (``RemoteBackend`` for production,
            ``UnsafeLocalDockerBackend`` for local development).
          include_artifact_tools: If true (default), ``save_artifact`` /
            ``load_artifact`` / ``list_artifacts`` are injected as top-level
            tools. Set ``False`` for a strict tool surface.
          save_tool_results_as_artifacts: If true (default), every non-artifact
            tool is wrapped so its result is also saved as a session artifact;
            large results are elided from the reply and reloadable via
            ``load_artifact``. Set ``False`` to return tool results inline.
          append_function_stubs_to_system_instruction: If true (default), every
            model turn's system instruction gets a ``<code-mode>`` block
            listing every available function's signature and docstring. If the
            rendered block would exceed ``max_catalog_chars``, nothing is
            appended at all for that turn — the model can still discover
            functions by listing ``/tools/`` and reading a stub's docstring
            from within the code it runs.
          max_catalog_chars: Only consulted when
            ``append_function_stubs_to_system_instruction`` is true; see above.
          max_output_chars: Caps stdout/stderr handed back to the model.
            Overflow is saved as a session artifact and the model sees a
            head-and-tail view pointing to it.
          max_code_chars: Rejects oversized code payloads before starting a
            container.
          timeout_seconds: Caps overall execution time of one ``execute_code``
            call.
          per_tool_timeout_seconds: Caps each individual tool call made from
            within the sandbox.
          session_idle_timeout_seconds: Idle reaper: closes a turn's container
            once it goes untouched this long. Backstop for turns that never
            call ``release_invocation``.
          on_artifacts_saved: Optional async callback fired after each
            ``execute_code`` whose sandbox-side ``save_artifact`` calls
            produced new artifact versions. Receives the live
            ``InvocationContext`` and a ``{filename: version}`` dict. Hook
            errors are logged and swallowed — they must not break execution.
        """
        super().__init__(name="execute_code", description=_STATIC_DESCRIPTION)

        adapted: list[BaseTool | BaseToolset] = []
        if include_artifact_tools:
            adapted.extend(ARTIFACT_TOOLS)
        for item in tools:
            if isinstance(item, (BaseTool, BaseToolset)):
                adapted.append(item)
            elif callable(item):
                from google.adk.tools.function_tool import FunctionTool

                adapted.append(FunctionTool(item))
            else:
                raise TypeError(
                    f"Unsupported tool input {item!r}: expected BaseTool, BaseToolset, or callable"
                )
        self._tools = adapted
        self._backend = backend
        self._save_tool_results_as_artifacts = save_tool_results_as_artifacts
        self._append_function_stubs_to_system_instruction = (
            append_function_stubs_to_system_instruction
        )
        self._max_catalog_chars = max_catalog_chars
        self._max_output_chars = max_output_chars
        self._max_code_chars = max_code_chars
        self.timeout_seconds = timeout_seconds
        self._per_tool_timeout_seconds = per_tool_timeout_seconds
        self._session_idle_timeout_seconds = session_idle_timeout_seconds
        self._on_artifacts_saved = on_artifacts_saved

        self._resolved_tools: _WeakContextCache[tuple[str, list[namespacing.NamespacedTool]]] = (
            _WeakContextCache()
        )
        self._execution_locks: _WeakContextCache[asyncio.Lock] = _WeakContextCache()
        self._turns: dict[str, _TurnSession] = {}
        self._turns_lock = threading.Lock()
        self._reaper_task: asyncio.Task[None] | None = None

        # Validate the eagerly-known tool surface (bare BaseTools). Toolset-
        # derived tools are async to resolve, so any collisions involving them
        # surface on first ``execute_code`` instead.
        eager_resolved = [
            normaliser.ResolvedTool(tool=t, toolset=None)
            for t in adapted
            if isinstance(t, BaseTool)
        ]
        if eager_resolved:
            namespacing.build(eager_resolved)

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute.",
                    },
                },
                "required": ["code"],
            },
        )

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: "LlmRequest"
    ) -> None:
        await super().process_llm_request(tool_context=tool_context, llm_request=llm_request)
        if not self._append_function_stubs_to_system_instruction:
            return
        ns_tools = await self._get_or_resolve_tools(tool_context)
        catalog = render_catalog(ns_tools)
        if len(catalog) > self._max_catalog_chars:
            return
        llm_request.append_instructions(
            [_FUNCTION_STUBS_PREAMBLE, f"{_TOOLS_OPEN}\n{catalog}\n{_TOOLS_CLOSE}"]
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        self._ensure_reaper_started()

        invocation_context = tool_context._invocation_context
        lock = self._get_execution_lock(invocation_context)
        async with lock:
            return await self._run_async_locked(args=args, tool_context=tool_context)

    async def _run_async_locked(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        code = args["code"]
        if self._max_code_chars and len(code) > self._max_code_chars:
            return {
                "stdout": "",
                "stderr": (
                    f"Code exceeds maximum allowed length "
                    f"({len(code):,} > {self._max_code_chars:,} chars)."
                ),
                "output_files": [],
            }

        invocation_context = tool_context._invocation_context
        invocation_id = tool_context.invocation_id
        execution_id = tool_context.function_call_id or invocation_context.session.id

        turn = self._turns.get(invocation_id)
        if turn is None:
            ns_tools = await self._get_or_resolve_tools(tool_context)
            prepared = self._prepare_tool_surface(ns_tools)
            turn = await self._create_turn(invocation_id, prepared)
        else:
            prepared = turn.prepared

        turn.mark_in_use()
        try:
            try:
                turn, result, dispatcher = await self._run_block_reconnecting(
                    invocation_id, turn, prepared, invocation_context, code, execution_id
                )
            except asyncio.TimeoutError:
                return _timeout_result(self.timeout_seconds)
            except _BlockConnectionLost as exc:
                return _connection_lost_result(exc.state)

            output_files = turn.workspace.collect_outputs()
            await self._fire_artifacts_saved(invocation_context, dispatcher.artifact_delta)

            stderr = _stderr_with_exit_code(result.stderr, result.exit_code)
            stdout_res, stderr_res = await asyncio.gather(
                truncate(
                    result.stdout,
                    limit=self._max_output_chars,
                    stream_name="stdout",
                    execution_id=execution_id,
                    invocation_context=invocation_context,
                ),
                truncate(
                    stderr,
                    limit=self._max_output_chars,
                    stream_name="stderr",
                    execution_id=execution_id,
                    invocation_context=invocation_context,
                ),
            )
            saved_output_files = await self._save_output_files_as_artifacts(
                output_files, tool_context
            )
            return {
                "stdout": stdout_res.text,
                "stderr": stderr_res.text,
                "output_files": saved_output_files,
            }
        finally:
            turn.mark_idle()

    async def release_invocation(self, invocation_id: str) -> None:
        """Release the turn's sandbox container as soon as the turn ends.

        Wire this to the agent's ``after_agent_callback`` (with the callback's
        ``invocation_id``). An idle reaper (``session_idle_timeout_seconds``)
        is a backstop for turns that never reach this call. No-op when the
        invocation never called ``execute_code``.
        """
        with self._turns_lock:
            turn = self._turns.pop(invocation_id, None)
        if turn is not None:
            await turn.aclose()

    def _ensure_reaper_started(self) -> None:
        if self._reaper_task is not None and not self._reaper_task.done():
            return
        self._reaper_task = asyncio.create_task(_reap_idle_turns(weakref.ref(self)))

    def _get_execution_lock(self, invocation_context: InvocationContext) -> asyncio.Lock:
        lock = self._execution_locks.get(invocation_context)
        if lock is None:
            lock = asyncio.Lock()
            self._execution_locks.set(invocation_context, lock)
        return lock

    async def _get_or_resolve_tools(
        self, tool_context: ToolContext
    ) -> list[namespacing.NamespacedTool]:
        """Resolve + namespace this tool's configured surface, cached per invocation.

        Shared by ``process_llm_request`` (catalog rendering) and ``run_async``
        (stub tree for a new turn) so a toolset is only resolved once per
        invocation regardless of which runs first.
        """
        invocation_context = tool_context._invocation_context
        cached_entry = self._resolved_tools.get(invocation_context)
        if cached_entry is not None:
            cached_invocation_id, cached = cached_entry
            if cached_invocation_id == tool_context.invocation_id:
                return cached
        resolved = await normaliser.resolve(list(self._tools), tool_context)
        ns_tools = namespacing.build(self._apply_tool_result_wrapping(resolved))
        cached_entry = self._resolved_tools.get(invocation_context)
        if cached_entry is not None:
            cached_invocation_id, cached = cached_entry
            if cached_invocation_id == tool_context.invocation_id:
                return cached
        self._resolved_tools.set(invocation_context, (tool_context.invocation_id, ns_tools))
        return ns_tools

    def _prepare_tool_surface(
        self, ns_tools: list[namespacing.NamespacedTool]
    ) -> "_PreparedToolSurface":
        registry = namespacing.Registry(ns_tools)
        tree_files = stubs.render_tree(ns_tools)
        tools_map = {f.path: f.source for f in tree_files}
        return _PreparedToolSurface(registry=registry, tools_map=tools_map)

    def _apply_tool_result_wrapping(
        self, resolved: list[normaliser.ResolvedTool]
    ) -> list[normaliser.ResolvedTool]:
        """Wrap each non-artifact tool so its result is saved as an artifact.

        No-op unless ``save_tool_results_as_artifacts`` is set. Applied right
        before namespacing so both the catalog and the executed stubs expose
        the same signatures. Safe to call on an already-wrapped list.
        """
        if not self._save_tool_results_as_artifacts:
            return resolved
        wrapped: list[normaliser.ResolvedTool] = []
        for rt in resolved:
            if isinstance(rt.tool, ToolResultArtifactTool) or rt.tool.name in _ARTIFACT_TOOL_NAMES:
                wrapped.append(rt)
            else:
                wrapped.append(dataclasses.replace(rt, tool=ToolResultArtifactTool(rt.tool)))
        return wrapped

    async def _run_attempt(
        self,
        turn: "_TurnSession",
        invocation_context: InvocationContext,
        code: str,
        execution_id: str,
    ) -> tuple[SandboxResult, Dispatcher]:
        """Run one code block on ``turn``'s session, bounded by ``timeout_seconds``.

        Drives the per-block contract. Raises ``asyncio.TimeoutError`` on
        timeout and ``_BlockConnectionLost`` if the connection dropped
        mid-block.
        """
        dispatcher = Dispatcher(
            invocation_context=invocation_context,
            registry=turn.prepared.registry,
            execution_id=execution_id,
            per_tool_timeout_seconds=self._per_tool_timeout_seconds,
        )
        result = await asyncio.wait_for(
            _run_block(session=turn.session, dispatcher=dispatcher, code=code),
            timeout=self.timeout_seconds if self.timeout_seconds else None,
        )
        return result, dispatcher

    async def _run_block_reconnecting(
        self,
        invocation_id: str,
        turn: "_TurnSession",
        prepared: "_PreparedToolSurface",
        invocation_context: InvocationContext,
        code: str,
        execution_id: str,
    ) -> tuple["_TurnSession", SandboxResult, Dispatcher]:
        """Run the block, reconnecting once only if the code never reached the sandbox.

        Returns the (possibly reconnected) turn plus its result. On a terminal
        failure the offending turn is already discarded and the cause is re-raised
        for the caller to turn into a user-facing result: ``asyncio.TimeoutError``
        for a timeout, ``_BlockConnectionLost`` for a connection drop (whose
        ``state`` records whether the code ran, so a block that ran — or might
        have — is never silently re-executed), or a protocol/frame ``RuntimeError``.
        """
        for attempt in range(_MAX_BLOCK_ATTEMPTS):
            try:
                result, dispatcher = await self._run_attempt(
                    turn, invocation_context, code, execution_id
                )
                return turn, result, dispatcher
            except asyncio.TimeoutError:
                await self._discard_turn(invocation_id, turn)
                raise
            except _BlockConnectionLost as exc:
                await self._discard_turn(invocation_id, turn)
                if exc.state is not _BlockRunState.NOT_RUN or attempt + 1 >= _MAX_BLOCK_ATTEMPTS:
                    raise
                # Code never ran: reconnect on a fresh (cold) container and retry.
                # A failed reconnect leaves the code un-run, so report it as such.
                try:
                    turn = await self._create_turn(invocation_id, prepared)
                except Exception as reconnect_exc:
                    logger.debug("failed to reconnect turn session", exc_info=True)
                    raise _BlockConnectionLost(_BlockRunState.NOT_RUN) from reconnect_exc
                turn.mark_in_use()
            except RuntimeError:
                await self._discard_turn(invocation_id, turn)
                raise
        raise _BlockConnectionLost(_BlockRunState.NOT_RUN)  # unreachable: loop returns or raises

    async def _create_turn(
        self, invocation_id: str, prepared: "_PreparedToolSurface"
    ) -> "_TurnSession":
        workspace = _TurnWorkspace.create()
        try:
            session = await self._backend.start(
                tools_files=prepared.tools_map,
                workdir_path=workspace.root,
                timeout_seconds=self.timeout_seconds,
            )
        except BaseException:
            workspace.cleanup()
            raise
        turn = _TurnSession(
            session=session,
            workspace=workspace,
            prepared=prepared,
            last_used=time.monotonic(),
        )
        with _LIVE_TURN_SESSIONS_LOCK:
            _LIVE_TURN_SESSIONS.add(turn)
        with self._turns_lock:
            self._turns[invocation_id] = turn
        return turn

    async def _discard_turn(self, invocation_id: str, turn: "_TurnSession") -> None:
        with self._turns_lock:
            if self._turns.get(invocation_id) is turn:
                del self._turns[invocation_id]
        await turn.aclose()

    async def _reap_once(self) -> None:
        """Close and drop turns idle longer than ``session_idle_timeout_seconds``."""
        now = time.monotonic()
        stale: list[_TurnSession] = []
        with self._turns_lock:
            for invocation_id, turn in list(self._turns.items()):
                if not turn.in_use and (now - turn.last_used) > self._session_idle_timeout_seconds:
                    stale.append(turn)
                    del self._turns[invocation_id]
        for turn in stale:
            await turn.aclose()

    async def _fire_artifacts_saved(
        self, invocation_context: InvocationContext, delta: dict[str, int]
    ) -> None:
        if self._on_artifacts_saved is None or not delta:
            return
        try:
            await self._on_artifacts_saved(invocation_context, delta)
        except Exception:
            logger.exception("on_artifacts_saved hook raised; ignoring")

    async def _save_output_files_as_artifacts(
        self, files: list[File], tool_context: ToolContext
    ) -> list[str]:
        """Save changed workspace files as session artifacts; return their filenames.

        A no-op when there's nothing to save or no ``ArtifactService`` is
        configured (mirrors ``ToolResultArtifactTool``'s transparent-pass-
        through guard) — the files still existed in the sandbox during the
        block, they just aren't persisted past the turn in that case.
        """
        if not files or tool_context._invocation_context.artifact_service is None:
            return []
        saved: list[str] = []
        for file in files:
            data = file.content if isinstance(file.content, bytes) else file.content.encode("utf-8")
            await tool_context.save_artifact(
                filename=file.name,
                artifact=genai_types.Part(
                    inline_data=genai_types.Blob(data=data, mime_type=file.mime_type)
                ),
            )
            saved.append(file.name)
        return saved


@dataclasses.dataclass(frozen=True)
class _PreparedToolSurface:
    registry: namespacing.Registry
    tools_map: dict[str, str]


@dataclasses.dataclass
class _TurnWorkspace:
    """One host workspace directory held for the whole turn.

    ``baseline_hashes`` is the rolling per-path content snapshot, advanced to
    the post-block state after each ``collect_outputs`` so the next block only
    reports what *it* changed.
    """

    root: str
    baseline_hashes: dict[str, str]

    @classmethod
    def create(cls) -> "_TurnWorkspace":
        return cls(root=tempfile.mkdtemp(prefix="adk-code-mode-turn-"), baseline_hashes={})

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def collect_outputs(self) -> list[File]:
        """Files whose content differs from the rolling baseline; then advance it."""
        files: list[File] = []
        current: dict[str, str] = {}
        for rel_path in walk_workspace(self.root):
            abs_path = os.path.join(self.root, rel_path)
            try:
                digest, _size = hash_file(abs_path)
            except OSError:
                continue
            current[rel_path] = digest
            if self.baseline_hashes.get(rel_path) == digest:
                continue
            with open(abs_path, "rb") as fh:
                data = fh.read()
            mime_type, _ = mimetypes.guess_type(rel_path)
            files.append(
                File(
                    name=rel_path,
                    content=data,
                    mime_type=mime_type or "application/octet-stream",
                )
            )
        self.baseline_hashes = current
        return files


@dataclasses.dataclass(eq=False)
class _TurnSession:
    """A live sandbox session bound to one invocation (turn).

    ``eq=False`` keeps identity-based equality/hashing so instances can live in
    the ``_LIVE_TURN_SESSIONS`` set and be matched with ``is``.
    """

    session: SandboxSession
    workspace: _TurnWorkspace
    prepared: _PreparedToolSurface
    last_used: float
    in_use: bool = False

    def mark_in_use(self) -> None:
        self.in_use = True
        self.last_used = time.monotonic()

    def mark_idle(self) -> None:
        self.in_use = False
        self.last_used = time.monotonic()

    async def aclose(self) -> None:
        with _LIVE_TURN_SESSIONS_LOCK:
            _LIVE_TURN_SESSIONS.discard(self)
        try:
            await self.session.close()
        except Exception:
            logger.debug("error closing turn session", exc_info=True)
        self.workspace.cleanup()


def _timeout_result(timeout_seconds: int | None) -> dict[str, Any]:
    return {
        "stdout": "",
        "stderr": (
            f"Execution exceeded timeout of {timeout_seconds}s and was terminated."
            if timeout_seconds is not None
            else "Execution timed out and was terminated."
        ),
        "output_files": [],
    }


def _connection_lost_result(state: _BlockRunState) -> dict[str, Any]:
    """Message the model when a connection drop stops us from returning output."""
    if state is _BlockRunState.RAN:
        stderr = (
            "Your code ran, but the sandbox connection dropped before its output could be "
            "returned, so the output is lost. Do not run it again — it already executed."
        )
    elif state is _BlockRunState.UNKNOWN:
        stderr = (
            "The sandbox connection dropped while your code was running, so it is unknown "
            "whether it executed and any output is lost. Do not blindly re-run it — check "
            "whether its effects took place first."
        )
    else:  # NOT_RUN
        stderr = (
            "The sandbox connection was lost before your code ran, so it did not execute. "
            "You can run it again."
        )
    return {"stdout": "", "stderr": stderr, "output_files": []}


def _stderr_with_exit_code(stderr: str, exit_code: int) -> str:
    if exit_code == 0:
        return stderr
    marker = f"Process exited with code {exit_code}."
    if not stderr:
        return marker
    return f"{stderr.rstrip()}\n{marker}"


async def _run_block(
    *,
    session: SandboxSession,
    dispatcher: Dispatcher,
    code: str,
) -> SandboxResult:
    """Run one code block on an already-open session (no connect, no shutdown).

    Per-block contract: begin block → send ``RunFrame`` → drain frames until
    ``DoneFrame`` → read the block's ``OutputFrame``. ``ShutdownFrame`` is *not*
    sent here — it belongs to ``session.close()`` at turn end.

    A mid-block connection drop is re-raised as ``_BlockConnectionLost`` tagged
    with whether the code could have run: the ``RunFrame`` never left the host
    (``NOT_RUN``, safe to retry), a ``DoneFrame`` came back so it did run
    (``RAN``), or the ``RunFrame`` was sent but no ``DoneFrame`` arrived
    (``UNKNOWN``).
    """
    code_sent = False
    done_exit_code: int | None = None
    try:
        await session.begin_block([])
        host_loop = asyncio.create_task(_host_loop(session=session, dispatcher=dispatcher))
        try:
            await session.send(RunFrame(code=code))
            code_sent = True
            done_exit_code = await host_loop
        finally:
            if not host_loop.done():
                host_loop.cancel()
                await asyncio.gather(host_loop, return_exceptions=True)
        result = await session.wait()
    except SandboxConnectionError as exc:
        if done_exit_code is not None:
            state = _BlockRunState.RAN
        elif code_sent:
            state = _BlockRunState.UNKNOWN
        else:
            state = _BlockRunState.NOT_RUN
        raise _BlockConnectionLost(state) from exc
    if done_exit_code is None:
        return result
    return SandboxResult(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=done_exit_code,
    )


async def _host_loop(
    *,
    session: SandboxSession,
    dispatcher: Dispatcher,
) -> int | None:
    """Consume frames from the sandbox until a ``DoneFrame`` arrives.

    ``tool_call`` frames are dispatched concurrently as background tasks so a
    slow tool doesn't block other calls.
    """
    pending: list[asyncio.Task[Any]] = []
    frames = session.frames()
    try:
        async for frame in frames:
            if isinstance(frame, ReadyFrame):
                if frame.protocol_version != PROTOCOL_VERSION:
                    raise ProtocolVersionMismatchError(
                        f"sandbox uses protocol v{frame.protocol_version}, "
                        f"host uses v{PROTOCOL_VERSION}; rebuild the sandbox image"
                    )
                continue
            if isinstance(frame, DoneFrame):
                return frame.exit_code
            if isinstance(frame, ToolCallFrame):
                pending.append(asyncio.create_task(_handle_tool_call(session, dispatcher, frame)))
                continue
            if isinstance(frame, LogFrame):
                logger.log(
                    _level_name_to_int(frame.level),
                    "[code-mode sandbox] %s",
                    frame.msg,
                )
                continue
    finally:
        if pending:
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    return None


async def _handle_tool_call(
    session: SandboxSession,
    dispatcher: Dispatcher,
    frame: ToolCallFrame,
) -> None:
    result = await dispatcher.dispatch(frame.name, frame.args, timeout=frame.timeout)
    try:
        if result.ok:
            reply: Frame = ToolResultFrame(id=frame.id, ok=True, value=_json_safe(result.value))
        else:
            reply = ToolResultFrame(
                id=frame.id,
                ok=False,
                error=ToolErrorPayload(
                    type=result.error_type or "Error",
                    message=result.error_message or "",
                ),
            )
        await session.send(reply)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("failed to send tool_result frame for id=%s", frame.id)
        fallback = ToolResultFrame(
            id=frame.id,
            ok=False,
            error=ToolErrorPayload(
                type=type(exc).__name__,
                message=f"failed to serialise or send tool result: {exc}",
            ),
        )
        try:
            await session.send(fallback)
        except Exception:
            logger.exception("failed to send fallback tool_result frame for id=%s", frame.id)


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of a tool result to a JSON-safe value."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {
            "kind": "bytes",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
            "encoding": "base64",
        }
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    return repr(value)


def _level_name_to_int(name: str) -> int:
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    return levels.get((name or "info").lower(), logging.INFO)


async def _reap_idle_turns(tool_ref: "weakref.ref[ExecuteCodeTool]") -> None:
    """Poll the tool's turns and close idle ones.

    Holds only a weak reference so a dropped ``ExecuteCodeTool`` can be
    garbage-collected; the task then exits on its next tick instead of pinning
    it alive forever.
    """
    while True:
        tool = tool_ref()
        if tool is None:
            return
        interval = max(
            0.05, min(tool._session_idle_timeout_seconds / 2.0, _REAPER_MAX_POLL_SECONDS)
        )
        del tool  # don't keep the tool alive across the sleep
        await asyncio.sleep(interval)
        tool = tool_ref()
        if tool is None:
            return
        await tool._reap_once()


__all__ = ["ArtifactsSavedCallback", "ExecuteCodeTool", "ProtocolVersionMismatchError"]
