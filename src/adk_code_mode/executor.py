# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""``CodeModeCodeExecutor`` — the user-facing ``BaseCodeExecutor`` implementation.

Wires together the normaliser, namespacer, stub renderer, progressive-disclosure
selector, dispatcher, workspace bridge, runtime, and output truncator.

A sandbox container is held open for the duration of a **turn** (one ADK
invocation): the first code block connects and uploads the tools, later blocks
reuse the live session with persistent globals + ``/workspace``, and the
container is released when the turn ends (``release_invocation``, with an idle
reaper + ``atexit`` as safety nets).
"""

from __future__ import annotations

import asyncio
import atexit
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
from concurrent.futures import Future
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Callable, ClassVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import (
    CodeExecutionInput,
    CodeExecutionResult,
    File,
)
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from pydantic import ConfigDict, Field, PrivateAttr

from adk_code_mode._artifact_tools import ARTIFACT_TOOLS
from adk_code_mode.output import truncate
from adk_code_mode.tool_result_artifacts import ToolResultArtifactTool
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
from adk_code_mode.tools import namespacing, normaliser, stubs
from adk_code_mode.tools.dispatcher import Dispatcher
from adk_code_mode.workspace.files import hash_bytes, hash_file, walk_workspace


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


logger = logging.getLogger("adk_code_mode.executor")

_REAPER_MAX_POLL_SECONDS = 30.0

# One block runs at most twice: the original attempt plus a single reconnect, and
# only when the code never reached the sandbox (``_BlockRunState.NOT_RUN``).
_MAX_BLOCK_ATTEMPTS = 2

# Built-in artifact tools are never wrapped for tool-result saving — saving the
# result of ``save_artifact``/``load_artifact``/``list_artifacts`` is pointless.
_ARTIFACT_TOOL_NAMES = frozenset(tool.name for tool in ARTIFACT_TOOLS)

# Every open turn session is registered here (removed on ``aclose``) so ``atexit``
# and the test harness can release sandbox containers even if an executor is no
# longer referenced.
_LIVE_TURN_SESSIONS: "set[_TurnSession]" = set()
_LIVE_TURN_SESSIONS_LOCK = threading.Lock()

CODE_MODE_SYSTEM_INSTRUCTION = """\
# How to execute code and use tools
Code you write in a fenced Python block (i.e. ```python) will be executed in a sandbox.
You have no callable function tools. The Python Standard Library and a set of custom tools are available to you as an importable library, listed in the `<code-mode>` section below. To use a tool you must write a fenced Python block that imports and calls it — never respond with a function or tool call.
To see the result of your code, you need to print it.

For example, if you had the following tool:

```
<code-mode>
from tools.slack import send_message

def send_message(*, channel: str, text: str, thread_ts: str | None = ...) -> Any:
    \"\"\"Send a message to a Slack channel.\"\"\"
    ...
</code-mode>
```

To call the tool, you should write:

````
```python
from tools.slack import send_message

print(send_message(channel="C123", text="hi"))
```
````

# How to use files and variables in between executions
Within a turn the sandbox is stateful: variables you define and files you write under `/workspace` (the working directory) persist across the successive code blocks you run before replying to the user. They reset at the start of your next turn.
To carry data across turns, use Artifacts: list them with the `list_artifacts` tool, save with `save_artifact`, and load with `load_artifact`.
"""


ArtifactsSavedCallback = Callable[[InvocationContext, dict[str, int]], Awaitable[None]]


class CodeModeCodeExecutor(BaseCodeExecutor):
    """Execute model-written Python in a sandbox with access to ADK tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[BaseTool | BaseToolset] = Field(default_factory=list)
    backend: SandboxBackend = Field(...)
    max_output_chars: int = 50_000
    max_code_chars: int = 1_000_000
    max_catalog_chars: int = 50_000
    """Soft cap on the rendered catalog string. Above this the callback drops
    the per-tool sections and tells the model to navigate ``/tools/`` from
    Python instead."""
    per_tool_timeout_seconds: float | None = None
    session_idle_timeout_seconds: float = 600
    """Turn sessions idle longer than this are closed by the background idle
    reaper. Safety net for turns that never call ``release_invocation`` (errors,
    missing hook); the primary release is the ``after_agent_callback``."""
    include_artifact_tools: bool = True
    """If true (default), ``save_artifact`` / ``load_artifact`` /
    ``list_artifacts`` are injected at the front of ``tools`` as top-level
    tools. Set ``False`` for a strict tool surface."""
    save_tool_results_as_artifacts: bool = True
    """If true (default), every non-artifact tool is wrapped so its result is
    saved as a session artifact tagged ``code_mode.tool_result``. The model may
    pass optional ``artifact_name`` / ``artifact_description`` to name it; large
    results are elided from the reply and reloadable via ``load_artifact``. Set
    ``False`` to return tool results inline without persisting them."""
    on_artifacts_saved: ArtifactsSavedCallback | None = None
    """Optional async callback fired after each ``execute_code`` whose
    sandbox-side ``save_artifact`` calls produced new artifact versions.
    Receives the live ``InvocationContext`` and a ``{filename: version}``
    dict. The empty case is skipped (no call). Hook errors are logged and
    swallowed — they must not break code execution."""

    code_block_delimiters: list[tuple[str, str]] = [
        ("```python\n", "\n```"),
        ("```tool_code\n", "\n```"),
    ]
    """Prefer ``python`` fences over ``tool_code`` — some Gemini model versions
    interpret ``tool_code`` as a native function-call marker and return
    ``MALFORMED_FUNCTION_CALL`` when no tools are declared."""

    stateful: bool = Field(default=True, frozen=True, exclude=True)
    optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

    _bg_loop: "_BackgroundLoop | None" = PrivateAttr(default=None)
    _loop_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _resolution_cache: dict[str, list[normaliser.ResolvedTool]] = PrivateAttr(default_factory=dict)
    _resolution_cache_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _sessions: dict[str, "_TurnSession"] = PrivateAttr(default_factory=dict)
    _sessions_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    _ROOT_LOOP_REGISTRY: ClassVar[list["_BackgroundLoop"]] = []
    _RESOLUTION_CACHE_LIMIT: ClassVar[int] = 64

    def __init__(
        self,
        *,
        tools: Sequence[BaseTool | BaseToolset | Callable[..., Any]] = (),
        backend: SandboxBackend,
        include_artifact_tools: bool = True,
        save_tool_results_as_artifacts: bool = True,
        **kwargs: Any,
    ) -> None:
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
        super().__init__(
            tools=adapted,
            backend=backend,
            include_artifact_tools=include_artifact_tools,
            save_tool_results_as_artifacts=save_tool_results_as_artifacts,
            **kwargs,
        )
        # Validate the eagerly-known tool surface (bare BaseTools and FunctionTool-
        # wrapped callables). Toolset-derived tools are async to resolve, so any
        # collisions involving them surface on first ``execute_code`` instead.
        eager_resolved = [
            normaliser.ResolvedTool(tool=t, toolset=None)
            for t in adapted
            if isinstance(t, BaseTool)
        ]
        if eager_resolved:
            namespacing.build(eager_resolved)

    def execute_code(
        self,
        invocation_context: InvocationContext,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult:
        loop = self._ensure_background_loop()
        fut: Future[CodeExecutionResult] = asyncio.run_coroutine_threadsafe(
            self._aexecute(invocation_context, code_execution_input), loop
        )
        return fut.result()

    def release_invocation(self, invocation_id: str) -> None:
        """Release the turn's sandbox container as soon as the turn ends.

        Wire this to the agent's ``after_agent_callback`` (with the callback's
        ``invocation_id``). Pops the cached session and schedules its close onto
        the background loop the session is bound to — never awaits a loop-bound
        session from ADK's callback loop. No-op when the invocation had no code.
        """
        with self._sessions_lock:
            turn = self._sessions.pop(invocation_id, None)
        if turn is not None:
            self._schedule_close(turn)

    def _ensure_background_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._bg_loop is None or not self._bg_loop.is_running:
                self._bg_loop = _BackgroundLoop()
                self._bg_loop.start()
                CodeModeCodeExecutor._ROOT_LOOP_REGISTRY.append(self._bg_loop)
                # Idle reaper runs as a task on the loop; the loop's ``stop()``
                # drain cancels it at shutdown. It holds only a weakref to the
                # executor so a dropped executor can still be garbage-collected.
                asyncio.run_coroutine_threadsafe(
                    _reap_idle_sessions(weakref.ref(self)), self._bg_loop.loop
                )
            return self._bg_loop.loop

    async def _aexecute(
        self,
        invocation_context: InvocationContext,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult:
        invocation_id = invocation_context.invocation_id
        execution_id = code_execution_input.execution_id or invocation_context.session.id

        code = code_execution_input.code
        if self.max_code_chars and len(code) > self.max_code_chars:
            return CodeExecutionResult(
                stdout="",
                stderr=f"Code exceeds maximum allowed length ({len(code):,} > {self.max_code_chars:,} chars).",
                output_files=[],
            )

        turn = self._get_cached_turn(invocation_id)
        if turn is None:
            prepared = await self._prepare_tool_surface(invocation_context)
            turn = await self._create_turn(invocation_id, prepared)
        else:
            prepared = turn.prepared

        turn.mark_in_use()
        try:
            try:
                turn, result, dispatcher = await self._run_block_reconnecting(
                    invocation_id,
                    turn,
                    prepared,
                    invocation_context,
                    code,
                    execution_id,
                    code_execution_input.input_files,
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
                    limit=self.max_output_chars,
                    stream_name="stdout",
                    execution_id=execution_id,
                    invocation_context=invocation_context,
                ),
                truncate(
                    stderr,
                    limit=self.max_output_chars,
                    stream_name="stderr",
                    execution_id=execution_id,
                    invocation_context=invocation_context,
                ),
            )
            return CodeExecutionResult(
                stdout=stdout_res.text,
                stderr=stderr_res.text,
                output_files=output_files,
            )
        finally:
            turn.mark_idle()

    async def _run_attempt(
        self,
        turn: "_TurnSession",
        invocation_context: InvocationContext,
        code: str,
        execution_id: str,
        input_files: Sequence[File],
    ) -> tuple[SandboxResult, Dispatcher]:
        """Run one code block on ``turn``'s session, bounded by ``timeout_seconds``.

        Stages this block's inputs into the turn workspace first, then drives the
        per-block contract. Raises ``asyncio.TimeoutError`` on timeout and
        ``_BlockConnectionLost`` if the connection dropped mid-block.
        """
        dispatcher = Dispatcher(
            invocation_context=invocation_context,
            registry=turn.prepared.registry,
            execution_id=execution_id,
            per_tool_timeout_seconds=self.per_tool_timeout_seconds,
        )
        input_paths = turn.workspace.stage_inputs(input_files)
        result = await asyncio.wait_for(
            _run_block(
                session=turn.session,
                dispatcher=dispatcher,
                code=code,
                input_paths=input_paths,
            ),
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
        input_files: Sequence[File],
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
                    turn, invocation_context, code, execution_id, input_files
                )
                return turn, result, dispatcher
            except asyncio.TimeoutError:
                self._discard_turn(invocation_id, turn)
                raise
            except _BlockConnectionLost as exc:
                self._discard_turn(invocation_id, turn)
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
                self._discard_turn(invocation_id, turn)
                raise
        raise _BlockConnectionLost(_BlockRunState.NOT_RUN)  # unreachable: loop returns or raises

    def _get_cached_turn(self, invocation_id: str) -> "_TurnSession | None":
        with self._sessions_lock:
            return self._sessions.get(invocation_id)

    async def _create_turn(
        self, invocation_id: str, prepared: "_PreparedToolSurface"
    ) -> "_TurnSession":
        workspace = _TurnWorkspace.create()
        try:
            session = await self.backend.start(
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
            loop=asyncio.get_running_loop(),
            last_used=time.monotonic(),
        )
        with _LIVE_TURN_SESSIONS_LOCK:
            _LIVE_TURN_SESSIONS.add(turn)
        with self._sessions_lock:
            self._sessions[invocation_id] = turn
        return turn

    def _discard_turn(self, invocation_id: str, turn: "_TurnSession") -> None:
        with self._sessions_lock:
            if self._sessions.get(invocation_id) is turn:
                del self._sessions[invocation_id]
        self._schedule_close(turn)

    def _schedule_close(self, turn: "_TurnSession") -> None:
        try:
            asyncio.run_coroutine_threadsafe(turn.aclose(), turn.loop)
        except RuntimeError:
            # Loop not running (e.g. torn down); nothing more we can safely do.
            logger.debug("could not schedule turn session close", exc_info=True)

    async def _reap_once(self) -> None:
        """Close and drop sessions idle longer than ``session_idle_timeout_seconds``."""
        now = time.monotonic()
        stale: list[_TurnSession] = []
        with self._sessions_lock:
            for invocation_id, turn in list(self._sessions.items()):
                if not turn.in_use and (now - turn.last_used) > self.session_idle_timeout_seconds:
                    stale.append(turn)
                    del self._sessions[invocation_id]
        for turn in stale:
            await turn.aclose()

    async def _fire_artifacts_saved(
        self, invocation_context: InvocationContext, delta: dict[str, int]
    ) -> None:
        if self.on_artifacts_saved is None or not delta:
            return
        try:
            await self.on_artifacts_saved(invocation_context, delta)
        except Exception:
            logger.exception("on_artifacts_saved hook raised; ignoring")

    async def _prepare_tool_surface(
        self, invocation_context: InvocationContext
    ) -> "_PreparedToolSurface":
        cached = self._consume_resolved_tools(invocation_context.invocation_id)
        if cached is not None:
            resolved = cached
        else:
            resolved = await normaliser.resolve(
                list(self.tools), ReadonlyContext(invocation_context)
            )
        ns_tools = namespacing.build(self.apply_tool_result_wrapping(resolved))
        registry = namespacing.Registry(ns_tools)

        tree_files = stubs.render_tree(ns_tools)
        tools_map = {f.path: f.source for f in tree_files}
        return _PreparedToolSurface(
            registry=registry,
            tools_map=tools_map,
        )

    def apply_tool_result_wrapping(
        self, resolved: list[normaliser.ResolvedTool]
    ) -> list[normaliser.ResolvedTool]:
        """Wrap each non-artifact tool so its result is saved as an artifact.

        No-op unless ``save_tool_results_as_artifacts`` is set. Applied to the
        resolved surface right before namespacing so both the catalog (rendered
        by ``code_mode_before_model_callback``) and the executed stubs expose
        the same signatures. Safe to call on an already-wrapped list.
        """
        if not self.save_tool_results_as_artifacts:
            return resolved
        wrapped: list[normaliser.ResolvedTool] = []
        for rt in resolved:
            if isinstance(rt.tool, ToolResultArtifactTool) or rt.tool.name in _ARTIFACT_TOOL_NAMES:
                wrapped.append(rt)
            else:
                wrapped.append(dataclasses.replace(rt, tool=ToolResultArtifactTool(rt.tool)))
        return wrapped

    def _record_resolved_tools(
        self, invocation_id: str, resolved: list[normaliser.ResolvedTool]
    ) -> None:
        """Store a resolved tool list for ``_prepare_tool_surface`` to consume.

        Used by ``code_mode_before_model_callback`` to avoid re-resolving
        toolsets in the follow-up ``execute_code`` call. FIFO eviction kicks
        in once the cache exceeds ``_RESOLUTION_CACHE_LIMIT`` so an
        invocation that never reaches the executor (model rejected, error
        before code ran, etc.) doesn't leak forever.
        """
        with self._resolution_cache_lock:
            cache = self._resolution_cache
            cache[invocation_id] = resolved
            while len(cache) > self._RESOLUTION_CACHE_LIMIT:
                cache.pop(next(iter(cache)))

    def _consume_resolved_tools(self, invocation_id: str) -> list[normaliser.ResolvedTool] | None:
        """Pop and return cached resolved tools for ``invocation_id``.

        Returns ``None`` when no entry is cached for this id (e.g. the
        callback wasn't wired or the cache evicted it); the executor then
        resolves fresh. The argument itself must always be a real id.
        """
        with self._resolution_cache_lock:
            return self._resolution_cache.pop(invocation_id, None)


@dataclasses.dataclass(frozen=True)
class _PreparedToolSurface:
    registry: namespacing.Registry
    tools_map: dict[str, str]


@dataclasses.dataclass
class _TurnWorkspace:
    """One host workspace directory held for the whole turn.

    ``baseline_hashes`` is the rolling per-path content snapshot: staged inputs
    are folded in (so they aren't misreported as outputs), and it is advanced to
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

    def stage_inputs(self, input_files: Sequence[File]) -> list[str]:
        """Write this block's inputs into the workspace; return their rel paths."""
        staged: list[str] = []
        seen: set[str] = set()
        for file in input_files:
            rel_path = _normalise_workspace_rel_path(file.name)
            if rel_path in seen:
                raise ValueError(f"duplicate input file name {file.name!r}")
            seen.add(rel_path)
            abs_path = os.path.join(self.root, rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            data = _input_file_bytes(file)
            with open(abs_path, "wb") as fh:
                fh.write(data)
            self.baseline_hashes[rel_path] = hash_bytes(data)
            staged.append(rel_path)
        return staged

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
    loop: asyncio.AbstractEventLoop
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


def _normalise_workspace_rel_path(raw_path: str) -> str:
    candidate = raw_path.replace("\\", "/")
    path = PurePosixPath(candidate)
    if path.is_absolute():
        raise ValueError(f"input file path must be relative: {raw_path!r}")
    parts = [part for part in path.parts if part not in ("", ".")]
    if any(part == ".." for part in parts) or not parts:
        raise ValueError(f"invalid input file path: {raw_path!r}")
    return "/".join(parts)


def _input_file_bytes(file: File) -> bytes:
    if isinstance(file.content, bytes):
        return file.content
    return base64.b64decode(file.content)


def _timeout_result(timeout_seconds: int | None) -> CodeExecutionResult:
    return CodeExecutionResult(
        stdout="",
        stderr=(
            f"Execution exceeded timeout of {timeout_seconds}s and was terminated."
            if timeout_seconds is not None
            else "Execution timed out and was terminated."
        ),
        output_files=[],
    )


def _connection_lost_result(state: _BlockRunState) -> CodeExecutionResult:
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
    return CodeExecutionResult(stdout="", stderr=stderr, output_files=[])


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
    input_paths: Sequence[str],
) -> SandboxResult:
    """Run one code block on an already-open session (no connect, no shutdown).

    Per-block contract: stage inputs → send ``RunFrame`` → drain frames until
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
        await session.begin_block(input_paths)
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


class _BackgroundLoop:
    """A dedicated asyncio loop running on a daemon thread.

    Used so ``execute_code`` (sync) can drive async work even when the caller
    already has a running event loop (Jupyter, ``adk web``, etc.).
    """

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="adk-code-mode-loop", daemon=True)
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._thread.is_alive()

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self._running = False

    def stop(self) -> None:
        if not self._running:
            return

        # Cancel and await any in-flight tasks so subprocesses / sockets they own
        # get a chance to clean up before we tear the loop down.
        async def _drain() -> None:
            current = asyncio.current_task(self.loop)
            tasks = [t for t in asyncio.all_tasks(self.loop) if t is not current and not t.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_drain(), self.loop).result(timeout=5.0)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5.0)
        try:
            self.loop.close()
        except Exception:
            pass


async def _reap_idle_sessions(executor_ref: "weakref.ref[CodeModeCodeExecutor]") -> None:
    """Poll the executor's sessions and close idle ones.

    Holds only a weak reference so a dropped executor can be garbage-collected;
    the task then exits on its next tick instead of pinning the executor for the
    life of the background loop.
    """
    while True:
        executor = executor_ref()
        if executor is None:
            return
        interval = max(
            0.05, min(executor.session_idle_timeout_seconds / 2.0, _REAPER_MAX_POLL_SECONDS)
        )
        del executor  # don't keep the executor alive across the sleep
        await asyncio.sleep(interval)
        executor = executor_ref()
        if executor is None:
            return
        await executor._reap_once()


def _close_live_sessions_blocking() -> None:
    """Schedule every open turn session's close onto its loop and wait briefly."""
    with _LIVE_TURN_SESSIONS_LOCK:
        turns = list(_LIVE_TURN_SESSIONS)
    futures = []
    for turn in turns:
        try:
            futures.append(asyncio.run_coroutine_threadsafe(turn.aclose(), turn.loop))
        except RuntimeError:
            pass
    for fut in futures:
        try:
            fut.result(timeout=5.0)
        except Exception:
            pass


@atexit.register
def _stop_background_loops() -> None:
    # Close sandbox containers before tearing down the loops they run on.
    _close_live_sessions_blocking()
    for loop in CodeModeCodeExecutor._ROOT_LOOP_REGISTRY:
        try:
            loop.stop()
        except Exception:
            pass


__all__ = ["ArtifactsSavedCallback", "CODE_MODE_SYSTEM_INSTRUCTION", "CodeModeCodeExecutor"]
