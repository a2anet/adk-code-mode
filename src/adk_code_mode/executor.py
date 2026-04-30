# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""``CodeModeCodeExecutor`` — the user-facing ``BaseCodeExecutor`` implementation.

Wires together the normaliser, namespacer, stub renderer, progressive-disclosure
selector, dispatcher, workspace bridge, runtime, and output truncator.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import dataclasses
import datetime as dt
import logging
import mimetypes
import os
import shutil
import tempfile
import threading
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
from adk_code_mode.runtime.base import SandboxHandle, SandboxResult, SandboxRuntime
from adk_code_mode.runtime.protocol import (
    PROTOCOL_VERSION,
    DoneFrame,
    Frame,
    LogFrame,
    ReadyFrame,
    RunFrame,
    ShutdownFrame,
    ToolCallFrame,
    ToolErrorPayload,
    ToolResultFrame,
)
from adk_code_mode.tools import namespacing, normaliser, stubs
from adk_code_mode.tools.dispatcher import Dispatcher
from adk_code_mode.workspace.files import hash_bytes, hash_file, walk_workspace


class ProtocolVersionMismatchError(RuntimeError):
    """Sandbox image's wire protocol does not match the host's."""


logger = logging.getLogger("adk_code_mode.executor")

CODE_MODE_SYSTEM_INSTRUCTION = """\
# How to execute code and use tools
Code you write in a fenced Python block (i.e. ```python) will be executed in a sandbox.
The Python Standard Library and a custom set of tools are available to you.
To see the result of your code, you need to print it. Don't make assumptions about the tool result format.

# How to use files and variables in between executions
Code is executed in a new environment each time. Tool results are automatically saved as Artifacts.
To list available Artifacts, use the `list_artifacts` tool. To save an Artifact, use the `save_artifact` tool, and to load an Artifact, use the `load_artifact` tool.
"""


ArtifactsSavedCallback = Callable[[InvocationContext, dict[str, int]], Awaitable[None]]


class CodeModeCodeExecutor(BaseCodeExecutor):
    """Execute model-written Python in a sandbox with access to ADK tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[BaseTool | BaseToolset] = Field(default_factory=list)
    runtime: SandboxRuntime = Field(...)
    max_output_chars: int = 50_000
    max_catalog_chars: int = 50_000
    """Soft cap on the rendered catalog string. Above this the callback drops
    the per-tool sections and tells the model to navigate ``/tools/`` from
    Python instead."""
    per_tool_timeout_seconds: float | None = None
    include_artifact_tools: bool = True
    """If true (default), ``save_artifact`` / ``load_artifact`` /
    ``list_artifacts`` are injected at the front of ``tools`` as top-level
    tools. Set ``False`` for a strict tool surface."""
    on_artifacts_saved: ArtifactsSavedCallback | None = None
    """Optional async callback fired after each ``execute_code`` whose
    sandbox-side ``save_artifact`` calls produced new artifact versions.
    Receives the live ``InvocationContext`` and a ``{filename: version}``
    dict. The empty case is skipped (no call). Hook errors are logged and
    swallowed — they must not break code execution."""

    stateful: bool = Field(default=True, frozen=True, exclude=True)
    optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

    _bg_loop: "_BackgroundLoop | None" = PrivateAttr(default=None)
    _loop_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _resolution_cache: dict[str, list[normaliser.ResolvedTool]] = PrivateAttr(default_factory=dict)
    _resolution_cache_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    _ROOT_LOOP_REGISTRY: ClassVar[list["_BackgroundLoop"]] = []
    _RESOLUTION_CACHE_LIMIT: ClassVar[int] = 64

    def __init__(
        self,
        *,
        tools: Sequence[BaseTool | BaseToolset | Callable[..., Any]] = (),
        runtime: SandboxRuntime,
        include_artifact_tools: bool = True,
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
            runtime=runtime,
            include_artifact_tools=include_artifact_tools,
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

    def _ensure_background_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._bg_loop is None or not self._bg_loop.is_running:
                self._bg_loop = _BackgroundLoop()
                self._bg_loop.start()
                CodeModeCodeExecutor._ROOT_LOOP_REGISTRY.append(self._bg_loop)
            return self._bg_loop.loop

    async def _aexecute(
        self,
        invocation_context: InvocationContext,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult:
        session_id = invocation_context.session.id
        execution_id = code_execution_input.execution_id or session_id

        prepared = await self._prepare_tool_surface(invocation_context)

        run_workspace = _prepare_run_workspace(code_execution_input.input_files)

        try:
            dispatcher = Dispatcher(
                invocation_context=invocation_context,
                registry=prepared.registry,
                execution_id=execution_id,
                per_tool_timeout_seconds=self.per_tool_timeout_seconds,
            )

            handle = await self.runtime.start(
                tools_files=prepared.tools_map,
                workdir_path=run_workspace.root,
                timeout_seconds=self.timeout_seconds,
            )

            timed_out = False
            try:
                run_task = asyncio.create_task(
                    _run_sandbox(
                        handle=handle,
                        dispatcher=dispatcher,
                        code=code_execution_input.code,
                    )
                )
                result = await asyncio.wait_for(
                    run_task,
                    timeout=self.timeout_seconds if self.timeout_seconds else None,
                )
            except asyncio.TimeoutError:
                timed_out = True
                result = None
            finally:
                await handle.close()

            output_files = run_workspace.collect_output_files()
            await self._fire_artifacts_saved(invocation_context, dispatcher.artifact_delta)

            if timed_out or result is None:
                return CodeExecutionResult(
                    stdout="",
                    stderr=(
                        f"Execution exceeded timeout of {self.timeout_seconds}s and was terminated."
                        if self.timeout_seconds is not None
                        else "Execution timed out and was terminated."
                    ),
                    output_files=[],
                )

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
            run_workspace.cleanup()

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
        ns_tools = namespacing.build(resolved)
        registry = namespacing.Registry(ns_tools)

        tree_files = stubs.render_tree(ns_tools)
        tools_map = {f.path: f.source for f in tree_files}
        return _PreparedToolSurface(
            registry=registry,
            tools_map=tools_map,
        )

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
class _RunWorkspace:
    root: str
    initial_hashes: dict[str, str]

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def collect_output_files(self) -> list[File]:
        files: list[File] = []
        for rel_path in walk_workspace(self.root):
            abs_path = os.path.join(self.root, rel_path)
            try:
                digest, _size = hash_file(abs_path)
            except OSError:
                continue
            if self.initial_hashes.get(rel_path) == digest:
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
        return files


def _prepare_run_workspace(input_files: Sequence[File]) -> _RunWorkspace:
    root = tempfile.mkdtemp(prefix="adk-code-mode-run-")
    seen_paths: set[str] = set()
    initial_hashes: dict[str, str] = {}
    for file in input_files:
        rel_path = _normalise_workspace_rel_path(file.name)
        if rel_path in seen_paths:
            raise ValueError(f"duplicate input file name {file.name!r}")
        seen_paths.add(rel_path)
        abs_path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        data = _input_file_bytes(file)
        with open(abs_path, "wb") as fh:
            fh.write(data)
        initial_hashes[rel_path] = hash_bytes(data)
    return _RunWorkspace(root=root, initial_hashes=initial_hashes)


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


def _stderr_with_exit_code(stderr: str, exit_code: int) -> str:
    if exit_code == 0:
        return stderr
    marker = f"Process exited with code {exit_code}."
    if not stderr:
        return marker
    return f"{stderr.rstrip()}\n{marker}"


async def _run_sandbox(
    *,
    handle: SandboxHandle,
    dispatcher: Dispatcher,
    code: str,
) -> SandboxResult:
    host_loop = asyncio.create_task(_host_loop(handle=handle, dispatcher=dispatcher))
    done_exit_code: int | None = None
    try:
        await handle.send(RunFrame(code=code))
        done_exit_code = await host_loop
    finally:
        if not host_loop.done():
            host_loop.cancel()
            await asyncio.gather(host_loop, return_exceptions=True)
    try:
        await handle.send(ShutdownFrame())
    except Exception:
        pass
    result = await handle.wait()
    if done_exit_code is None:
        return result
    return SandboxResult(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=done_exit_code,
    )


async def _host_loop(
    *,
    handle: SandboxHandle,
    dispatcher: Dispatcher,
) -> int | None:
    """Consume frames from the sandbox until a ``DoneFrame`` arrives.

    ``tool_call`` frames are dispatched concurrently as background tasks so a
    slow tool doesn't block other calls.
    """
    pending: list[asyncio.Task[Any]] = []
    frames = handle.frames()
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
                pending.append(asyncio.create_task(_handle_tool_call(handle, dispatcher, frame)))
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
    handle: SandboxHandle,
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
        await handle.send(reply)
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
            await handle.send(fallback)
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


@atexit.register
def _stop_background_loops() -> None:
    for loop in CodeModeCodeExecutor._ROOT_LOOP_REGISTRY:
        try:
            loop.stop()
        except Exception:
            pass


__all__ = ["ArtifactsSavedCallback", "CODE_MODE_SYSTEM_INSTRUCTION", "CodeModeCodeExecutor"]
