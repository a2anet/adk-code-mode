# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Stop the running code block when the host asks, keeping the sandbox's state.

The host sends an ``InterruptFrame`` when a block runs past its timeout. Raising
``CodeTimeout`` in the thread running the block ends it like any other exception,
so its output so far, the globals and the working directory all survive for the
turn's next block. Code stuck outside the interpreter (a C call, a long
``time.sleep``) only sees the exception once it returns, which is why the host
still falls back to restarting the sandbox after a grace period.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Iterator
from contextlib import contextmanager


class CodeTimeout(BaseException):
    """Raised in running code when the host stops it at its timeout.

    A ``BaseException`` so the code's own ``except Exception`` can't swallow it.
    """

    message = ""

    # Tracebacks print a builtin's bare name, so the code sees `CodeTimeout: ...`.
    __module__ = "builtins"

    def __init__(self, *args: object) -> None:
        # The interpreter raises an asynchronous exception without arguments.
        super().__init__(*(args or (type(self).message,)))


_lock = threading.Lock()
_running_thread: int | None = None


def _set_async_exc(thread_id: int, exc: type[BaseException] | None) -> None:
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), ctypes.py_object(exc) if exc else None
    )


@contextmanager
def running() -> Iterator[None]:
    """Mark the current thread as the one running a code block."""
    global _running_thread
    with _lock:
        _running_thread = threading.get_ident()
    try:
        yield
    finally:
        with _lock:
            _running_thread = None
            # A stop that arrived as the block finished must not fire in the
            # sandbox's own code afterwards.
            _set_async_exc(threading.get_ident(), None)


def interrupt(message: str) -> None:
    """Raise ``CodeTimeout`` with ``message`` in the running block, if there is one."""
    with _lock:
        if _running_thread is None:
            return
        CodeTimeout.message = message
        _set_async_exc(_running_thread, CodeTimeout)
