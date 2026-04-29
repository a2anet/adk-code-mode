# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``runtime/docker`` helpers that don't require Docker."""

from __future__ import annotations

import socket
import threading

import pytest

from adk_code_mode.runtime.docker import _accept_connection


def _bind_listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    return listener


def test_accept_connection_returns_token_authenticated_peer() -> None:
    listener = _bind_listener()
    port = listener.getsockname()[1]

    def _real_sandbox() -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s.sendall(b"good-token\n")
        # Hold the connection open so the host can return it.
        s.recv(0)

    threading.Thread(target=_real_sandbox, daemon=True).start()
    conn = _accept_connection(listener, timeout=3.0, expected_token="good-token")
    assert conn is not None
    conn.close()
    listener.close()


def test_accept_connection_drops_bad_token_and_keeps_waiting() -> None:
    listener = _bind_listener()
    port = listener.getsockname()[1]

    def _imposter() -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s.sendall(b"wrong-token\n")
        try:
            s.recv(1)
        except OSError:
            pass

    def _real_sandbox() -> None:
        # Race the imposter; either ordering is fine.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s.sendall(b"good-token\n")
        s.recv(0)

    threading.Thread(target=_imposter, daemon=True).start()
    threading.Thread(target=_real_sandbox, daemon=True).start()

    conn = _accept_connection(listener, timeout=3.0, expected_token="good-token")
    assert conn is not None
    conn.close()
    listener.close()


def test_accept_connection_times_out_when_no_one_authenticates() -> None:
    listener = _bind_listener()

    with pytest.raises((TimeoutError, socket.timeout)):
        _accept_connection(listener, timeout=0.2, expected_token="good-token")
    listener.close()
