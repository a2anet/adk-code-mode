# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Workspace file helpers."""

from __future__ import annotations

import hashlib
import os


def hash_bytes(data: bytes) -> str:
    return hashlib.blake2s(data, digest_size=16).hexdigest()


def hash_file(path: str) -> tuple[str, int]:
    """Return ``(blake2s_hex, size)`` for a file on disk."""
    hasher = hashlib.blake2s(digest_size=16)
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def walk_workspace(root: str) -> list[str]:
    """Return posix-relative paths of every file under ``root``, sorted.

    Symlinks are skipped. Hidden files (starting with ``.``) are included.
    """
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for fn in filenames:
            abs_path = os.path.join(dirpath, fn)
            if os.path.islink(abs_path):
                continue
            rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
            out.append(rel)
    out.sort()
    return out


__all__ = ["hash_bytes", "hash_file", "walk_workspace"]
