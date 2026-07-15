# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tempfile

from adk_code_mode.workspace import files


def test_walk_workspace_sorted_relative() -> None:
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "nested", "deep"))
        for p in ["a.txt", "z.txt", "nested/b.txt", "nested/deep/c.txt"]:
            with open(os.path.join(root, p), "w") as fh:
                fh.write("x")
        walked = files.walk_workspace(root)
        assert walked == ["a.txt", "nested/b.txt", "nested/deep/c.txt", "z.txt"]
