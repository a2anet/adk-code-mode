# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
import runpy
from pathlib import Path

from adk_code_mode import __version__


def test_version() -> None:
    assert __version__


def test_sandbox_version_matches_host_version() -> None:
    sandbox_about = (
        Path(__file__).resolve().parent.parent
        / "sandbox-wheel"
        / "src"
        / "adk_code_mode_sandbox"
        / "__about__.py"
    )
    sandbox_version = runpy.run_path(str(sandbox_about))["__version__"]
    assert sandbox_version == __version__
