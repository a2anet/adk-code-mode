# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Docker-backed tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SANDBOX_WHEEL_ROOT = _REPO_ROOT / "sandbox-wheel"
_SANDBOX_DIST = _SANDBOX_WHEEL_ROOT / "dist"


def docker_ok() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def build_sandbox_wheel() -> Path:
    subprocess.run(
        ["uv", "build", "--wheel"],
        cwd=_SANDBOX_WHEEL_ROOT,
        check=True,
        timeout=180,
    )
    wheels = sorted(_SANDBOX_DIST.glob("adk_code_mode_sandbox-*.whl"))
    assert wheels, "sandbox wheel build succeeded but no wheel was produced"
    return wheels[-1]


def build_sandbox_image(*, image_tag: str, sandbox_wheel: Path) -> str:
    # Ensure the image uses the same Dockerfile path as `make docker-image`.
    wheel_dest = _SANDBOX_DIST / sandbox_wheel.name
    if sandbox_wheel.resolve() != wheel_dest.resolve():
        wheel_dest.write_bytes(sandbox_wheel.read_bytes())
    subprocess.run(
        [
            "docker",
            "build",
            "-f",
            "docker/Dockerfile",
            "-t",
            image_tag,
            str(_REPO_ROOT),
        ],
        check=True,
        timeout=180,
    )
    return image_tag
