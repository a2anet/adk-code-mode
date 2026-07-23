# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for sandbox environment reporting."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from adk_code_mode_sandbox import _entry  # type: ignore[import-not-found]


@dataclass(frozen=True)
class _Distribution:
    name: str
    version: str


def test_ready_frame_maps_import_names_to_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributions = [
        _Distribution(name="Provider-Z", version="2.0"),
        _Distribution(name="PyYAML", version="6.0.2"),
        _Distribution(name="Provider-A", version="1.0"),
        _Distribution(name="websockets", version="15.0"),
    ]
    package_distributions = {
        "yaml": ["PyYAML"],
        "shared": ["Provider-Z", "Provider-A"],
        "websockets": ["websockets"],
    }
    monkeypatch.setattr(_entry.importlib.metadata, "distributions", lambda: distributions)
    monkeypatch.setattr(
        _entry.importlib.metadata,
        "packages_distributions",
        lambda: package_distributions,
    )

    frame = _entry.ready_frame()

    assert frame.packages == {
        "shared": {
            "Provider-A": "1.0",
            "Provider-Z": "2.0",
        },
        "yaml": {"PyYAML": "6.0.2"},
    }
    assert list(frame.packages) == ["shared", "yaml"]
    assert list(frame.packages["shared"]) == ["Provider-A", "Provider-Z"]
