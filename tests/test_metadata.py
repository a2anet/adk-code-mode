# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``<code-mode>`` block: environment cache and tiering."""

from __future__ import annotations

from typing import Iterator

import pytest
from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types

from adk_code_mode import metadata
from adk_code_mode.runtime.protocol import ReadyFrame
from adk_code_mode.tools.namespacing import NamespacedTool, build
from adk_code_mode.tools.normaliser import ResolvedTool


class _SchemaTool(BaseTool):
    def __init__(self, name: str, description: str = "Tool.") -> None:
        super().__init__(name=name, description=description)

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {"message": {"type": "string", "description": "What to say."}},
                "required": ["message"],
            },
        )


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    metadata.reset()
    yield
    metadata.reset()


def _namespaced(count: int = 1) -> list[NamespacedTool]:
    return build([ResolvedTool(tool=_SchemaTool(f"tool_{i}"), toolset=None) for i in range(count)])


def test_record_stores_python_version_and_packages() -> None:
    metadata.record(
        "img",
        ReadyFrame(
            python_version="3.13.2",
            packages={"pandas": {"pandas": "2.3.3"}},
        ),
    )

    block = metadata.render(identity="img", namespaced=[], max_chars=10_000)

    assert "<python-version>3.13.2</python-version>" in block
    assert "pandas 2.3.3" in block


def test_record_ignores_frames_carrying_no_environment() -> None:
    metadata.record(
        "img",
        ReadyFrame(
            python_version="3.13.2",
            packages={"pandas": {"pandas": "2.3.3"}},
        ),
    )
    # An older sandbox, or a reconnect that failed to report, must not blank out
    # an environment already known — that would flip the cached prompt prefix.
    metadata.record("img", ReadyFrame())

    assert "3.13.2" in metadata.render(identity="img", namespaced=[], max_chars=10_000)


def test_import_names_and_distributions_are_rendered_in_sorted_order() -> None:
    packages = {
        "yaml": {"PyYAML": "6.0.2"},
        "google": {
            "google-cloud-storage": "3.2.0",
            "google-api-core": "2.25.1",
        },
    }
    metadata.record(
        "img",
        ReadyFrame(
            python_version="3.13.2",
            packages=packages,
        ),
    )
    metadata.record(
        "img-reversed",
        ReadyFrame(
            python_version="3.13.2",
            packages={
                import_name: dict(reversed(list(distributions.items())))
                for import_name, distributions in reversed(list(packages.items()))
            },
        ),
    )

    block = metadata.render(identity="img", namespaced=[], max_chars=10_000)
    reversed_block = metadata.render(identity="img-reversed", namespaced=[], max_chars=10_000)

    assert reversed_block == block
    assert block.index("google:") < block.index("yaml:")
    assert block.index("google-api-core 2.25.1") < block.index("google-cloud-storage 3.2.0")
    assert "yaml: PyYAML 6.0.2" in block


def test_environments_are_isolated_per_backend_identity() -> None:
    metadata.record(
        "img-a",
        ReadyFrame(
            python_version="3.13.2",
            packages={"pandas": {"pandas": "2.3.3"}},
        ),
    )

    assert "pandas" not in metadata.render(identity="img-b", namespaced=[], max_chars=10_000)


def test_unknown_and_empty_tags_are_omitted_entirely() -> None:
    # Nothing recorded yet, and no tools configured: the block carries only the
    # how-to-use text. A blank <installed-packages> would read as "none exist".
    block = metadata.render(identity="img", namespaced=[], max_chars=10_000)

    assert "<python-version>" not in block
    assert "<installed-packages>" not in block
    assert "<tools-package>" not in block
    assert "<how-to-use>" in block


def test_empty_package_list_omits_the_tag() -> None:
    metadata.record("img", ReadyFrame(python_version="3.13.2", packages={}))

    block = metadata.render(identity="img", namespaced=[], max_chars=10_000)

    assert "<python-version>3.13.2</python-version>" in block
    assert "<installed-packages>" not in block


def test_full_tier_includes_signatures_and_docstrings() -> None:
    block = metadata.render(identity="img", namespaced=_namespaced(), max_chars=10_000)

    assert "def tool_0" in block
    assert "What the model sees" not in block
    assert "message: str" in block


def test_names_tier_keeps_import_lines_only() -> None:
    full = metadata.render(identity="img", namespaced=_namespaced(3), max_chars=100_000)

    block = metadata.render(identity="img", namespaced=_namespaced(3), max_chars=len(full) - 1)

    assert "from tools import tool_0, tool_1, tool_2" in block
    assert "def tool_0" not in block
    assert "message: str" not in block


def test_pointer_tier_is_unconditional() -> None:
    block = metadata.render(identity="img", namespaced=_namespaced(20), max_chars=10)

    assert len(block) > 10  # the floor wins over the budget
    assert metadata.DISCOVER_TOOLS in block
    assert metadata.HOW_TO_USE in block
    assert "from tools import" not in block


def test_budget_is_measured_against_the_final_tagged_block() -> None:
    namespaced = _namespaced(3)
    full = metadata.render(identity="img", namespaced=namespaced, max_chars=100_000)

    exact = metadata.render(identity="img", namespaced=namespaced, max_chars=len(full))

    assert exact == full


def test_tags_are_indented_but_their_text_is_not() -> None:
    metadata.record(
        "img",
        ReadyFrame(
            python_version="3.13.2",
            packages={"pandas": {"pandas": "2.3.3"}},
        ),
    )

    block = metadata.render(identity="img", namespaced=_namespaced(), max_chars=10_000)

    assert "\n  <installed-packages>\npandas: pandas 2.3.3\n  </installed-packages>" in block
    # The tools package body is Python source; indenting it would hand the model
    # a phantom indent level right before asking it to write Python.
    assert "\n  <tools-package>\n# tools\n" in block
    assert block.startswith("<code-mode>\n")
    assert block.endswith("\n</code-mode>")


def test_backend_identity_prefers_the_backends_own_key() -> None:
    class _Backend:
        identity = "https://sandbox.example"

    class _Anonymous:
        pass

    assert metadata.backend_identity(_Backend()) == "https://sandbox.example"
    assert metadata.backend_identity(_Anonymous()) == "_Anonymous"
