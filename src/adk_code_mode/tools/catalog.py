# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Render the tool catalog string injected into the model's system prompt.

The catalog is grouped by module: ``# tools.<namespace>`` sections (sorted),
then ``# tools`` for any top-level tools. Each section opens with an import
line and is followed by ``.pyi``-style function definitions (signature +
docstring + ``...`` body). ``ExecuteCodeTool.process_llm_request`` wraps the
result in ``<code-mode>`` / ``</code-mode>`` tags before appending it to the
system instruction; this module returns just the inner content.
"""

from __future__ import annotations

from adk_code_mode.tools.namespacing import NamespacedTool
from adk_code_mode.tools.stubs import RenderedTool, render_tool


def _render_entry(rt: RenderedTool) -> str:
    """One ``.pyi``-style entry: signature + docstring + ``...`` placeholder."""
    return f"{rt.signature_for('catalog')}\n    {rt.docstring}\n    ...\n"


def _render_module_section(*, header: str, import_line: str, entries: list[str]) -> str:
    blocks = "\n".join(entries)
    return f"# {header}\n\n{import_line}\n\n{blocks}"


def render_catalog(namespaced: list[NamespacedTool]) -> str:
    """Render the full tool catalog (without ``<tools>`` wrapper).

    Namespaced sections come first in namespace-sorted order, then top-level
    tools. Tools within a section are sorted by attribute name for
    determinism.
    """
    grouped: dict[str | None, list[NamespacedTool]] = {}
    for nt in namespaced:
        grouped.setdefault(nt.namespace, []).append(nt)

    sections: list[str] = []

    for ns in sorted(n for n in grouped if n is not None):
        tools = sorted(grouped[ns], key=lambda t: t.attribute)
        rendered = [render_tool(t) for t in tools]
        names = ", ".join(rt.attribute for rt in rendered)
        entries = [_render_entry(rt) for rt in rendered]
        sections.append(
            _render_module_section(
                header=f"tools.{ns}",
                import_line=f"from tools.{ns} import {names}",
                entries=entries,
            )
        )

    top_level = sorted(grouped.get(None, []), key=lambda t: t.attribute)
    if top_level:
        rendered = [render_tool(t) for t in top_level]
        names = ", ".join(rt.attribute for rt in rendered)
        entries = [_render_entry(rt) for rt in rendered]
        sections.append(
            _render_module_section(
                header="tools",
                import_line=f"from tools import {names}",
                entries=entries,
            )
        )

    return "\n".join(sections)


__all__ = [
    "render_catalog",
]
