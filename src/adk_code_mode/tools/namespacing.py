# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Group normalised tools into dotted Python namespaces.

``slack_send_message`` → ``tools.slack.send_message``. The model writes
``from tools.slack import send_message``; the dispatcher looks up the original
``BaseTool.name`` via the registry built here.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass

from google.adk.tools.base_toolset import BaseToolset

from adk_code_mode.tools.normaliser import ResolvedTool

_IDENT_STRIP = re.compile(r"[^a-zA-Z0-9_]")


class ToolSurfaceError(ValueError):
    """Raised when tools cannot be mapped to a safe/generated Python surface."""


class DuplicateToolNameError(ToolSurfaceError):
    """Raised when two resolved tools share the same raw ADK tool name."""


class NamespaceCollisionError(ToolSurfaceError):
    """Raised when two toolsets collapse onto the same generated namespace."""


class PythonNameCollisionError(ToolSurfaceError):
    """Raised when generated Python identifiers collide."""


@dataclass(frozen=True)
class NamespacedTool:
    """A resolved tool with a chosen dotted path."""

    resolved: ResolvedTool
    namespace: str | None
    """The module name the tool lives in, e.g. ``"slack"``. ``None`` for top-level."""
    attribute: str
    """The Python identifier used for the tool function inside its module."""

    @property
    def dotted_path(self) -> str:
        if self.namespace:
            return f"{self.namespace}.{self.attribute}"
        return self.attribute

    @property
    def tool_name(self) -> str:
        return self.resolved.tool.name


def _sanitise_identifier(raw: str, *, fallback: str = "tool") -> str:
    cleaned = _IDENT_STRIP.sub("_", raw).strip("_").lower()
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def _toolset_namespace(toolset: BaseToolset) -> str:
    """Pick a stable module name for a toolset."""
    name_attr = getattr(toolset, "name", None)
    if isinstance(name_attr, str) and name_attr.strip():
        return _sanitise_identifier(name_attr)
    type_name = type(toolset).__name__
    for suffix in ("Toolset", "ToolSet", "Tools"):
        if type_name.endswith(suffix):
            type_name = type_name[: -len(suffix)]
            break
    return _sanitise_identifier(type_name, fallback="toolset")


def build(tools: list[ResolvedTool]) -> list[NamespacedTool]:
    """Assign a dotted path to every tool and resolve all collisions.

    Stable / deterministic: tools from the same toolset share a namespace; bare
    tools live at the top level. Any ambiguity in the generated Python surface
    is rejected up front.
    """
    out: list[NamespacedTool] = []
    seen_attributes: set[tuple[str | None, str]] = set()
    seen_raw_names: dict[str, NamespacedTool] = {}
    namespace_sources: dict[str, BaseToolset] = {}
    bare_attributes: dict[str, NamespacedTool] = {}
    for resolved in tools:
        namespace = _toolset_namespace(resolved.toolset) if resolved.toolset else None
        if resolved.tool.name in seen_raw_names:
            other = seen_raw_names[resolved.tool.name]
            candidate = (
                f"{namespace}.{_sanitise_identifier(resolved.tool.name, fallback='tool')}"
                if namespace
                else _sanitise_identifier(resolved.tool.name, fallback="tool")
            )
            raise DuplicateToolNameError(
                f"Duplicate ADK tool name {resolved.tool.name!r} is not supported; this would "
                f"make host dispatch ambiguous between {other.dotted_path!r} (from "
                f"{_origin(other.resolved)}) and {candidate!r} (from {_origin(resolved)}). "
                "Rename one of the tools or apply distinct ``tool_name_prefix`` values."
            )
        if namespace is not None and resolved.toolset is not None:
            existing_source = namespace_sources.get(namespace)
            if existing_source is not None and existing_source is not resolved.toolset:
                raise NamespaceCollisionError(
                    f"Toolsets {type(existing_source).__name__!r} and "
                    f"{type(resolved.toolset).__name__!r} both map to generated namespace "
                    f"{namespace!r}. Override ``BaseToolset.name`` on one of them so the "
                    "generated import paths stay unique."
                )
            namespace_sources[namespace] = resolved.toolset
        attribute = _sanitise_identifier(resolved.tool.name, fallback="tool")
        if namespace is not None and namespace in bare_attributes:
            colliding_bare = bare_attributes[namespace]
            raise PythonNameCollisionError(
                f"Top-level tool {colliding_bare.tool_name!r} (Python name {namespace!r}) "
                f"collides with namespace {namespace!r} generated from "
                f"{type(resolved.toolset).__name__ if resolved.toolset else 'tool'}. "
                "Rename the bare tool or override the toolset's ``name`` so the surfaces don't share an identifier."
            )
        key = (namespace, attribute)
        if key in seen_attributes:
            dotted = f"{namespace}.{attribute}" if namespace else attribute
            raise PythonNameCollisionError(
                f"Generated Python identifier collision for {dotted!r}: ADK tool names "
                f"would sanitise to the same identifier. Rename one of the tools so the "
                "import surface is unique."
            )
        seen_attributes.add(key)
        nt = NamespacedTool(resolved=resolved, namespace=namespace, attribute=attribute)
        seen_raw_names[resolved.tool.name] = nt
        if namespace is None:
            bare_attributes[attribute] = nt
        out.append(nt)

    namespace_names = set(namespace_sources.keys())
    for bare, nt in bare_attributes.items():
        if bare in namespace_names:
            source = namespace_sources[bare]
            raise PythonNameCollisionError(
                f"Top-level tool {nt.tool_name!r} (Python name {bare!r}) collides with "
                f"namespace {bare!r} generated from {type(source).__name__}. "
                "Rename the bare tool or override the toolset's ``name``."
            )
    return out


def _origin(resolved: ResolvedTool) -> str:
    """Human-readable origin of a resolved tool, for error messages."""
    if resolved.toolset is not None:
        return f"toolset {type(resolved.toolset).__name__}"
    return "top-level tools"


class Registry:
    """Bidirectional lookup: dotted path ↔ ``BaseTool.name``."""

    def __init__(self, tools: list[NamespacedTool]) -> None:
        self._by_path: dict[str, NamespacedTool] = {t.dotted_path: t for t in tools}
        by_tool_name: dict[str, NamespacedTool] = {}
        for tool in tools:
            existing = by_tool_name.get(tool.tool_name)
            if existing is not None:
                raise DuplicateToolNameError(
                    "Duplicate ADK tool name "
                    f"{tool.tool_name!r} is not supported; use unique tool names."
                )
            by_tool_name[tool.tool_name] = tool
        self._by_tool_name = by_tool_name
        self._tools = list(tools)

    @property
    def tools(self) -> list[NamespacedTool]:
        return list(self._tools)

    def by_dotted_path(self, path: str) -> NamespacedTool | None:
        return self._by_path.get(path)

    def by_tool_name(self, tool_name: str) -> NamespacedTool | None:
        return self._by_tool_name.get(tool_name)

    def resolve_call(self, name: str) -> NamespacedTool:
        """Accept either the dotted path or the original ``BaseTool.name``."""
        hit = self._by_path.get(name) or self._by_tool_name.get(name)
        if hit is None:
            raise KeyError(f"no tool registered for {name!r}")
        return hit


__all__ = ["NamespacedTool", "Registry", "build"]
