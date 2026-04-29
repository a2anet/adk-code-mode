# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Generate Python stub modules for ADK tools.

Each stub is a real ``.py`` file the sandbox writes into ``/tools/``. The
function body is a single delegation into the host via the RPC client, so the
generated file is small and robust. Type hints come from the tool's JSON
Schema (``BaseTool._get_declaration().parameters_json_schema``).

``render_tool`` returns a structured ``RenderedTool``; ``render_tool_source``
turns one into the on-disk ``.py`` stub. The catalog renderer (in
``adk_code_mode.tools.catalog``) consumes the same ``RenderedTool`` to produce
the ``.pyi``-style block injected into the model's system prompt.
"""

from __future__ import annotations

import keyword
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Literal

from adk_code_mode.tools.namespacing import NamespacedTool, PythonNameCollisionError

_PRIMITIVES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
}

Target = Literal["stub", "catalog"]


def _schema_dict_from_declaration(declaration: Any) -> dict[str, Any]:
    """Return a JSON-Schema-shaped dict for a tool declaration.

    ADK tools expose their schema one of two ways: ``parameters_json_schema``
    (RestApiTool, MCP tools, anything built from a real JSON Schema) or
    ``parameters`` (FunctionTool, which builds a Gemini ``types.Schema`` from
    the function signature). We normalise both to a plain dict.
    """
    if declaration is None:
        return {}
    pjs = getattr(declaration, "parameters_json_schema", None)
    if pjs:
        return dict(pjs)
    params = getattr(declaration, "parameters", None)
    if params is None:
        return {}
    if hasattr(params, "model_dump"):
        dumped = params.model_dump(mode="json", exclude_none=True)
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(params, dict):
        return dict(params)
    return {}


def _schema_to_type(schema: Any) -> str:
    """Convert a JSON-Schema fragment to a Python type expression (PEP 604).

    Best-effort. Anything we can't resolve cleanly falls back to ``Any``.
    """
    if not isinstance(schema, dict):
        return "Any"

    # Compositions.
    if "anyOf" in schema or "oneOf" in schema:
        variants = schema.get("anyOf") or schema.get("oneOf") or []
        parts = [_schema_to_type(v) for v in variants]
        parts = list(dict.fromkeys(parts))
        if not parts:
            return "Any"
        if len(parts) == 1:
            return parts[0]
        return " | ".join(parts)
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for part in schema["allOf"]:
            if isinstance(part, dict):
                merged.update(part)
        if merged:
            return _schema_to_type(merged)
        return "Any"

    if "enum" in schema and schema["enum"]:
        lits: list[str] = []
        for value in schema["enum"]:
            if isinstance(value, str):
                lits.append(repr(value))
            elif isinstance(value, bool):
                lits.append("True" if value else "False")
            elif isinstance(value, (int, float)):
                lits.append(repr(value))
            elif value is None:
                lits.append("None")
            else:
                return "Any"
        return "Literal[" + ", ".join(lits) + "]"

    ty = schema.get("type")
    if isinstance(ty, str):
        ty = ty.lower()
    nullable = bool(schema.get("nullable"))
    if isinstance(ty, list):
        parts = [
            _schema_to_type(
                {
                    "type": t.lower() if isinstance(t, str) else t,
                    **{k: v for k, v in schema.items() if k != "type"},
                }
            )
            for t in ty
        ]
        parts = list(dict.fromkeys(parts))
        result = parts[0] if len(parts) == 1 else " | ".join(parts)
    elif ty == "array":
        items = schema.get("items")
        prefix_items = schema.get("prefixItems")
        if prefix_items:
            inner = ", ".join(_schema_to_type(p) for p in prefix_items)
            result = f"tuple[{inner}]"
        elif isinstance(items, dict):
            result = f"list[{_schema_to_type(items)}]"
        else:
            result = "list[Any]"
    elif ty == "object":
        result = "dict[str, Any]"
    elif isinstance(ty, str) and ty in _PRIMITIVES:
        result = _PRIMITIVES[ty]
    else:
        result = "Any"

    if nullable and result != "Any":
        result = f"{result} | None"
    return result


def _python_default(schema: dict[str, Any]) -> str | None:
    """Return a Python source expression for the schema's default, or None."""
    if "default" not in schema:
        return None
    try:
        return repr(schema["default"])
    except Exception:
        return None


def _format_docstring(*, description: str, param_docs: list[tuple[str, str, str | None]]) -> str:
    lines: list[str] = []
    summary = (description or "").strip() or "Call the tool."
    lines.extend(textwrap.wrap(summary, width=88) or [summary])
    if param_docs:
        lines.append("")
        lines.append("Args:")
        for name, desc, default_repr in param_docs:
            desc = (desc or "").strip()
            first, *rest = (desc or "").splitlines() or [""]
            suffix = f" (default: {default_repr})" if default_repr is not None else ""
            head = first + suffix
            lines.append(f"    {name}: {head}" if head else f"    {name}:")
            for extra in rest:
                lines.append(f"        {extra}")
    body = "\n    ".join(lines)
    return f'"""{body}\n    """'


def _sanitise_param(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw).strip("_")
    if not cleaned:
        cleaned = "arg"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


@dataclass(frozen=True)
class RenderedParam:
    """One parameter as it appears in a generated tool function."""

    py_name: str
    raw_name: str
    type_expr: str
    is_required: bool
    schema_default: str | None
    """``repr()`` of the schema's default value if present; otherwise ``None``."""


@dataclass(frozen=True)
class RenderedTool:
    """Structured tool ready for stub or catalog rendering.

    Argument forwarding rules (Option A):

    - Required, no schema default: ``name: T``; always forwarded.
    - Required, schema default: ``name: T = <default>``; always forwarded.
    - Optional (with or without schema default): ``name: T | None = _MISSING``
      in the on-disk stub (forwarded only if the caller passed a value, so
      the host-side tool's own default behaviour applies on omission); the
      catalog renders this as ``name: T | None = ...`` since the sentinel is
      an implementation detail. Schema default (when present) is surfaced in
      the docstring rather than baked into the signature, since Python erases
      "argument was not passed" once a real default lands in the parameter
      slot.
    """

    attribute: str
    dotted_path: str
    namespace: str | None
    docstring: str
    params: tuple[RenderedParam, ...]

    def signature_for(self, target: Target) -> str:
        """Return the ``def name(...) -> Any:`` line for the requested target."""
        return f"def {self.attribute}({_render_params(self.params, target=target)}) -> Any:"

    @property
    def needs_missing_sentinel(self) -> bool:
        return any(not p.is_required for p in self.params)


def _render_params(params: tuple[RenderedParam, ...], *, target: Target) -> str:
    """Render the keyword-only parameter list."""
    if not params:
        return ""
    pieces: list[str] = ["*"]
    optional_default = "_MISSING" if target == "stub" else "..."
    for p in params:
        if p.is_required:
            if p.schema_default is None:
                pieces.append(f"{p.py_name}: {p.type_expr}")
            else:
                pieces.append(f"{p.py_name}: {p.type_expr} = {p.schema_default}")
        else:
            pieces.append(f"{p.py_name}: {_ensure_optional_type(p.type_expr)} = {optional_default}")
    return ", ".join(pieces)


def _ensure_optional_type(type_expr: str) -> str:
    """Return ``type_expr`` augmented with ``| None`` if not already present.

    Avoids ``str | None | None`` for params whose schema was already
    ``nullable: true`` (which ``_schema_to_type`` rendered as ``str | None``).
    """
    parts = [p.strip() for p in type_expr.split("|")]
    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            deduped.append(part)
    if "None" not in seen:
        deduped.append("None")
    return " | ".join(deduped)


@dataclass(frozen=True)
class StubFile:
    path: str
    source: str


def render_tool(nt: NamespacedTool) -> RenderedTool:
    """Build a ``RenderedTool`` for a normalised tool.

    Captures everything stub and catalog rendering need: signature parts,
    formatted docstring, dotted path, namespace.
    """
    declaration = nt.resolved.tool._get_declaration()
    schema: dict[str, Any] = _schema_dict_from_declaration(declaration)

    props: dict[str, Any] = dict(schema.get("properties") or {})
    required = set(schema.get("required") or [])
    ordered = sorted(props.items(), key=lambda kv: (kv[0] not in required, kv[0]))

    params: list[RenderedParam] = []
    param_docs: list[tuple[str, str, str | None]] = []

    seen_param_names: dict[str, str] = {}

    for raw_name, subschema in ordered:
        py_name = _sanitise_param(raw_name)
        existing_raw_name = seen_param_names.get(py_name)
        if existing_raw_name is not None:
            raise PythonNameCollisionError(
                f"Tool {nt.tool_name!r} has parameter names {existing_raw_name!r} and "
                f"{raw_name!r} that both map to Python parameter {py_name!r}. Rename one "
                "of the tool parameters so the generated stub signature is unambiguous."
            )
        seen_param_names[py_name] = raw_name
        subschema = subschema if isinstance(subschema, dict) else {}
        type_expr = _schema_to_type(subschema)
        default = _python_default(subschema)
        is_required = raw_name in required
        params.append(
            RenderedParam(
                py_name=py_name,
                raw_name=raw_name,
                type_expr=type_expr,
                is_required=is_required,
                schema_default=default,
            )
        )
        doc_default = default if (not is_required and default is not None) else None
        param_docs.append((py_name, str(subschema.get("description", "")), doc_default))

    description = ""
    if declaration is not None and declaration.description:
        description = declaration.description
    elif nt.resolved.tool.description:
        description = nt.resolved.tool.description
    docstring = _format_docstring(description=description, param_docs=param_docs)

    return RenderedTool(
        attribute=nt.attribute,
        dotted_path=nt.dotted_path,
        namespace=nt.namespace,
        docstring=docstring,
        params=tuple(params),
    )


def render_tool_source(rt: RenderedTool) -> str:
    """Render the on-disk Python stub source for a tool."""
    sig = rt.signature_for("stub")
    lines: list[str] = [
        "# Generated by adk-code-mode. Do not edit.",
        "from __future__ import annotations",
        "",
        "from typing import Any, Literal",
        "",
        "from adk_code_mode_sandbox._rpc_client import call as _call",
        "",
        "",
    ]
    if rt.needs_missing_sentinel:
        lines.extend(["_MISSING = object()", "", ""])
    lines.append(sig)
    lines.append(f"    {rt.docstring}")

    unconditional = [p for p in rt.params if p.is_required]
    optional = [p for p in rt.params if not p.is_required]
    if unconditional:
        lines.append("    _args: dict[str, Any] = {")
        for p in unconditional:
            lines.append(f"        {p.raw_name!r}: {p.py_name},")
        lines.append("    }")
    else:
        lines.append("    _args: dict[str, Any] = {}")
    for p in optional:
        lines.append(f"    if {p.py_name} is not _MISSING:")
        lines.append(f"        _args[{p.raw_name!r}] = {p.py_name}")
    lines.append(f"    return _call({rt.dotted_path!r}, _args)")
    lines.append("")
    return "\n".join(lines)


def render_namespace_init(tools: list[NamespacedTool]) -> str:
    """Render ``__init__.py`` for a namespace package.

    Re-exports each tool in the namespace so the model can write
    ``from tools.<namespace> import <name>``.
    """
    sorted_tools = sorted(tools, key=lambda t: t.attribute)
    imports = "\n".join(f"from .{t.attribute} import {t.attribute}" for t in sorted_tools)
    all_list = ", ".join(repr(t.attribute) for t in sorted_tools)
    header = "# Generated by adk-code-mode. Do not edit.\n"
    return f"{header}from __future__ import annotations\n\n{imports}\n\n__all__ = [{all_list}]\n"


_ROOT_MARKER = "# Generated by adk-code-mode. Do not edit.\n"


def render_root_init(top_level: list[NamespacedTool]) -> str:
    """Render the generated ``tools`` package ``__init__.py``.

    Re-exports any top-level (non-namespaced) tools so the model can write
    ``from tools import <name>``. Namespaced tools are *not* re-exported
    here — the model uses ``from tools.<namespace> import <name>``, which
    only loads that namespace's stubs.
    """
    if not top_level:
        return _ROOT_MARKER
    sorted_tools = sorted(top_level, key=lambda t: t.attribute)
    imports = "\n".join(f"from .{t.attribute} import {t.attribute}" for t in sorted_tools)
    all_list = ", ".join(repr(t.attribute) for t in sorted_tools)
    return (
        f"{_ROOT_MARKER}from __future__ import annotations\n\n{imports}\n\n__all__ = [{all_list}]\n"
    )


def render_tree(namespaced: list[NamespacedTool]) -> list[StubFile]:
    """Render the full stub tree for a list of tools.

    Files are returned with POSIX paths rooted at the generated ``tools``
    package directory. The runtime mounts that package directory at ``/tools``
    and adds its parent to ``sys.path`` so ``import tools`` resolves directly
    to ``/tools/__init__.py``. The root ``__init__.py`` re-exports top-level tools (so
    ``from tools import <name>`` works) but does **not** re-export
    namespaced tools — the model receives a tool catalog up-front and writes
    ``from tools.<namespace> import <name>`` directly, so eagerly re-exporting
    every stub from the root would be wasteful at large surfaces.
    """
    per_tool = sorted(namespaced, key=lambda t: t.dotted_path)
    files: list[StubFile] = []

    grouped: dict[str | None, list[NamespacedTool]] = {}
    for nt in per_tool:
        grouped.setdefault(nt.namespace, []).append(nt)

    for nt in per_tool:
        ns = nt.namespace
        sub = "" if ns is None else f"{ns}/"
        rendered = render_tool(nt)
        files.append(StubFile(path=f"{sub}{nt.attribute}.py", source=render_tool_source(rendered)))

    for ns, tools in grouped.items():
        if ns is None:
            continue
        files.append(
            StubFile(
                path=f"{ns}/__init__.py",
                source=render_namespace_init(tools),
            )
        )

    top_level = grouped.get(None, [])
    files.append(StubFile(path="__init__.py", source=render_root_init(top_level)))
    return sorted(files, key=lambda f: f.path)


__all__ = [
    "RenderedParam",
    "RenderedTool",
    "StubFile",
    "render_namespace_init",
    "render_root_init",
    "render_tool",
    "render_tool_source",
    "render_tree",
]
