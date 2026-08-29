# SPDX-FileCopyrightText: 2025-present A2A Net <hello@a2anet.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Coverage of the JSON Schema keyword set, vocabulary by vocabulary.

Grounded in JSON Schema 2020-12 (Core, Applicator, Validation, Meta-Data,
Format-Annotation and Content vocabularies) plus the OpenAPI 3.0 dialect's
`nullable` and singular `example`, since OpenAPI documents are what tools are
built from in practice.

The contract each test asserts is the same: information a caller needs in order
to construct a valid argument must reach the generated stub, because the stub
and its docstring are the whole of what the model sees. A keyword that is
deliberately not surfaced is asserted as such in `TestDeliberateOmissions`, so
the decision is visible rather than looking like an oversight.
"""

from __future__ import annotations

from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types

from adk_code_mode.tools import namespacing, stubs
from adk_code_mode.tools.normaliser import ResolvedTool


class _SchemaTool(BaseTool):
    def __init__(self, name: str, *, schema: dict[str, object]) -> None:
        super().__init__(name=name, description="Tool.")
        self._schema = schema

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._schema,
        )


def _render(properties: dict[str, object], *, required: list[str] | None = None) -> str:
    """Render a one-tool stub whose body is the given properties."""
    schema: dict[str, object] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    tool = _SchemaTool("op", schema=schema)
    resolved = [ResolvedTool(tool=tool, toolset=None)]
    return stubs.render_tool_source(stubs.render_tool(namespacing.build(resolved)[0]))


class TestValidationVocabularyAnyInstance:
    """§6.1 — `type`, `enum`, `const`."""

    def test_type_maps_to_python_primitives(self) -> None:
        source = _render(
            {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
            },
            required=["s", "i", "n", "b"],
        )
        assert "s: str" in source
        assert "i: int" in source
        assert "n: float" in source
        assert "b: bool" in source

    def test_union_type_array_becomes_pep604_union(self) -> None:
        source = _render({"x": {"type": ["string", "integer"]}}, required=["x"])
        assert "x: str | int" in source

    def test_enum_becomes_literal_in_the_signature(self) -> None:
        source = _render(
            {"colour": {"type": "string", "enum": ["red", "green"]}}, required=["colour"]
        )
        assert "colour: Literal['red', 'green']" in source

    def test_const_reaches_the_docstring(self) -> None:
        source = _render({"kind": {"type": "string", "const": "rule"}})
        assert "const=rule" in source


class TestValidationVocabularyNumeric:
    """§6.2 — `multipleOf`, `maximum`, `exclusiveMaximum`, `minimum`, `exclusiveMinimum`."""

    def test_all_numeric_bounds_reach_the_docstring(self) -> None:
        source = _render(
            {
                "ratio": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "exclusiveMinimum": -1,
                    "exclusiveMaximum": 2,
                    "multipleOf": 0.5,
                }
            }
        )
        for expected in (
            "minimum=0",
            "maximum=1",
            "exclusiveMinimum=-1",
            "exclusiveMaximum=2",
            "multipleOf=0.5",
        ):
            assert expected in source

    def test_bounds_on_array_items_are_not_hidden_by_the_list_type(self) -> None:
        """`list[int]` erases the item schema, so the bound is folded upward."""
        source = _render(
            {
                "daysOfWeek": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 7},
                }
            }
        )
        assert "daysOfWeek: list[int]" in source
        assert "items: minimum=1, maximum=7" in source


class TestValidationVocabularyString:
    """§6.3 — `maxLength`, `minLength`, `pattern`."""

    def test_string_constraints_reach_the_docstring(self) -> None:
        source = _render(
            {"code": {"type": "string", "minLength": 3, "maxLength": 3, "pattern": "^[A-Z]{3}$"}}
        )
        assert "minLength=3" in source
        assert "maxLength=3" in source
        assert "pattern=^[A-Z]{3}$" in source


class TestValidationVocabularyArray:
    """§6.4 — `maxItems`, `minItems`, `uniqueItems`, `maxContains`, `minContains`."""

    def test_array_cardinality_reaches_the_docstring(self) -> None:
        source = _render(
            {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                    "uniqueItems": True,
                    "minContains": 1,
                    "maxContains": 3,
                }
            }
        )
        for expected in (
            "minItems=1",
            "maxItems=10",
            "uniqueItems=true",
            "minContains=1",
            "maxContains=3",
        ):
            assert expected in source


class TestValidationVocabularyObject:
    """§6.5 — `maxProperties`, `minProperties`, `required`, `dependentRequired`."""

    def test_property_counts_reach_the_docstring(self) -> None:
        source = _render({"meta": {"type": "object", "minProperties": 1, "maxProperties": 5}})
        assert "minProperties=1" in source
        assert "maxProperties=5" in source

    def test_required_marks_nested_fields(self) -> None:
        source = _render(
            {
                "window": {
                    "type": "object",
                    "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                    "required": ["from"],
                }
            }
        )
        assert "from: str (required)" in source
        assert "to: str" in source

    def test_required_at_the_top_level_drops_the_optional_sentinel(self) -> None:
        source = _render({"a": {"type": "string"}, "b": {"type": "string"}}, required=["a"])
        assert "a: str," in source
        assert "b: str | None = _MISSING" in source

    def test_dependent_required_reaches_the_docstring(self) -> None:
        source = _render(
            {
                "payment": {
                    "type": "object",
                    "properties": {"card": {"type": "string"}, "cvv": {"type": "string"}},
                    "dependentRequired": {"card": ["cvv"]},
                }
            }
        )
        assert "card requires cvv" in source


class TestApplicatorVocabulary:
    """§10 — `properties`, `items`, `prefixItems`, `additionalProperties`,
    `patternProperties`, `allOf`, `anyOf`, `oneOf`."""

    def test_nested_properties_are_expanded_recursively(self) -> None:
        source = _render(
            {
                "conditions": {
                    "type": "object",
                    "properties": {
                        "timeRange": {
                            "type": "object",
                            "properties": {"from": {"type": "string"}},
                        }
                    },
                }
            }
        )
        assert "timeRange: dict[str, Any]" in source
        assert "from: str" in source

    def test_prefix_items_becomes_a_tuple(self) -> None:
        source = _render(
            {"pair": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]}},
            required=["pair"],
        )
        assert "pair: tuple[str, int]" in source

    def test_additional_properties_schema_types_an_open_map(self) -> None:
        source = _render(
            {"labels": {"type": "object", "additionalProperties": {"type": "string"}}},
            required=["labels"],
        )
        assert "labels: dict[str, str]" in source

    def test_additional_properties_false_is_stated(self) -> None:
        source = _render(
            {
                "conditions": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": False,
                }
            }
        )
        assert "no other keys" in source

    def test_pattern_properties_keys_are_stated(self) -> None:
        source = _render({"env": {"type": "object", "patternProperties": {"^[A-Z_]+$": {}}}})
        assert "key patterns: ^[A-Z_]+$" in source

    def test_allof_wrapped_ref_keeps_description_and_properties(self) -> None:
        """OpenAPI 3.0 spells a nullable `$ref` this way; the wrapper must not
        mask the target."""
        source = _render(
            {
                "capacity": {
                    "allOf": [
                        {
                            "type": "object",
                            "description": "Min/max capacity range.",
                            "properties": {"min": {"type": "integer"}},
                        }
                    ],
                    "nullable": True,
                }
            }
        )
        assert "Min/max capacity range." in source
        assert "min: int" in source

    def test_anyof_becomes_a_union_and_a_lone_branch_is_folded(self) -> None:
        source = _render(
            {
                "many": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "one": {"anyOf": [{"type": "string", "description": "Only branch."}]},
            },
            required=["many", "one"],
        )
        assert "many: str | int" in source
        assert "one: str" in source
        assert "Only branch." in source
        # Primitive unions belong in the type expression, not a second listing.
        assert "-- or --" not in source

    def test_oneof_object_branches_expand_their_properties(self) -> None:
        """A discriminator union has no `properties` at the `oneOf` itself.

        Folding only the single-branch case leaves `list[dict[str, Any]]` with
        an empty docstring — the shape of `create_agent(tools=...)`.
        """
        source = _render(
            {
                "tools": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["type", "specUrl"],
                                "properties": {
                                    "type": {"type": "string", "enum": ["openapi"]},
                                    "specUrl": {"type": "string"},
                                    "operationOverrides": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["sourceId", "enabled"],
                                            "properties": {
                                                "sourceId": {"type": "string"},
                                                "enabled": {"type": "boolean"},
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["type", "url"],
                                "properties": {
                                    "type": {"type": "string", "enum": ["mcp"]},
                                    "url": {"type": "string"},
                                    "toolOverrides": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["sourceId", "enabled"],
                                            "properties": {
                                                "sourceId": {"type": "string"},
                                                "enabled": {"type": "boolean"},
                                            },
                                        },
                                    },
                                },
                            },
                        ]
                    },
                }
            },
            required=["tools"],
        )
        assert "tools: list[dict[str, Any]]" in source
        assert "(type=openapi) (no other keys)" in source
        assert "specUrl: str (required)" in source
        assert "operationOverrides: list[dict[str, Any]]" in source
        assert "-- or --" in source
        assert "(type=mcp) (no other keys)" in source
        assert "url: str (required)" in source
        assert "toolOverrides: list[dict[str, Any]]" in source
        assert "sourceId: str (required)" in source
        assert "enabled: bool (required)" in source

    def test_oneof_uses_discriminator_property_for_the_branch_tag(self) -> None:
        source = _render(
            {
                "tool": {
                    "discriminator": {"propertyName": "kind"},
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"const": "http"},
                                "path": {"type": "string"},
                            },
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"const": "stdio"},
                                "command": {"type": "string"},
                            },
                        },
                    ],
                }
            }
        )
        assert "(kind=http)" in source
        assert "(kind=stdio)" in source
        assert "path: str" in source
        assert "command: str" in source


class TestMetaDataVocabulary:
    """§9 — `title`, `description`, `default`, `deprecated`, `readOnly`,
    `writeOnly`, `examples`."""

    def test_description_reaches_the_docstring(self) -> None:
        source = _render({"label": {"type": "string", "description": "Human label."}})
        assert "label: Human label." in source

    def test_title_is_used_when_there_is_no_description(self) -> None:
        source = _render({"label": {"type": "string", "title": "Human label"}})
        assert "label: Human label" in source

    def test_default_is_documented_rather_than_baked_into_the_signature(self) -> None:
        source = _render({"expand": {"type": "boolean", "default": False}})
        assert "(default: False)" in source
        assert "expand: bool = False" not in source

    def test_nested_default_reaches_the_docstring(self) -> None:
        source = _render(
            {
                "opts": {
                    "type": "object",
                    "properties": {"expand": {"type": "boolean", "default": False}},
                }
            }
        )
        assert "default=false" in source

    def test_deprecated_read_only_and_write_only_are_flagged(self) -> None:
        source = _render(
            {
                "old": {"type": "string", "deprecated": True},
                "computed": {"type": "string", "readOnly": True},
                "secret": {"type": "string", "writeOnly": True},
            }
        )
        assert "deprecated" in source
        assert "read-only" in source
        assert "write-only" in source

    def test_examples_plural_and_singular_both_reach_the_docstring(self) -> None:
        source = _render(
            {
                "start": {"type": "string", "examples": ["18:00", "19:30"]},
                "end": {"type": "string", "example": "23:59"},
            }
        )
        assert "examples=18:00, 19:30" in source
        assert "example=23:59" in source


class TestFormatAndContentVocabularies:
    """§7 `format`, §8 `contentEncoding` / `contentMediaType`."""

    def test_format_reaches_the_docstring(self) -> None:
        source = _render({"day": {"type": "string", "format": "date"}})
        assert "format=date" in source

    def test_content_keywords_reach_the_docstring(self) -> None:
        source = _render(
            {
                "blob": {
                    "type": "string",
                    "contentEncoding": "base64",
                    "contentMediaType": "image/png",
                }
            }
        )
        assert "contentEncoding=base64" in source
        assert "contentMediaType=image/png" in source


class TestOpenApiDialect:
    """OpenAPI 3.0's departures from JSON Schema proper."""

    def test_nullable_widens_the_type(self) -> None:
        source = _render({"note": {"type": "string", "nullable": True}}, required=["note"])
        assert "note: str | None" in source

    def test_gemini_schema_casing_is_normalised(self) -> None:
        """With ADK's `JSON_SCHEMA_FOR_FUNC_DECL` off, schemas arrive via a
        Gemini `types.Schema`: upper-cased types, snake_cased keywords."""
        source = _render(
            {
                "label": {"type": "STRING", "max_length": 255},
                "tags": {"type": "ARRAY", "min_items": 1, "items": {"type": "STRING"}},
            }
        )
        assert "label: str | None" in source
        assert "maxLength=255" in source
        assert "tags: list[str] | None" in source
        assert "minItems=1" in source


class TestDeliberateOmissions:
    """Keywords intentionally not rendered.

    They express conditional or document-level structure that reads as noise in
    a one-line summary, and none appear in the OpenAPI documents this renders in
    practice. Listed here so the choice stays visible.
    """

    def test_conditional_and_document_keywords_are_not_rendered(self) -> None:
        source = _render(
            {
                "x": {
                    "type": "string",
                    "$comment": "internal note",
                    "$anchor": "thing",
                    "not": {"const": "banned"},
                    "if": {"const": "a"},
                    "then": {"const": "b"},
                    "propertyNames": {"pattern": "^x"},
                    "unevaluatedProperties": False,
                }
            }
        )
        for absent in ("internal note", "$anchor", "unevaluatedProperties", "propertyNames"):
            assert absent not in source

    def test_an_unrenderable_schema_still_produces_a_callable_stub(self) -> None:
        source = _render({"x": {"not": {"type": "string"}}})
        assert "def op(" in source
        assert "x: Any | None = _MISSING" in source


class TestKeywordSetIsFullyAccountedFor:
    """Every keyword in the 2020-12 vocabularies is a deliberate decision.

    The keyword list is read from the meta-schemas rather than hard-coded here,
    so a keyword this renderer has never considered cannot slip through as an
    oversight — it fails this test until it is placed in one bucket or the
    other.
    """

    # Keywords that reach the generated stub, in its type expression or its
    # docstring. Every Validation and Meta-Data keyword is here: between them
    # they are what describes the shape of a value a caller has to construct.
    HANDLED = {
        # Applicator
        "additionalProperties",
        "allOf",
        "anyOf",
        "items",
        "oneOf",
        "patternProperties",
        "prefixItems",
        "properties",
        # Validation
        "const",
        "dependentRequired",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "pattern",
        "required",
        "type",
        "uniqueItems",
        # Meta-Data
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
        # Format-Annotation and Content
        "format",
        "contentEncoding",
        "contentMediaType",
    }

    # Core keywords describe the document, not the value — and ADK resolves
    # `$ref` before a declaration reaches us. The applicator keywords here
    # express conditional structure with no useful one-line rendering, and
    # `unevaluated*`/`contentSchema` only make sense alongside those.
    OMITTED = {
        "$anchor",
        "$comment",
        "$defs",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$ref",
        "$schema",
        "$vocabulary",
        "contains",
        "dependentSchemas",
        "else",
        "if",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contentSchema",
    }

    @staticmethod
    def _spec_keywords() -> set[str]:
        from jsonschema_specifications import REGISTRY

        base = "https://json-schema.org/draft/2020-12/meta/"
        vocabularies = (
            "core",
            "applicator",
            "validation",
            "meta-data",
            "format-annotation",
            "content",
            "unevaluated",
        )
        keywords: set[str] = set()
        for vocabulary in vocabularies:
            contents = REGISTRY.get_or_retrieve(base + vocabulary).value.contents
            keywords.update(contents.get("properties") or {})
        return keywords

    def test_every_spec_keyword_is_either_handled_or_knowingly_omitted(self) -> None:
        spec = self._spec_keywords()
        classified = self.HANDLED | self.OMITTED
        assert spec - classified == set(), "unclassified JSON Schema keywords"
        assert classified - spec == set(), "classified keywords that are not in the spec"

    def test_every_validation_and_metadata_keyword_is_handled(self) -> None:
        """The two vocabularies that describe a value, as opposed to the
        document or a conditional, must be fully covered."""
        from jsonschema_specifications import REGISTRY

        base = "https://json-schema.org/draft/2020-12/meta/"
        for vocabulary in ("validation", "meta-data"):
            contents = REGISTRY.get_or_retrieve(base + vocabulary).value.contents
            keywords = set(contents.get("properties") or {})
            assert keywords <= self.HANDLED, (
                f"{vocabulary} keywords missing: {keywords - self.HANDLED}"
            )
