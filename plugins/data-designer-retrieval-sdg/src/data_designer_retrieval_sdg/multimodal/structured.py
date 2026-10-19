# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict retrieval JSON decoding within Data Designer's native generation loop."""

from __future__ import annotations

import json
import math
import re
from functools import cached_property
from typing import Any

from data_designer.engine.column_generators.generators.llm_completion import LLMStructuredCellGenerator
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n(?P<body>.*?)\r?\n```", re.DOTALL | re.IGNORECASE)


def _reject_constant(value: str) -> None:
    """Reject JavaScript constants that are not valid JSON numbers."""
    raise ValueError(f"Non-standard JSON constant: {value}")


def _finite_float(value: str) -> float:
    """Reject floating-point overflow instead of treating it as source evidence."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON number is outside the finite floating-point range")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous repeated object keys, including nested judge fields."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


class RetrievalStructuredResponseRecipe(StructuredResponseRecipe):
    """Accept one complete JSON object while retaining native schema validation."""

    def parse(self, response: str) -> dict:
        """Decode bare or singly fenced JSON without repairing its contents.

        Args:
            response: Complete model response, optionally enclosed in a JSON fence.

        Returns:
            The object validated by Data Designer against the original schema.

        Raises:
            ParserException: If decoding or schema validation fails. Data Designer
                retains ownership of bounded correction retries and row failures.
        """
        try:
            body = response.strip()
            fence = _JSON_FENCE.fullmatch(body)
            if fence is not None:
                body = fence.group("body")
            value = json.loads(
                body, parse_constant=_reject_constant, parse_float=_finite_float, object_pairs_hook=_unique_object
            )
            if not isinstance(value, dict):
                raise ValueError("Expected exactly one JSON object")
        except (ValueError, TypeError, AttributeError, RecursionError) as exc:
            raise ParserException(f"Invalid retrieval JSON response: {exc}", source=response) from exc
        # Format normalization only: no field pruning, query rewriting, type
        # coercion, value repair, or conversion of negative judgements to passes.
        return super().parse("```json\n" + json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n```")


class RetrievalStructuredCellGenerator(LLMStructuredCellGenerator):
    """Reuse native image handling, scheduling, traces, and bounded corrections."""

    @cached_property
    def response_recipe(self) -> RetrievalStructuredResponseRecipe:
        """Return the plugin parser with native schema validation and no pruning."""
        return RetrievalStructuredResponseRecipe(json_schema=self.config.output_format, pruning=False)
