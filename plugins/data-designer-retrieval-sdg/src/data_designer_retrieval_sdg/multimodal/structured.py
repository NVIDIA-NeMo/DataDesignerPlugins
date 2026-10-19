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
from data_designer.engine.models.errors import ModelGenerationValidationFailureError
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

from data_designer_retrieval_sdg.multimodal.models import FailedGeneration

_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n(?P<body>.*?)\r?\n```", re.DOTALL | re.IGNORECASE)
FAILURE_KEY = "_retrieval_sdg_failure"


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

    def record_result(self, data: dict) -> dict:
        """Persist uniform JSON strings in skip mode to avoid mixed Arrow structs.

        Args:
            data: Native row containing the validated response or failure diagnostic.

        Returns:
            The row with a JSON-string cell in skip mode, unchanged in error mode.
        """
        if self.config.failed_record_policy == "skip":
            data[self.config.name] = json.dumps(data[self.config.name], ensure_ascii=False, allow_nan=False)
        return data

    def record_failure(self, data: dict, error: ModelGenerationValidationFailureError) -> dict:
        """Retain a validation diagnostic under the original native row identity.

        Args:
            data: Original row, including successful sibling columns.
            error: Validation failure after Data Designer's native retry budget.

        Returns:
            The row with a tagged diagnostic in the failed column.

        Raises:
            ModelGenerationValidationFailureError: Under the default error policy.
        """
        if self.config.failed_record_policy != "skip":
            raise error
        cause, response, seen = error, None, set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if isinstance(cause, ParserException) and cause.source is not None:
                response = cause.source
            cause = cause.__cause__ or cause.__context__
        failure = FailedGeneration(
            kind="model_validation_exhausted",
            detail=error.detail or str(error),
            max_model_attempts=(self.max_conversation_correction_steps + 1) * (self.max_conversation_restarts + 1),
            last_invalid_response=response,
        )
        data[self.config.name] = {FAILURE_KEY: failure.model_dump()}
        return self.record_result(data)

    def generate(self, data: dict) -> dict:
        """Apply the opt-in policy after native validation retries are exhausted.

        Args:
            data: Native input row.

        Returns:
            The row with a validated response or an explicitly enabled diagnostic.

        Raises:
            ModelGenerationValidationFailureError: Under the default error policy.
        """
        try:
            return self.record_result(super().generate(data))
        except ModelGenerationValidationFailureError as error:
            return self.record_failure(data, error)

    async def agenerate(self, data: dict) -> dict:
        """Apply the same policy while retaining native asynchronous generation.

        Args:
            data: Native input row.

        Returns:
            The row with a validated response or an explicitly enabled diagnostic.

        Raises:
            ModelGenerationValidationFailureError: Under the default error policy.
        """
        try:
            return self.record_result(await super().agenerate(data))
        except ModelGenerationValidationFailureError as error:
            return self.record_failure(data, error)
