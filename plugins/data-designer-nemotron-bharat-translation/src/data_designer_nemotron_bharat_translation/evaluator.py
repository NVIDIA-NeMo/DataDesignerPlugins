# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MQM evaluation logic for ``document-translation``.

Mirrors :class:`~data_designer_nemotron_bharat_translation.translator.Translator`:
:class:`Evaluator` renders one combined evaluation prompt -- ``text``,
``translation_result.translated_text``, ``source_language_description``,
and ``target_language_description`` all in a single user turn -- and parses
the model's JSON response into a :class:`TranslationEvaluation`, rather
than the generator maintaining its own copy of the render-parse logic.
Because it's a separate object with no dependency on the translate step,
it can score *any* text against a source (not only ``Translator``'s own
output) -- e.g. re-evaluating a document after an out-of-band, non-model
transformation such as ``script_remap.py``'s deterministic script
conversion (see ``DocumentTranslationConfig.target_language_script``),
which runs MQM once on the Bengali-script translation and again on the
Meetei Mayek remapped result.
"""

from __future__ import annotations

from typing import Literal

from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from jinja2 import Template
from pydantic import BaseModel, Field

# Union of the category vocabulary in GENERIC_EVALUATION_PROMPT's framework
# and expected-output sections (they don't list identical sets).
ErrorCategory = Literal["Faithfulness", "Fluency", "Terminology", "Style", "Format Preservation", "Script Compliance"]
Severity = Literal["Minor", "Major", "Critical"]


class ErrorSpan(BaseModel):
    """A single flagged error span in a translation evaluation."""

    text_span: str = ""
    error_category: ErrorCategory
    severity: Severity
    explanation: str = ""


class TranslationEvaluation(BaseModel):
    """Structured response from the MQM evaluation step."""

    analysis: str = ""
    error_analysis: list[ErrorSpan] = Field(default_factory=list)
    justification: str = ""
    overall_score: int = Field(description="Overall quality score from 0 to 100.")


class Evaluator:
    """Renders a single combined MQM evaluation prompt and parses the model's JSON response."""

    def __init__(self, template: str) -> None:
        self._template = Template(template)
        self._recipe = PydanticResponseRecipe(data_type=TranslationEvaluation)

    def render(
        self, text: str, translation: str, source_language_description: str, target_language_description: str
    ) -> str:
        user_prompt = self._template.render(
            source_language_description=source_language_description,
            target_language_description=target_language_description,
            text=text,
            translation_result={"translated_text": translation},
        )
        return self._recipe.apply_recipe_to_user_prompt(user_prompt)

    def evaluate(
        self,
        model,
        text: str,
        translation: str,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
    ) -> TranslationEvaluation:
        prompt = self.render(text, translation, source_language_description, target_language_description)
        result, _trace = model.generate(prompt, parser=self._recipe.parse, max_correction_steps=max_correction_steps)
        return result

    async def aevaluate(
        self,
        model,
        text: str,
        translation: str,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
    ) -> TranslationEvaluation:
        prompt = self.render(text, translation, source_language_description, target_language_description)
        result, _trace = await model.agenerate(
            prompt, parser=self._recipe.parse, max_correction_steps=max_correction_steps
        )
        return result
