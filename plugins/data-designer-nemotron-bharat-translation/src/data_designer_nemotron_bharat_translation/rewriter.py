# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared self-correction rewrite logic for ``document-translation``.

Mirrors :class:`~data_designer_nemotron_bharat_translation.translator.Translator`
and :class:`~data_designer_nemotron_bharat_translation.evaluator.Evaluator`:
:class:`Rewriter` renders one combined rewrite prompt -- the original
source, the current translation, and its flagged
:class:`~data_designer_nemotron_bharat_translation.evaluator.TranslationEvaluation`
errors, all in a single user turn -- and treats the model's plain-text
response as the corrected translation, rather than the generator
maintaining its own copy of the render-parse logic.
"""

from __future__ import annotations

from data_designer.engine.models.recipes.response_recipes import TextResponseRecipe
from jinja2 import Template

from data_designer_nemotron_bharat_translation.evaluator import TranslationEvaluation


class Rewriter:
    """Renders a single combined self-correction prompt and parses the model's plain-text response."""

    def __init__(self, template: str) -> None:
        self._template = Template(template)
        self._recipe = TextResponseRecipe()

    def render(
        self,
        text_source: str,
        translation: str,
        evaluation: TranslationEvaluation,
        source_language_description: str,
        target_language_description: str,
    ) -> str:
        user_prompt = self._template.render(
            source_language_description=source_language_description,
            target_language_description=target_language_description,
            text_source=text_source,
            text=translation,
            overall_score=evaluation.overall_score,
            analysis=evaluation.analysis,
            error_analysis=[error.model_dump() for error in evaluation.error_analysis],
            justification=evaluation.justification,
        )
        return self._recipe.apply_recipe_to_user_prompt(user_prompt)

    def rewrite(
        self,
        model,
        text_source: str,
        translation: str,
        evaluation: TranslationEvaluation,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
    ) -> str:
        prompt = self.render(
            text_source, translation, evaluation, source_language_description, target_language_description
        )
        result, _trace = model.generate(prompt, parser=self._recipe.parse, max_correction_steps=max_correction_steps)
        return result

    async def arewrite(
        self,
        model,
        text_source: str,
        translation: str,
        evaluation: TranslationEvaluation,
        source_language_description: str,
        target_language_description: str,
        max_correction_steps: int,
    ) -> str:
        prompt = self.render(
            text_source, translation, evaluation, source_language_description, target_language_description
        )
        result, _trace = await model.agenerate(
            prompt, parser=self._recipe.parse, max_correction_steps=max_correction_steps
        )
        return result
