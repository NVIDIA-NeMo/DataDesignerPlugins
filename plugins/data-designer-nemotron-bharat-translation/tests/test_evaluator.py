# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from data_designer_nemotron_bharat_translation.evaluator import Evaluator, TranslationEvaluation


class FakeModelFacade:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, *, parser, max_correction_steps: int = 0):
        self.calls.append((prompt, {"parser": parser, "max_correction_steps": max_correction_steps}))
        return self._responses.pop(0), []

    async def agenerate(self, prompt: str, *, parser, max_correction_steps: int = 0):
        return self.generate(prompt, parser=parser, max_correction_steps=max_correction_steps)


def evaluation(score: int) -> TranslationEvaluation:
    return TranslationEvaluation(analysis="ok", justification="fine", overall_score=score)


class TestEvaluator:
    def test_render_includes_text_and_translation_result(self) -> None:
        evaluator = Evaluator(
            "{{ source_language_description }}/{{ target_language_description }}: {{ text }} -> {{ translation_result.translated_text }}"
        )
        rendered = evaluator.render("hello", "नमस्ते", "English", "Hindi")
        # PydanticResponseRecipe appends JSON-schema formatting instructions after the rendered template.
        assert rendered.startswith("English/Hindi: hello -> नमस्ते")

    def test_evaluate_calls_model_with_parser_and_max_correction_steps(self) -> None:
        evaluator = Evaluator("{{ text }} -> {{ translation_result.translated_text }}")
        model = FakeModelFacade([evaluation(90)])
        result = evaluator.evaluate(model, "hello", "नमस्ते", "English", "Hindi", max_correction_steps=2)
        assert result.overall_score == 90
        prompt, kwargs = model.calls[0]
        assert prompt.startswith("hello -> नमस्ते")
        assert kwargs["max_correction_steps"] == 2
        assert callable(kwargs["parser"])

    def test_aevaluate_matches_evaluate(self) -> None:
        evaluator = Evaluator("{{ text }} -> {{ translation_result.translated_text }}")
        model = FakeModelFacade([evaluation(75)])
        result = asyncio.run(evaluator.aevaluate(model, "hello", "नमस्ते", "English", "Hindi", max_correction_steps=0))
        assert result.overall_score == 75
