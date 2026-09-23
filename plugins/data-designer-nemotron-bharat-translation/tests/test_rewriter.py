# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from data_designer_nemotron_bharat_translation.evaluator import ErrorSpan, TranslationEvaluation
from data_designer_nemotron_bharat_translation.rewriter import Rewriter


class FakeModelFacade:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, *, parser, max_correction_steps: int = 0):
        self.calls.append((prompt, {"parser": parser, "max_correction_steps": max_correction_steps}))
        return self._responses.pop(0), []

    async def agenerate(self, prompt: str, *, parser, max_correction_steps: int = 0):
        return self.generate(prompt, parser=parser, max_correction_steps=max_correction_steps)


def evaluation() -> TranslationEvaluation:
    return TranslationEvaluation(
        analysis="missed a clause",
        justification="dropped content",
        overall_score=40,
        error_analysis=[
            ErrorSpan(text_span="foo", error_category="Faithfulness", severity="Major", explanation="dropped")
        ],
    )


class TestRewriter:
    def test_render_includes_source_translation_and_evaluation_fields(self) -> None:
        rewriter = Rewriter(
            "{{ text_source }}|{{ text }}|{{ overall_score }}|{{ analysis }}|{{ justification }}|{{ error_analysis }}"
        )
        rendered = rewriter.render("source doc", "draft translation", evaluation(), "English", "Hindi")
        assert rendered.startswith("source doc|draft translation|40|missed a clause|dropped content|")
        assert "foo" in rendered

    def test_rewrite_calls_model_with_parser_and_max_correction_steps(self) -> None:
        rewriter = Rewriter("{{ text }}")
        model = FakeModelFacade(["corrected translation"])
        result = rewriter.rewrite(
            model, "source doc", "draft", evaluation(), "English", "Hindi", max_correction_steps=2
        )
        assert result == "corrected translation"
        prompt, kwargs = model.calls[0]
        assert prompt == "draft"
        assert kwargs["max_correction_steps"] == 2
        assert callable(kwargs["parser"])

    def test_arewrite_matches_rewrite(self) -> None:
        rewriter = Rewriter("{{ text }}")
        model = FakeModelFacade(["corrected translation"])
        result = asyncio.run(
            rewriter.arewrite(model, "source doc", "draft", evaluation(), "English", "Hindi", max_correction_steps=0)
        )
        assert result == "corrected translation"
