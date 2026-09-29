# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from data_designer_nemotron_bharat_translation.translator import Translator


class FakeModelFacade:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, *, parser, max_correction_steps: int = 0):
        self.calls.append((prompt, {"parser": parser, "max_correction_steps": max_correction_steps}))
        return self._responses.pop(0), []

    async def agenerate(self, prompt: str, *, parser, max_correction_steps: int = 0):
        return self.generate(prompt, parser=parser, max_correction_steps=max_correction_steps)


class TestTranslator:
    def test_render_applies_placeholders(self) -> None:
        translator = Translator("{{ source_language_description }} -> {{ target_language_description }}: {{ text }}")
        assert translator.render("hello", "English", "Hindi") == "English -> Hindi: hello"

    def test_translate_calls_model_with_parser_and_max_correction_steps(self) -> None:
        translator = Translator("{{ text }}")
        model = FakeModelFacade(["नमस्ते"])
        result, windows = translator.translate(model, "hello", "English", "Hindi", max_correction_steps=2)
        assert result == "नमस्ते"
        assert windows is None
        prompt, kwargs = model.calls[0]
        assert prompt == "hello"
        assert kwargs["max_correction_steps"] == 2
        assert callable(kwargs["parser"])

    def test_atranslate_matches_translate(self) -> None:
        translator = Translator("{{ text }}")
        model = FakeModelFacade(["नमस्ते"])
        result, windows = asyncio.run(translator.atranslate(model, "hello", "English", "Hindi", max_correction_steps=0))
        assert result == "नमस्ते"
        assert windows is None

    def test_enable_windowing_without_merge_model_raises(self) -> None:
        with pytest.raises(ValueError, match="merge_model is required"):
            Translator("{{ text }}", enable_windowing=True)


class TestBuildSpans:
    def test_token_budget_sets_the_window_size(self) -> None:
        spans = Translator.build_spans([10] * 5, window_tokens=25, overlap=1)
        assert spans == [(0, 2), (1, 3), (2, 4), (3, 5)]

    def test_bigger_budget_takes_more_chunks(self) -> None:
        spans = Translator.build_spans([10] * 5, window_tokens=35, overlap=1)
        assert spans == [(0, 3), (2, 5)]

    def test_at_least_two_chunks_even_over_budget(self) -> None:
        spans = Translator.build_spans([500, 500, 500], window_tokens=10, overlap=0)
        assert all(hi - lo >= 2 for lo, hi in spans)
        assert spans[0] == (0, 2)

    def test_large_overlap_still_advances(self) -> None:
        spans = Translator.build_spans([10] * 4, window_tokens=25, overlap=99)
        assert [lo for lo, _ in spans] == sorted({lo for lo, _ in spans})
        assert spans[-1][1] == 4


class FakeWindowingModelFacade:
    """Serves both phases: translation echoes ``<t>..</t>``, a merge prompt echoes the owned segment."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.merge_prompts: list[str] = []
        self.fail = fail

    def generate(self, prompt: str, *, parser=None, max_correction_steps: int = 0, **kwargs):
        return self._respond(prompt)

    async def agenerate(self, prompt: str, *, parser=None, max_correction_steps: int = 0, **kwargs):
        return self._respond(prompt)

    def _respond(self, prompt: str):
        if "# The portion to return" in prompt:
            self.merge_prompts.append(prompt)
            if self.fail:
                raise RuntimeError("model unavailable")
            body = prompt.split("# The portion to return, in ", 1)[1]
            return body.split("\n", 1)[1].split("\n\nRules:")[0], []
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("model unavailable")
        return f"<t>{prompt}</t>", []


class StubEncoding:
    def __init__(self, token_count: int) -> None:
        self.ids = [0] * token_count


class StubTokenizer:
    """1 token per whitespace-separated word -- keeps the window arithmetic obvious."""

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[StubEncoding]:
        return [StubEncoding(max(len(t.split()), 1)) for t in texts]


def make_windowed_translator(
    model: FakeWindowingModelFacade | None = None, merge_model: FakeWindowingModelFacade | None = None, **overrides
) -> tuple[Translator, FakeWindowingModelFacade]:
    stub = model or FakeWindowingModelFacade()
    translator = Translator("{{ text }}", enable_windowing=True, merge_model=merge_model or stub, **overrides)
    translator._tokenizer = StubTokenizer()
    return translator, stub


class TestWindowedTranslate:
    def test_single_chunk_still_goes_through_the_merge_model(self) -> None:
        translator, model = make_windowed_translator()
        out, windows = translator.translate(model, "only one line", "English", "Hindi", max_correction_steps=0)
        assert out == "only one line"
        assert windows == [
            {
                "window": "only one line",
                "translated_window": "<t>only one line</t>",
                "extracted_window": "only one line",
                "extracted_translated_window": "only one line",
            }
        ]
        assert len(model.merge_prompts) == 1

    def test_one_call_per_window(self) -> None:
        translator, model = make_windowed_translator(window_tokens=2)
        _merged, windows = translator.translate(model, "a\nb\nc", "English", "Hindi", max_correction_steps=0)
        assert len(model.calls) == 2
        assert "a\nb" in model.calls[0]
        assert "b\nc" in model.calls[1]
        assert len(windows) == 2

    def test_model_failure_raises(self) -> None:
        translator, model = make_windowed_translator(model=FakeWindowingModelFacade(fail=True))
        with pytest.raises(RuntimeError, match="model unavailable"):
            translator.translate(model, "a\nb", "English", "Hindi", max_correction_steps=0)

    def test_unusable_model_output_raises(self) -> None:
        class EmptyStub(FakeWindowingModelFacade):
            def generate(self, prompt, *, parser=None, max_correction_steps=0, **kwargs):
                return "   ", []

        translator, model = make_windowed_translator(model=EmptyStub())
        with pytest.raises(ValueError, match="no usable text"):
            translator.translate(model, "a\nb", "English", "Hindi", max_correction_steps=0)

    def test_empty_text(self) -> None:
        translator, model = make_windowed_translator()
        out, windows = translator.translate(model, "", "English", "Hindi", max_correction_steps=0)
        assert out == ""
        assert windows == [
            {"window": "", "translated_window": "", "extracted_window": None, "extracted_translated_window": None}
        ]

    def test_atranslate_matches_translate(self) -> None:
        translator, model = make_windowed_translator(window_tokens=2)
        asyncio.run(translator.atranslate(model, "a\nb\nc", "English", "Hindi", max_correction_steps=0))
        assert len(model.calls) == 2

    def test_keep_window_traces_false_drops_the_traces(self) -> None:
        translator, model = make_windowed_translator(window_tokens=2, keep_window_traces=False)
        out, windows = translator.translate(model, "a\nb\nc", "English", "Hindi", max_correction_steps=0)
        assert out  # merging still happened
        assert windows is None
        assert len(model.calls) == 2  # windows were still translated


class TestWindowedMerge:
    """Phase 2: each window's own non-overlapping segment is extracted, then stacked in order."""

    def test_segment_count_matches_the_windows_that_own_chunks(self) -> None:
        translator, model = make_windowed_translator(window_tokens=2)
        out, windows = translator.translate(model, "a\nb\nc\nd\ne", "English", "Hindi", max_correction_steps=0)
        assert out
        assert len(windows) == 4  # spans [0,2) [1,3) [2,4) [3,5) -- see build_spans
        assert len(model.merge_prompts) == 4  # a\nb, c, d, e -- see _owned_range
        assert all(w["extracted_translated_window"] is not None for w in windows)

    def test_a_window_that_owns_no_chunks_gets_no_extracted_segment(self) -> None:
        # Contrived spans where window 1 is fully subsumed by window 0's owned
        # range ([spans[0][1], spans[1][1]) is empty) -- _merge_calls skips it.
        translator, model = make_windowed_translator()
        chunks = ["a", "b", "c"]
        spans = [(0, 3), (0, 3), (3, 3)]
        translations = ["<t>abc</t>", "<t>abc</t>", ""]
        _merged, _steps, extracted_windows, extracted_translations = translator._merge_all(
            chunks, spans, translations, {}, "English", "Hindi"
        )
        assert extracted_translations[1] is None
        assert extracted_translations[2] is None
        assert extracted_translations[0] is not None
        assert extracted_windows[1] is None
        assert extracted_windows[2] is None
        assert extracted_windows[0] is not None

    def test_merge_preserves_indentation_but_strips_stray_newline_padding(self) -> None:
        class IndentedStub(FakeWindowingModelFacade):
            def generate(self, prompt, *, parser=None, max_correction_steps: int = 0, **kwargs):
                return "\n\n    indented line\n\n", []

        translator, model = make_windowed_translator(model=IndentedStub())
        merged, _count, _extracted_windows, extracted_translations = translator._merge_all(
            ["a"], [(0, 1)], ["<t>a</t>"], {}, "English", "Hindi"
        )
        assert merged == "    indented line"
        assert extracted_translations[0] == "    indented line"

    def test_merge_with_a_multi_character_separator_does_not_strip_unrelated_content(self) -> None:
        # window_separator=" | " means the char set {' ', '|'} would also match unrelated
        # leading/trailing pipes (e.g. real markdown table syntax) if stripped via
        # str.strip(window_separator) -- only exact " | " runs should be removed, and
        # "|Column1|" contains no such run at either edge, so it must survive untouched.
        class TableRowStub(FakeWindowingModelFacade):
            def generate(self, prompt, *, parser=None, max_correction_steps: int = 0, **kwargs):
                return "|Column1|", []

        translator, model = make_windowed_translator(model=TableRowStub(), window_separator=" | ")
        merged, _count, _extracted_windows, extracted_translations = translator._merge_all(
            ["a"], [(0, 1)], ["<t>a</t>"], {}, "English", "Hindi"
        )
        assert merged == "|Column1|"
        assert extracted_translations[0] == "|Column1|"

    def test_merge_uses_the_merge_model(self) -> None:
        translation_model, merge_model = FakeWindowingModelFacade(), FakeWindowingModelFacade()
        translator, _ = make_windowed_translator(model=translation_model, merge_model=merge_model, window_tokens=2)
        translator.translate(translation_model, "a\nb\nc\nd\ne", "English", "Hindi", max_correction_steps=0)
        assert translation_model.merge_prompts == []
        assert len(merge_model.merge_prompts) == 4
        assert merge_model.calls == []

    def test_async_merge_matches_sync(self) -> None:
        class AsyncStub(FakeWindowingModelFacade):
            async def agenerate(self, prompt, *, parser=None, max_correction_steps=0, **kwargs):
                return self.generate(prompt)

        sync_translator, sync_model = make_windowed_translator(window_tokens=2)
        sync_out, sync_windows = sync_translator.translate(
            sync_model, "a\nb\nc\nd\ne", "English", "Hindi", max_correction_steps=0
        )

        async_translator, async_model = make_windowed_translator(model=AsyncStub(), window_tokens=2)
        async_out, async_windows = asyncio.run(
            async_translator.atranslate(async_model, "a\nb\nc\nd\ne", "English", "Hindi", max_correction_steps=0)
        )
        assert async_out == sync_out
        assert async_windows == sync_windows

    def test_merge_prompt_includes_record_fields(self) -> None:
        translator, model = make_windowed_translator()
        translator.translate(model, "a\nb", "English", "Hindi", max_correction_steps=0, record={"topic": "weather"})
        assert (
            "weather" not in model.merge_prompts[0]
        )  # MERGE_PROMPT doesn't reference `topic`, but rendering must not fail
