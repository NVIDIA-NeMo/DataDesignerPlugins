# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings

import pytest
from pydantic import ValidationError

from data_designer_nemotron_bharat_translation.config import DocumentTranslationConfig
from data_designer_nemotron_bharat_translation.document_translation import (
    DocumentTranslationColumnGenerator,
    TranslationEvaluation,
)
from data_designer_nemotron_bharat_translation.language_codes import (
    BODHAN_LANGUAGE_DESCRIPTIONS,
    GENERIC_LANGUAGE_DESCRIPTIONS,
)
from data_designer_nemotron_bharat_translation.prompts import BODHAN_TRANSLATION_PROMPT, GENERIC_TRANSLATION_PROMPT


class FakeModelFacade:
    """Stand-in for ``ModelFacade`` that replays a queue of canned responses."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, *, parser, max_correction_steps: int = 0):
        self.calls.append(prompt)
        return self._responses.pop(0), []

    async def agenerate(self, prompt: str, *, parser, max_correction_steps: int = 0):
        self.calls.append(prompt)
        return self._responses.pop(0), []


class FakeModelRegistry:
    def __init__(self, models: dict[str, FakeModelFacade]) -> None:
        self._models = models

    def get_model(self, *, model_alias: str) -> FakeModelFacade:
        return self._models[model_alias]


class FakeResourceProvider:
    def __init__(self, models: dict[str, FakeModelFacade]) -> None:
        self.model_registry = FakeModelRegistry(models)


def make_generator(
    config: DocumentTranslationConfig, models: dict[str, FakeModelFacade]
) -> DocumentTranslationColumnGenerator:
    return DocumentTranslationColumnGenerator(config, FakeResourceProvider(models))


def make_config(**overrides) -> DocumentTranslationConfig:
    fields = {
        "name": "translation",
        "model_alias": "translator",
        "evaluator_model_alias": "translator",
        "source_language": "en-Latn-IN",
        "target_language": "hi-Deva-IN",
    }
    fields.update(overrides)
    return DocumentTranslationConfig(**fields)


def evaluation(score: int) -> TranslationEvaluation:
    return TranslationEvaluation(analysis="ok", justification="fine", overall_score=score)


def test_translate_and_evaluate_without_rewrite() -> None:
    config = make_config()
    models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90)])}
    generator = make_generator(config, models)

    data = {"text": "hello"}
    result = generator.generate(data)

    assert result["translation"] == "नमस्ते"
    assert result["translation_evaluation"]["overall_score"] == 90
    assert result["translation_overall_score"] == 90
    assert result["translation_translation_history"] == []
    assert result["translation_evaluation_history"] == []
    assert result["translation_intermediate"] is None
    assert result["translation_intermediate_evaluation"] is None
    assert result["translation_intermediate_overall_score"] is None
    assert result["translation_window_traces"] is None
    assert result["translation_intermediate_window_traces"] is None


def test_custom_translation_prompt_is_rendered_with_text_and_language_descriptions() -> None:
    config = make_config(
        translation_prompt="From {{ source_language_description }} to {{ target_language_description }}: {{ text }}"
    )
    models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90)])}
    generator = make_generator(config, models)

    data = {"text": "hello"}
    generator.generate(data)

    assert models["translator"].calls[0] == "From Indian English in Latin script to Hindi in Devanagari script: hello"


def test_translation_mode_defaults_to_generic_prompt_and_descriptions() -> None:
    config = make_config()

    assert config.translation_prompt == GENERIC_TRANSLATION_PROMPT
    assert config.language_descriptions == GENERIC_LANGUAGE_DESCRIPTIONS


def test_translation_mode_bodhan_picks_bodhan_prompt_and_descriptions() -> None:
    config = make_config(translation_mode="bodhan")

    assert config.translation_prompt == BODHAN_TRANSLATION_PROMPT
    assert config.language_descriptions == BODHAN_LANGUAGE_DESCRIPTIONS


def test_explicit_translation_prompt_overrides_translation_mode() -> None:
    config = make_config(translation_mode="bodhan", translation_prompt="{{ text }}")

    assert config.translation_prompt == "{{ text }}"
    assert config.language_descriptions == BODHAN_LANGUAGE_DESCRIPTIONS


def test_explicit_language_descriptions_overrides_translation_mode() -> None:
    config = make_config(translation_mode="bodhan", language_descriptions={"en-Latn-IN": "English"})

    assert config.translation_prompt == BODHAN_TRANSLATION_PROMPT
    assert config.language_descriptions == {"en-Latn-IN": "English"}


def test_bodhan_mode_warns_when_translation_prompt_overridden() -> None:
    with pytest.warns(UserWarning, match="translation_mode='bodhan'"):
        make_config(translation_mode="bodhan", translation_prompt="{{ text }}")


def test_bodhan_mode_warns_when_language_descriptions_overridden() -> None:
    with pytest.warns(UserWarning, match="translation_mode='bodhan'"):
        make_config(translation_mode="bodhan", language_descriptions={"en-Latn-IN": "English"})


def test_bodhan_mode_does_not_warn_with_defaults() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_config(translation_mode="bodhan")


def test_bodhan_mode_does_not_warn_when_generator_revalidates_config() -> None:
    # DocumentTranslationColumnGenerator's base class re-validates the config
    # (model_validate) when wrapping it -- confirm that doesn't mistake the
    # config's own prior defaulting for a user override.
    config = make_config(translation_mode="bodhan")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_generator(config, {"translator": FakeModelFacade([])})


def test_bodhan_mode_forces_max_rewrites_to_zero() -> None:
    with pytest.warns(UserWarning, match="does not support the rewrite loop"):
        config = make_config(translation_mode="bodhan", max_rewrites=2, rewrite_threshold=75)

    assert config.max_rewrites == 0


def test_bodhan_mode_forces_max_rewrites_intermediate_to_zero() -> None:
    with pytest.warns(UserWarning, match="does not support the rewrite loop"):
        config = make_config(translation_mode="bodhan", max_rewrites_intermediate=2)

    assert config.max_rewrites_intermediate == 0


def test_bodhan_mode_does_not_warn_when_rewrites_already_zero() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_config(translation_mode="bodhan", max_rewrites=0, max_rewrites_intermediate=0)


def test_generic_mode_allows_rewrites() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = make_config(translation_mode="generic", max_rewrites=2)

    assert config.max_rewrites == 2


def test_generic_mode_does_not_warn_when_overridden() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_config(translation_mode="generic", translation_prompt="{{ text }}")


def test_bodhan_translation_mode_renders_bodhan_prompt_at_generation_time() -> None:
    config = make_config(translation_mode="bodhan")
    models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90)])}
    generator = make_generator(config, models)

    data = {"text": "hello"}
    generator.generate(data)

    assert models["translator"].calls[0] == "Translate the following text into Hindi:\n\nhello"


def test_rewrite_loop_runs_until_threshold_met() -> None:
    config = make_config(max_rewrites=2, rewrite_threshold=80)
    models = {
        "translator": FakeModelFacade(
            [
                "draft one",
                evaluation(50),
                "draft two",
                evaluation(85),
            ]
        )
    }
    generator = make_generator(config, models)

    data = {"text": "hello"}
    result = generator.generate(data)

    assert result["translation"] == "draft two"
    assert result["translation_evaluation"]["overall_score"] == 85
    assert result["translation_translation_history"] == ["draft one"]
    assert result["translation_evaluation_history"] == [evaluation(50).model_dump()]


def test_rewrite_loop_stops_at_max_rewrites_even_below_threshold() -> None:
    config = make_config(max_rewrites=1, rewrite_threshold=99)
    models = {
        "translator": FakeModelFacade(
            [
                "draft one",
                evaluation(50),
                "draft two",
                evaluation(60),
            ]
        )
    }
    generator = make_generator(config, models)

    data = {"text": "hello"}
    result = generator.generate(data)

    assert result["translation"] == "draft two"
    assert result["translation_evaluation"]["overall_score"] == 60
    assert len(result["translation_translation_history"]) == 1


def test_evaluator_model_alias_is_required() -> None:
    with pytest.raises(ValidationError):
        DocumentTranslationConfig(
            name="translation",
            model_alias="translator",
            source_language="en-Latn-IN",
            target_language="hi-Deva-IN",
        )


def test_evaluator_model_alias_routes_evaluation_to_separate_model() -> None:
    config = make_config(evaluator_model_alias="judge")
    models = {
        "translator": FakeModelFacade(["नमस्ते"]),
        "judge": FakeModelFacade([evaluation(95)]),
    }
    generator = make_generator(config, models)

    data = {"text": "hello"}
    result = generator.generate(data)

    assert result["translation_overall_score"] == 95
    assert len(models["translator"].calls) == 1  # translator only got the translation call
    assert len(models["judge"].calls) == 1


def test_intermediate_pivot_runs_two_stages() -> None:
    config = make_config(intermediate_language="ta-Taml-IN")
    models = {
        "translator": FakeModelFacade(
            [
                "intermediate text",
                evaluation(92),
                "final text",
                evaluation(88),
            ]
        )
    }
    generator = make_generator(config, models)

    data = {"text": "hello"}
    result = generator.generate(data)

    assert result["translation_intermediate"] == "intermediate text"
    assert result["translation_intermediate_overall_score"] == 92
    assert result["translation"] == "final text"
    assert result["translation_overall_score"] == 88


def test_intermediate_pivot_skipped_when_unset() -> None:
    config = make_config()
    models = {"translator": FakeModelFacade(["final text", evaluation(90)])}
    generator = make_generator(config, models)

    data = {"text": "hello"}
    result = generator.generate(data)

    assert result["translation_intermediate"] is None
    assert result["translation"] == "final text"


def test_unknown_source_language_raises_key_error() -> None:
    config = make_config(source_language="zz-Zzzz-ZZ")
    generator = make_generator(config, {"translator": FakeModelFacade([])})

    data = {"text": "hello"}
    with pytest.raises(KeyError, match="language_descriptions"):
        generator.generate(data)


def test_unknown_target_language_raises_key_error() -> None:
    config = make_config(target_language="zz-Zzzz-ZZ")
    generator = make_generator(config, {"translator": FakeModelFacade([])})

    data = {"text": "hello"}
    with pytest.raises(KeyError, match="language_descriptions"):
        generator.generate(data)


class FakeWindowingModelFacade:
    """Serves both phases: translation echoes ``<t>..</t>``, a merge prompt echoes the owned segment."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.merge_prompts: list[str] = []

    def generate(self, prompt: str, *, parser=None, max_correction_steps: int = 0, **kwargs):
        return self._respond(prompt)

    async def agenerate(self, prompt: str, *, parser=None, max_correction_steps: int = 0, **kwargs):
        return self._respond(prompt)

    def _respond(self, prompt: str):
        if "# The portion to return" in prompt:
            self.merge_prompts.append(prompt)
            body = prompt.split("# The portion to return, in ", 1)[1]
            return body.split("\n", 1)[1].split("\n\nRules:")[0], []
        self.calls.append(prompt)
        return f"<t>{prompt}</t>", []


class StubEncoding:
    def __init__(self, token_count: int) -> None:
        self.ids = [0] * token_count


class StubTokenizer:
    """1 token per whitespace-separated word -- keeps the window arithmetic obvious."""

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[StubEncoding]:
        return [StubEncoding(max(len(t.split()), 1)) for t in texts]


def test_enable_windowing_translates_via_windows_then_evaluates_the_merged_document() -> None:
    config = make_config(enable_windowing=True, window_tokens=2, evaluator_model_alias="judge")
    translator_model = FakeWindowingModelFacade()
    models = {"translator": translator_model, "judge": FakeModelFacade([evaluation(90)])}
    generator = make_generator(config, models)
    generator._translator._tokenizer = StubTokenizer()

    data = {"text": "a\nb\nc"}
    result = generator.generate(data)

    assert len(translator_model.calls) == 2  # 2 windows translated
    assert len(translator_model.merge_prompts) > 0  # merge phase ran
    assert result["translation_overall_score"] == 90
    windows = result["translation_window_traces"]
    assert [w["translated_window"] for w in windows] == [f"<t>{prompt}</t>" for prompt in translator_model.calls]
    assert [w["window"] for w in windows] == ["a\nb", "b\nc"]
    assert all(w["extracted_translated_window"] is not None for w in windows)
    assert all(w["extracted_window"] is not None for w in windows)
    assert result["translation_intermediate_window_traces"] is None


def test_keep_window_traces_false_drops_the_trace_data() -> None:
    config = make_config(
        enable_windowing=True, window_tokens=2, evaluator_model_alias="judge", keep_window_traces=False
    )
    translator_model = FakeWindowingModelFacade()
    models = {"translator": translator_model, "judge": FakeModelFacade([evaluation(90)])}
    generator = make_generator(config, models)
    generator._translator._tokenizer = StubTokenizer()

    data = {"text": "a\nb\nc"}
    result = generator.generate(data)

    assert len(translator_model.calls) == 2  # windowing still ran and translated
    assert result["translation"]  # merged translation still produced
    assert result["translation_window_traces"] is None
    assert result["translation_intermediate_window_traces"] is None


def test_enable_windowing_writes_intermediate_window_traces_for_the_intermediate_stage() -> None:
    config = make_config(
        enable_windowing=True, window_tokens=2, evaluator_model_alias="judge", intermediate_language="ta-Taml-IN"
    )
    translator_model = FakeWindowingModelFacade()
    models = {"translator": translator_model, "judge": FakeModelFacade([evaluation(90), evaluation(90)])}
    generator = make_generator(config, models)
    generator._translator._tokenizer = StubTokenizer()

    data = {"text": "a\nb\nc"}
    result = generator.generate(data)

    assert result["translation_intermediate_window_traces"] is not None
    assert result["translation_window_traces"] is not None


def test_get_model_aliases_includes_merge_alias_when_windowing_enabled() -> None:
    config = make_config(enable_windowing=True, window_merge_model_alias="patcher")
    assert config.get_model_aliases() == ["translator", "patcher"]


def test_get_model_aliases_merge_defaults_to_translator_when_windowing_enabled() -> None:
    config = make_config(enable_windowing=True)
    assert config.get_model_aliases() == ["translator"]


def test_enable_windowing_forces_max_rewrites_to_zero() -> None:
    with pytest.warns(UserWarning, match="enable_windowing is set"):
        config = make_config(enable_windowing=True, max_rewrites=2, rewrite_threshold=75)

    assert config.max_rewrites == 0


def test_enable_windowing_forces_max_rewrites_intermediate_to_zero() -> None:
    with pytest.warns(UserWarning, match="enable_windowing is set"):
        config = make_config(enable_windowing=True, max_rewrites_intermediate=2)

    assert config.max_rewrites_intermediate == 0


def test_enable_windowing_does_not_warn_when_rewrites_already_zero() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_config(enable_windowing=True, max_rewrites=0, max_rewrites_intermediate=0)


def test_windowing_disabled_allows_rewrites() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = make_config(max_rewrites=2)

    assert config.max_rewrites == 2


def test_target_language_script_unsupported_pair_raises() -> None:
    with pytest.raises(ValidationError, match="No script remapper"):
        make_config(target_language_script="Mtei")  # target_language defaults to hi-Deva-IN (Devanagari)


def test_target_language_script_invalid_format_raises() -> None:
    with pytest.raises(ValidationError, match="4-letter ISO 15924"):
        make_config(target_language="mni-Beng-IN", target_language_script="ab")


def test_target_language_script_normalizes_case() -> None:
    config = make_config(target_language="mni-Beng-IN", target_language_script="mtei")
    assert config.target_language_script == "Mtei"


def test_remapped_target_language_property() -> None:
    config = make_config(target_language="mni-Beng-IN", target_language_script="Mtei")
    assert config.remapped_target_language == "mni-Mtei-IN"


def test_remapped_target_language_is_none_when_script_unset() -> None:
    config = make_config()
    assert config.remapped_target_language is None


def test_target_language_script_remaps_final_translation_and_reevaluates() -> None:
    config = make_config(target_language="mni-Beng-IN", target_language_script="Mtei")
    models = {"translator": FakeModelFacade(["কা", evaluation(80), evaluation(95)])}
    generator = make_generator(config, models)

    data = {"text": "hello"}
    result = generator.generate(data)

    assert result["translation"] == "ꯀꯥ"  # remapped from কা
    assert result["translation_overall_score"] == 95
    assert result["translation_translation_history"] == ["কা"]
    assert result["translation_evaluation_history"] == [evaluation(80).model_dump()]


def test_target_language_script_preserves_code_literals_by_default() -> None:
    config = make_config(target_language="mni-Beng-IN", target_language_script="Mtei")
    translation = 'কা `label = "বাংলা"` কা'
    models = {"translator": FakeModelFacade([translation, evaluation(80), evaluation(95)])}
    generator = make_generator(config, models)

    result = generator.generate({"text": "hello"})

    assert result["translation"] == 'ꯀꯥ `label = "বাংলা"` ꯀꯥ'


def test_target_language_script_can_disable_protected_span_preservation() -> None:
    config = make_config(
        target_language="mni-Beng-IN", target_language_script="Mtei", remap_preserves_protected_spans=False
    )
    translation = "কা `কা` কা"
    models = {"translator": FakeModelFacade([translation, evaluation(80), evaluation(95)])}
    generator = make_generator(config, models)

    result = generator.generate({"text": "hello"})

    assert result["translation"] == "ꯀꯥ `ꯀꯥ` ꯀꯥ"


def test_target_language_script_reevaluates_against_the_original_source_text() -> None:
    config = make_config(target_language="mni-Beng-IN", target_language_script="Mtei")
    models = {"translator": FakeModelFacade(["কা", evaluation(80), evaluation(95)])}
    generator = make_generator(config, models)

    generator.generate({"text": "hello"})

    remap_evaluation_prompt = models["translator"].calls[-1]
    assert "hello" in remap_evaluation_prompt
    assert "ꯀꯥ" in remap_evaluation_prompt
    assert "কা" not in remap_evaluation_prompt


def test_target_language_script_unset_does_not_remap() -> None:
    config = make_config(target_language="mni-Beng-IN")
    models = {"translator": FakeModelFacade(["কা", evaluation(90)])}
    generator = make_generator(config, models)

    result = generator.generate({"text": "hello"})

    assert result["translation"] == "কা"
    assert result["translation_translation_history"] == []
    assert result["translation_evaluation_history"] == []


def test_target_language_script_leaves_intermediate_stage_unaffected() -> None:
    config = make_config(
        target_language="mni-Beng-IN",
        target_language_script="Mtei",
        intermediate_language="ta-Taml-IN",
    )
    models = {
        "translator": FakeModelFacade(
            [
                "intermediate text",
                evaluation(92),
                "কা",
                evaluation(80),
                evaluation(95),
            ]
        )
    }
    generator = make_generator(config, models)

    result = generator.generate({"text": "hello"})

    assert result["translation_intermediate"] == "intermediate text"
    assert result["translation_intermediate_overall_score"] == 92
    assert result["translation"] == "ꯀꯥ"
    assert result["translation_overall_score"] == 95


def test_native_numeral_probability_unsupported_language_raises() -> None:
    with pytest.raises(ValidationError, match="NATIVE_NUMBERS_DOC"):
        make_config(target_language="en-Latn-IN", native_numeral_probability=0.5)


def test_native_numeral_probability_requires_intermediate_language_coverage_too() -> None:
    with pytest.raises(ValidationError, match="NATIVE_NUMBERS_DOC"):
        make_config(
            target_language="hi-Deva-IN",
            intermediate_language="en-Latn-IN",  # not in NATIVE_NUMBERS_DOC
            native_numeral_probability=0.5,
        )


def test_native_numeral_probability_forced_to_zero_in_bodhan_mode() -> None:
    with pytest.warns(UserWarning, match="no native-numeral prompt variant"):
        config = make_config(translation_mode="bodhan", native_numeral_probability=0.5)

    assert config.native_numeral_probability == 0.0


def test_native_numeral_probability_zero_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_config(native_numeral_probability=0.0)


def test_native_numeral_probability_one_always_uses_the_native_numeral_prompt() -> None:
    config = make_config(native_numeral_probability=1.0)
    models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90)])}
    generator = make_generator(config, models)

    generator.generate({"text": "hello"})

    assert "NOT ASCII/Western 0-9 digits" in models["translator"].calls[0]
    assert "0->०" in models["translator"].calls[0]  # hi-Deva-IN's legend


def test_native_numeral_probability_zero_never_uses_the_native_numeral_prompt() -> None:
    config = make_config(native_numeral_probability=0.0)
    models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90)])}
    generator = make_generator(config, models)

    generator.generate({"text": "hello"})

    assert "NOT ASCII/Western 0-9 digits" not in models["translator"].calls[0]


def test_native_numeral_probability_pairs_are_cached_per_target_language() -> None:
    config = make_config(native_numeral_probability=1.0)
    models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90), "नमस्ते", evaluation(90)])}
    generator = make_generator(config, models)

    generator.generate({"text": "hello"})
    generator.generate({"text": "world"})

    assert len(generator._native_numeral_translators) == 1


def test_numeral_style_output_is_native_when_drawn() -> None:
    config = make_config(native_numeral_probability=1.0)
    models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90)])}
    generator = make_generator(config, models)

    result = generator.generate({"text": "hello"})

    assert result["translation_numeral_style"] == "native"


def test_numeral_style_output_is_none_when_not_drawn() -> None:
    config = make_config(native_numeral_probability=0.0)
    models = {"translator": FakeModelFacade(["namaste", evaluation(90)])}
    generator = make_generator(config, models)

    result = generator.generate({"text": "hello"})

    assert result["translation_numeral_style"] == "none"


def test_translation_prompt_native_numerals_defaults_to_the_generic_variant() -> None:
    config = make_config()
    assert "{native_numbers_doc}" in config.translation_prompt_native_numerals
    assert "NOT ASCII/Western 0-9 digits" in config.translation_prompt_native_numerals


def test_evaluation_prompt_is_the_same_regardless_of_numeral_style() -> None:
    native_config = make_config(native_numeral_probability=1.0)
    plain_config = make_config(native_numeral_probability=0.0)
    native_models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90)])}
    plain_models = {"translator": FakeModelFacade(["namaste", evaluation(90)])}

    make_generator(native_config, native_models).generate({"text": "hello"})
    make_generator(plain_config, plain_models).generate({"text": "hello"})

    native_evaluate_call = native_models["translator"].calls[1]
    plain_evaluate_call = plain_models["translator"].calls[1]
    assert native_evaluate_call.replace("नमस्ते", "namaste") == plain_evaluate_call
    assert "Numeral Script" not in native_evaluate_call
    assert "Numeral Script" not in plain_evaluate_call


def test_custom_translation_prompt_native_numerals_is_used_over_the_generic_default() -> None:
    config = make_config(
        native_numeral_probability=1.0,
        translation_prompt_native_numerals="Custom native prompt: {{ text }} {native_numbers_doc}",
    )
    models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90)])}
    generator = make_generator(config, models)

    generator.generate({"text": "hello"})

    assert models["translator"].calls[0].startswith("Custom native prompt: hello")
    assert "NOT ASCII/Western 0-9 digits" not in models["translator"].calls[0]
    assert "0->०" in models["translator"].calls[0]  # hi-Deva-IN's legend still resolved in


def test_missing_native_numbers_doc_placeholder_raises() -> None:
    with pytest.raises(ValidationError, match="native_numbers_doc"):
        make_config(
            native_numeral_probability=0.5,
            translation_prompt_native_numerals="No legend placeholder here: {{ text }}",
        )


def test_overriding_translation_prompt_without_native_numerals_counterpart_warns() -> None:
    with pytest.warns(UserWarning, match="translation_prompt_native_numerals is not"):
        make_config(
            native_numeral_probability=0.5,
            translation_prompt="custom {{ text }} {{ source_language_description }} {{ target_language_description }}",
        )


def test_overriding_evaluation_prompt_does_not_warn() -> None:
    # Evaluation always uses evaluation_prompt regardless of numeral style, so there is
    # no "did you forget the native-numeral counterpart" warning to fire here.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_config(
            native_numeral_probability=0.5,
            evaluation_prompt="custom {{ text }} {{ translation_result.translated_text }}",
        )


def test_overriding_both_prompt_and_its_native_numerals_counterpart_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_config(
            native_numeral_probability=0.5,
            translation_prompt="custom {{ text }}",
            translation_prompt_native_numerals="custom native {{ text }} {native_numbers_doc}",
        )


def test_agenerate_matches_generate_for_the_same_inputs() -> None:
    config = make_config()
    models = {"translator": FakeModelFacade(["नमस्ते", evaluation(90)])}
    generator = make_generator(config, models)

    data = {"text": "hello"}
    result = asyncio.run(generator.agenerate(data))

    assert result["translation"] == "नमस्ते"
    assert result["translation_overall_score"] == 90


def test_target_language_script_remaps_final_translation_async() -> None:
    config = make_config(target_language="mni-Beng-IN", target_language_script="Mtei")
    models = {"translator": FakeModelFacade(["কা", evaluation(80), evaluation(95)])}
    generator = make_generator(config, models)

    result = asyncio.run(generator.agenerate({"text": "hello"}))

    assert result["translation"] == "ꯀꯥ"
    assert result["translation_overall_score"] == 95
    assert result["translation_translation_history"] == ["কা"]
