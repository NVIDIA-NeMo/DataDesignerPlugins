# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct translate + MQM evaluate + optional rewrite loop generator."""

from __future__ import annotations

import functools
import random

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel

from data_designer_nemotron_bharat_translation.config import DocumentTranslationConfig
from data_designer_nemotron_bharat_translation.evaluator import Evaluator, TranslationEvaluation
from data_designer_nemotron_bharat_translation.language_codes import parse_bcp47
from data_designer_nemotron_bharat_translation.prompts import native_numbers_doc
from data_designer_nemotron_bharat_translation.rewriter import Rewriter
from data_designer_nemotron_bharat_translation.script_remap import remap_script
from data_designer_nemotron_bharat_translation.translator import Translator


class DocumentTranslationColumnGenerator(ColumnGeneratorWithModel[DocumentTranslationConfig]):
    """Translates ``text_column`` into ``config.target_language``, scores it, then rewrites as needed.

    Per row, runs a translate-evaluate-rewrite procedure (see
    ``_run_stage``/``_arun_stage``):

    1. Renders ``config.translation_prompt`` with ``text``,
       ``source_language_description``, and ``target_language_description``,
       sends it to the model, and treats the plain-text response as the
       current translation.
    2. Scores the current translation with ``config.evaluation_prompt`` (the
       MQM-based auditor prompt), producing the current evaluation.
    3. While the current evaluation's ``overall_score`` is below
       ``rewrite_threshold`` and fewer than ``max_rewrites`` rewrites have
       run: pushes the current translation/evaluation onto history,
       rewrites the translation with ``config.rewrite_prompt`` (a
       self-correction pass over the source, current translation, and its
       flagged errors), then re-evaluates the rewrite as the new current
       translation/evaluation.

    When ``config.intermediate_language`` is unset, or equals
    ``config.target_language``, this procedure runs exactly once per row:
    source language -> target language, using
    ``config.max_rewrites``/``config.rewrite_threshold``. Its final
    translation/evaluation are written to ``name``/``<name>_evaluation``,
    and every superseded translation/evaluation pair is in
    ``<name>_translation_history``/``<name>_evaluation_history`` (oldest
    first, same length as each other). The intermediate columns
    (``<name>_intermediate*``) are ``None``/empty in this case.

    When ``config.intermediate_language`` is set and differs from
    ``config.target_language``, the procedure runs twice per row: first
    source language -> the intermediate language (using
    ``config.max_rewrites_intermediate``/``config.rewrite_threshold_intermediate``),
    writing its final result to
    ``<name>_intermediate``/``<name>_intermediate_evaluation``/``<name>_intermediate_overall_score``/
    ``<name>_intermediate_translation_history``/``<name>_intermediate_evaluation_history``;
    then that intermediate language -> target language (using
    ``config.max_rewrites``/``config.rewrite_threshold`` as before),
    starting from the intermediate stage's final translation rather than
    the original ``text``. That second stage's final
    translation/evaluation/history are written to
    ``name``/``<name>_evaluation``/``<name>_translation_history``/``<name>_evaluation_history``
    -- i.e. ``name`` always holds the final (target-language) result
    regardless of whether an intermediate pass ran.

    No user-turn/persona phrasing is simulated -- this is a direct
    translate-evaluate-rewrite column. Every language code used --
    ``config.source_language``, ``config.target_language``, and (if set)
    ``config.intermediate_language`` -- must have an entry in
    ``config.language_descriptions``; a code with no entry raises
    ``KeyError``.

    If ``config.target_language_script`` is set, the final stage's
    translation/evaluation above (after the rewrite loop finishes) is
    superseded by a deterministic script remap (see ``_remap_script``):
    the pre-remap translation/evaluation are pushed onto
    ``<name>_translation_history``/``<name>_evaluation_history``, the
    translation is remapped into ``config.target_language_script``, and a
    fresh evaluation scores the remapped text against the same source --
    that becomes ``name``/``<name>_evaluation``. Only the final stage is
    remapped; the intermediate stage (if any) is unaffected.

    ``<name>_numeral_style`` records which numeral style was drawn for the
    row: ``"native"`` if ``config.native_numeral_probability`` was hit (see
    ``_translator_for``), otherwise ``"none"``. It reflects the draw shared
    by the intermediate and final stage, not just the final stage.
    Evaluation always uses ``config.evaluation_prompt`` regardless of which
    style was drawn.
    """

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._merge_model = None
        if self.config.enable_windowing:
            merge_alias = self.config.window_merge_model_alias or self.config.model_alias
            self._merge_model = (
                self.model if merge_alias == self.config.model_alias else self.get_model(model_alias=merge_alias)
            )
        self._translator = Translator(
            self.config.translation_prompt,
            enable_windowing=self.config.enable_windowing,
            window_separator=self.config.window_separator,
            window_tokenizer_model=self.config.window_tokenizer_model,
            window_tokens=self.config.window_tokens,
            window_overlap=self.config.window_overlap,
            window_max_concurrent=self.config.window_max_concurrent,
            keep_window_traces=self.config.keep_window_traces,
            merge_model=self._merge_model,
            merge_prompt=self.config.window_merge_prompt,
        )
        self._evaluator = Evaluator(self.config.evaluation_prompt)
        self._rewriter = Rewriter(self.config.rewrite_prompt)
        # Lazily built/cached (target_language_code -> Translator) for the native-numeral
        # translation prompt, one per distinct target language code actually requested -- see
        # _translator_for. Empty (and unused) when native_numeral_probability is 0. Evaluation
        # always uses self._evaluator regardless of numeral style -- there is no native-numeral
        # evaluation prompt variant.
        self._native_numeral_translators: dict[str, Translator] = {}

    @functools.cached_property
    def evaluator_model(self):
        """Model used for the MQM evaluation step, separate from translation/rewrite."""
        return self.get_model(model_alias=self.config.evaluator_model_alias)

    def _translator_for(self, target_language_code: str, use_native_numerals: bool) -> Translator:
        """Returns the ``Translator`` to use for one stage's translate step.

        ``use_native_numerals=False`` returns the plain ``self._translator``
        built from ``config.translation_prompt``. ``use_native_numerals=True``
        returns a ``Translator`` built from ``config.translation_prompt_native_numerals``
        with ``target_language_code``'s numeral legend resolved in, cached
        per ``target_language_code`` since the intermediate and final stage
        usually target different languages.
        """
        if not use_native_numerals:
            return self._translator
        if target_language_code not in self._native_numeral_translators:
            legend = native_numbers_doc(target_language_code)
            self._native_numeral_translators[target_language_code] = Translator(
                self.config.translation_prompt_native_numerals.replace("{native_numbers_doc}", legend),
                enable_windowing=self.config.enable_windowing,
                window_separator=self.config.window_separator,
                window_tokenizer_model=self.config.window_tokenizer_model,
                window_tokens=self.config.window_tokens,
                window_overlap=self.config.window_overlap,
                window_max_concurrent=self.config.window_max_concurrent,
                keep_window_traces=self.config.keep_window_traces,
                merge_model=self._merge_model,
                merge_prompt=self.config.window_merge_prompt,
            )
        return self._native_numeral_translators[target_language_code]

    def _describe_language(self, code: str) -> str:
        """Looks up ``code``'s natural-language description.

        Raises:
            KeyError: If ``code`` isn't covered by
                ``config.language_descriptions``.
        """
        try:
            return self.config.language_descriptions[code]
        except KeyError as exc:
            raise KeyError(
                f"No description for language code {code!r} in config.language_descriptions. "
                "Add an entry for it (see DocumentTranslationConfig.language_descriptions)."
            ) from exc

    @functools.cached_property
    def target_language_description(self) -> str:
        """Natural-language description of ``config.target_language``, computed once."""
        return self._describe_language(self.config.target_language)

    @functools.cached_property
    def source_language_description(self) -> str:
        """Natural-language description of ``config.source_language``, computed once."""
        return self._describe_language(self.config.source_language)

    @functools.cached_property
    def remapped_target_language_description(self) -> str | None:
        """Natural-language description of ``config.remapped_target_language``, computed once.

        ``None`` when ``config.target_language_script`` is unset.
        """
        if self.config.remapped_target_language is None:
            return None
        return self._describe_language(self.config.remapped_target_language)

    def _remap_script(
        self,
        translation: str,
        evaluation: TranslationEvaluation,
        translation_history: list[str],
        evaluation_history: list[dict],
        text_source: str,
        source_language_description: str,
    ) -> tuple[str, TranslationEvaluation, list[str], list[dict]]:
        """Deterministically remaps the final translation into ``config.target_language_script``.

        The pre-remap translation/evaluation are pushed onto history (same
        as a superseded rewrite attempt), the translation is remapped, and
        a fresh evaluation scores the remapped text against ``text_source``
        -- this becomes the returned (current) evaluation. Callers must
        only call this when ``config.target_language_script`` is set --
        see ``generate``/``agenerate``.
        """
        _, source_script, _ = parse_bcp47(self.config.target_language)
        remapped = remap_script(
            translation,
            source_script,
            self.config.target_language_script,
            preserve_protected_spans=self.config.remap_preserves_protected_spans,
        )
        remapped_evaluation = self._evaluator.evaluate(
            self.evaluator_model,
            text_source,
            remapped,
            source_language_description,
            self.remapped_target_language_description,
            self.config.max_correction_steps,
        )
        return (
            remapped,
            remapped_evaluation,
            [*translation_history, translation],
            [*evaluation_history, evaluation.model_dump()],
        )

    async def _aremap_script(
        self,
        translation: str,
        evaluation: TranslationEvaluation,
        translation_history: list[str],
        evaluation_history: list[dict],
        text_source: str,
        source_language_description: str,
    ) -> tuple[str, TranslationEvaluation, list[str], list[dict]]:
        """Async version of :meth:`_remap_script`."""
        _, source_script, _ = parse_bcp47(self.config.target_language)
        remapped = remap_script(
            translation,
            source_script,
            self.config.target_language_script,
            preserve_protected_spans=self.config.remap_preserves_protected_spans,
        )
        remapped_evaluation = await self._evaluator.aevaluate(
            self.evaluator_model,
            text_source,
            remapped,
            source_language_description,
            self.remapped_target_language_description,
            self.config.max_correction_steps,
        )
        return (
            remapped,
            remapped_evaluation,
            [*translation_history, translation],
            [*evaluation_history, evaluation.model_dump()],
        )

    def _run_stage(
        self,
        text_source: str,
        source_language_description: str,
        target_language_description: str,
        max_rewrites: int,
        rewrite_threshold: int,
        record: dict,
        target_language_code: str,
        use_native_numerals: bool,
    ) -> tuple[str, TranslationEvaluation, list[str], list[dict], list[dict[str, str | None]] | None]:
        """Runs one full translate-evaluate-rewrite procedure (sync)."""
        translator = self._translator_for(target_language_code, use_native_numerals)
        translation, window_traces = translator.translate(
            self.model,
            text_source,
            source_language_description,
            target_language_description,
            self.config.max_correction_steps,
            record=record,
        )

        evaluation = self._evaluator.evaluate(
            self.evaluator_model,
            text_source,
            translation,
            source_language_description,
            target_language_description,
            self.config.max_correction_steps,
        )

        translation_history: list[str] = []
        evaluation_history: list[dict] = []
        rewrites_done = 0
        while evaluation.overall_score < rewrite_threshold and rewrites_done < max_rewrites:
            translation_history.append(translation)
            evaluation_history.append(evaluation.model_dump())

            translation = self._rewriter.rewrite(
                self.model,
                text_source,
                translation,
                evaluation,
                source_language_description,
                target_language_description,
                self.config.max_correction_steps,
            )

            evaluation = self._evaluator.evaluate(
                self.evaluator_model,
                text_source,
                translation,
                source_language_description,
                target_language_description,
                self.config.max_correction_steps,
            )
            rewrites_done += 1

        return translation, evaluation, translation_history, evaluation_history, window_traces

    async def _arun_stage(
        self,
        text_source: str,
        source_language_description: str,
        target_language_description: str,
        max_rewrites: int,
        rewrite_threshold: int,
        record: dict,
        target_language_code: str,
        use_native_numerals: bool,
    ) -> tuple[str, TranslationEvaluation, list[str], list[dict], list[dict[str, str | None]] | None]:
        """Runs one full translate-evaluate-rewrite procedure (async). See ``_run_stage``."""
        translator = self._translator_for(target_language_code, use_native_numerals)
        translation, window_traces = await translator.atranslate(
            self.model,
            text_source,
            source_language_description,
            target_language_description,
            self.config.max_correction_steps,
            record=record,
        )

        evaluation = await self._evaluator.aevaluate(
            self.evaluator_model,
            text_source,
            translation,
            source_language_description,
            target_language_description,
            self.config.max_correction_steps,
        )

        translation_history: list[str] = []
        evaluation_history: list[dict] = []
        rewrites_done = 0
        while evaluation.overall_score < rewrite_threshold and rewrites_done < max_rewrites:
            translation_history.append(translation)
            evaluation_history.append(evaluation.model_dump())

            translation = await self._rewriter.arewrite(
                self.model,
                text_source,
                translation,
                evaluation,
                source_language_description,
                target_language_description,
                self.config.max_correction_steps,
            )

            evaluation = await self._evaluator.aevaluate(
                self.evaluator_model,
                text_source,
                translation,
                source_language_description,
                target_language_description,
                self.config.max_correction_steps,
            )
            rewrites_done += 1

        return translation, evaluation, translation_history, evaluation_history, window_traces

    @functools.cached_property
    def intermediate_language_description(self) -> str | None:
        """Natural-language description of ``config.intermediate_language``, computed once.

        ``None`` when ``config.intermediate_language`` is unset.
        """
        if self.config.intermediate_language is None:
            return None
        return self._describe_language(self.config.intermediate_language)

    @functools.cached_property
    def uses_intermediate(self) -> bool:
        """Whether the intermediate pivot stage runs for this column."""
        return (
            self.config.intermediate_language is not None
            and self.config.intermediate_language != self.config.target_language
        )

    def _write_output(
        self,
        data: dict,
        translation: str,
        evaluation: TranslationEvaluation,
        translation_history: list[str],
        evaluation_history: list[dict],
        window_traces: list[dict[str, str | None]] | None,
        intermediate_translation: str | None,
        intermediate_evaluation: TranslationEvaluation | None,
        intermediate_translation_history: list[str],
        intermediate_evaluation_history: list[dict],
        intermediate_window_traces: list[dict[str, str | None]] | None,
        use_native_numerals: bool,
    ) -> dict:
        data[self.config.name] = translation
        data[f"{self.config.name}_evaluation"] = evaluation.model_dump()
        data[f"{self.config.name}_overall_score"] = evaluation.overall_score
        data[f"{self.config.name}_translation_history"] = translation_history
        data[f"{self.config.name}_evaluation_history"] = evaluation_history
        data[f"{self.config.name}_window_traces"] = window_traces
        data[f"{self.config.name}_numeral_style"] = "native" if use_native_numerals else "none"
        data[f"{self.config.name}_intermediate"] = intermediate_translation
        data[f"{self.config.name}_intermediate_evaluation"] = (
            intermediate_evaluation.model_dump() if intermediate_evaluation is not None else None
        )
        data[f"{self.config.name}_intermediate_overall_score"] = (
            intermediate_evaluation.overall_score if intermediate_evaluation is not None else None
        )
        data[f"{self.config.name}_intermediate_translation_history"] = intermediate_translation_history
        data[f"{self.config.name}_intermediate_evaluation_history"] = intermediate_evaluation_history
        data[f"{self.config.name}_intermediate_window_traces"] = intermediate_window_traces
        return data

    def generate(self, data: dict) -> dict:
        text = data[self.config.text_column]
        source_language_description = self.source_language_description
        target_language_description = self.target_language_description
        use_native_numerals = random.random() < self.config.native_numeral_probability

        if self.uses_intermediate:
            intermediate_language_description = self.intermediate_language_description
            (
                intermediate_translation,
                intermediate_evaluation,
                intermediate_translation_history,
                intermediate_evaluation_history,
                intermediate_window_traces,
            ) = self._run_stage(
                text,
                source_language_description,
                intermediate_language_description,
                self.config.max_rewrites_intermediate,
                self.config.rewrite_threshold_intermediate,
                data,
                self.config.intermediate_language,
                use_native_numerals,
            )
            final_text_source = intermediate_translation
            final_source_language_description = intermediate_language_description
        else:
            intermediate_translation = None
            intermediate_evaluation = None
            intermediate_translation_history = []
            intermediate_evaluation_history = []
            intermediate_window_traces = None
            final_text_source = text
            final_source_language_description = source_language_description

        translation, evaluation, translation_history, evaluation_history, window_traces = self._run_stage(
            final_text_source,
            final_source_language_description,
            target_language_description,
            self.config.max_rewrites,
            self.config.rewrite_threshold,
            data,
            self.config.target_language,
            use_native_numerals,
        )
        if self.config.target_language_script is not None:
            translation, evaluation, translation_history, evaluation_history = self._remap_script(
                translation,
                evaluation,
                translation_history,
                evaluation_history,
                final_text_source,
                final_source_language_description,
            )

        return self._write_output(
            data,
            translation,
            evaluation,
            translation_history,
            evaluation_history,
            window_traces,
            intermediate_translation,
            intermediate_evaluation,
            intermediate_translation_history,
            intermediate_evaluation_history,
            intermediate_window_traces,
            use_native_numerals,
        )

    async def agenerate(self, data: dict) -> dict:
        text = data[self.config.text_column]
        source_language_description = self.source_language_description
        target_language_description = self.target_language_description
        use_native_numerals = random.random() < self.config.native_numeral_probability

        if self.uses_intermediate:
            intermediate_language_description = self.intermediate_language_description
            (
                intermediate_translation,
                intermediate_evaluation,
                intermediate_translation_history,
                intermediate_evaluation_history,
                intermediate_window_traces,
            ) = await self._arun_stage(
                text,
                source_language_description,
                intermediate_language_description,
                self.config.max_rewrites_intermediate,
                self.config.rewrite_threshold_intermediate,
                data,
                self.config.intermediate_language,
                use_native_numerals,
            )
            final_text_source = intermediate_translation
            final_source_language_description = intermediate_language_description
        else:
            intermediate_translation = None
            intermediate_evaluation = None
            intermediate_translation_history = []
            intermediate_evaluation_history = []
            intermediate_window_traces = None
            final_text_source = text
            final_source_language_description = source_language_description

        translation, evaluation, translation_history, evaluation_history, window_traces = await self._arun_stage(
            final_text_source,
            final_source_language_description,
            target_language_description,
            self.config.max_rewrites,
            self.config.rewrite_threshold,
            data,
            self.config.target_language,
            use_native_numerals,
        )
        if self.config.target_language_script is not None:
            translation, evaluation, translation_history, evaluation_history = await self._aremap_script(
                translation,
                evaluation,
                translation_history,
                evaluation_history,
                final_text_source,
                final_source_language_description,
            )

        return self._write_output(
            data,
            translation,
            evaluation,
            translation_history,
            evaluation_history,
            window_traces,
            intermediate_translation,
            intermediate_evaluation,
            intermediate_translation_history,
            intermediate_evaluation_history,
            intermediate_window_traces,
            use_native_numerals,
        )
