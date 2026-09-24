# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column config for the Nemotron Bharat document-translation plugin.

:class:`DocumentTranslationConfig` -- direct translate + MQM evaluate +
optional rewrite loop. Translation optionally runs through overlapping
token windows for long documents (see ``enable_windowing`` and the
``window_*`` fields); evaluation and rewrite always operate on the whole
(possibly windowed-and-merged) document. Optionally finishes with a
deterministic script remap (see ``target_language_script``) -- e.g.
Bengali-script Manipuri into Meetei Mayek -- via ``script_remap.py``.
"""

import re
import warnings
from typing import Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import Field, model_validator

from data_designer_nemotron_bharat_translation.language_codes import (
    BODHAN_LANGUAGE_DESCRIPTIONS,
    GENERIC_LANGUAGE_DESCRIPTIONS,
    parse_bcp47,
)
from data_designer_nemotron_bharat_translation.prompts import (
    BODHAN_TRANSLATION_PROMPT,
    GENERIC_EVALUATION_PROMPT,
    GENERIC_REWRITE_PROMPT,
    GENERIC_TRANSLATION_PROMPT,
    GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS,
    MERGE_PROMPT,
    NATIVE_NUMBERS_DOC,
)
from data_designer_nemotron_bharat_translation.script_remap import SCRIPT_REMAPPERS, can_remap_script

_TRANSLATION_MODE_PROMPTS = {"generic": GENERIC_TRANSLATION_PROMPT, "bodhan": BODHAN_TRANSLATION_PROMPT}
_TRANSLATION_MODE_LANGUAGE_DESCRIPTIONS = {
    "generic": GENERIC_LANGUAGE_DESCRIPTIONS,
    "bodhan": BODHAN_LANGUAGE_DESCRIPTIONS,
}


def _apply_translation_mode(
    translation_mode: Literal["generic", "bodhan"],
    translation_prompt: str | None,
    language_descriptions: dict[str, str] | None,
) -> tuple[str, dict[str, str], bool]:
    """Resolves ``translation_prompt``/``language_descriptions`` against ``translation_mode``'s defaults.

    Returns ``(resolved_prompt, resolved_language_descriptions, was_overridden)``.
    ``was_overridden`` is ``False`` both when neither field was set and when
    re-validating a config that's already been resolved to its mode's own
    defaults -- e.g. the second ``model_validate()`` ``ConfigurableTask.__init__``
    performs when wrapping a config into a generator -- so callers don't
    mistake their own prior defaulting for a user override.
    """
    mode_prompt = _TRANSLATION_MODE_PROMPTS[translation_mode]
    mode_descriptions = _TRANSLATION_MODE_LANGUAGE_DESCRIPTIONS[translation_mode]
    prompt_overridden = translation_prompt is not None and translation_prompt != mode_prompt
    descriptions_overridden = language_descriptions is not None and language_descriptions != mode_descriptions
    resolved_prompt = translation_prompt if translation_prompt is not None else mode_prompt
    resolved_descriptions = language_descriptions if language_descriptions is not None else dict(mode_descriptions)
    return resolved_prompt, resolved_descriptions, prompt_overridden or descriptions_overridden


class DocumentTranslationConfig(SingleColumnConfig):
    """Translates ``text_column`` from ``source_language`` to ``target_language``, then scores it.

    Writes ``<name>`` (translation), ``<name>_evaluation`` (MQM dict), and
    ``<name>_overall_score``. If ``max_rewrites > 0``, a self-correction
    loop reruns translate+evaluate while the score stays below
    ``rewrite_threshold``, keeping superseded attempts in
    ``<name>_translation_history``/``<name>_evaluation_history``.

    If ``intermediate_language`` is set and differs from ``target_language``,
    the whole procedure runs twice -- source -> intermediate (using the
    ``*_intermediate`` rewrite settings, results in ``<name>_intermediate*``),
    then intermediate -> target (results in ``<name>``/``<name>_evaluation``,
    which always hold the final stage). Useful as a pivot (e.g. native script
    first, then romanize). Skipped when unset or equal to ``target_language``.

    If ``enable_windowing`` is set, each translate step (both the
    intermediate and final stage, when both run) splits its input into
    overlapping token-bounded windows (sized by the ``window_*`` fields),
    translates each window, and merges them back into one document before
    evaluation -- when ``keep_window_traces`` is also set (the default),
    one dict per window (``"window"``, ``"translated_window"``,
    ``"extracted_window"``, ``"extracted_translated_window"``; see
    :class:`~data_designer_nemotron_bharat_translation.translator.Translator`)
    is written to ``<name>_window_traces``/``<name>_intermediate_window_traces``.
    Evaluation always runs over the whole document; only the initial
    translation is windowed. The rewrite loop is incompatible with
    windowing (it would rewrite the whole merged document in one shot,
    defeating the point of windowing it in the first place) -- setting
    ``enable_windowing`` alongside nonzero ``max_rewrites``/
    ``max_rewrites_intermediate`` forces both to 0 and emits a
    ``UserWarning``. ``<name>_window_traces``/``<name>_intermediate_window_traces``
    are ``None`` when ``enable_windowing`` is unset, or when
    ``keep_window_traces`` is ``False``.

    If ``target_language_script`` is set (a 4-letter ISO 15924 script code,
    e.g. ``"Mtei"``), the final stage's translation is deterministically
    remapped into that script after the translate-evaluate-rewrite
    procedure finishes -- e.g. translating Manipuri into Bengali script
    (``target_language="mni-Beng-IN"``) and remapping into Meetei Mayek
    (``target_language_script="Mtei"``), since the translation model has no
    way to produce Meetei Mayek directly. Only
    :data:`~data_designer_nemotron_bharat_translation.script_remap.SCRIPT_REMAPPERS`
    pairs are supported (currently Bengali -> Meetei Mayek only); an
    unsupported ``(target_language, target_language_script)`` pair raises
    ``ValueError`` at config construction time. The pre-remap translation
    and evaluation are pushed onto ``<name>_translation_history``/
    ``<name>_evaluation_history`` (same as a superseded rewrite attempt),
    the translation is remapped, and a fresh evaluation scores the
    remapped text against the same source text -- that becomes
    ``<name>``/``<name>_evaluation``/``<name>_overall_score``. Remapping
    only ever applies to the final stage; the intermediate stage (if any)
    is unaffected. The remapper is character-level and has no notion of
    code vs. prose, so by default (``remap_preserves_protected_spans=True``)
    fenced code blocks, inline code, and URLs are left untouched during
    remapping -- only the surrounding prose is remapped -- so that source-
    script characters inside a code literal or URL (which the translation
    step already promised to preserve byte-for-byte) aren't silently
    corrupted. Set ``remap_preserves_protected_spans=False`` to remap the whole
    text unconditionally.

    ``native_numeral_probability`` (default 0.0, matching the source
    prototype's ``NATIVE_NUMERAL_PCT = 0.5`` pattern but opted out by
    default) is the per-row chance of translating with
    ``translation_prompt_native_numerals`` (numbers in prose written in the
    target language's native numeral script) instead of the plain
    ``translation_prompt``, drawn independently per row but shared between
    the intermediate and final stage within that row. Evaluation always
    uses ``evaluation_prompt``, regardless of which style was drawn for
    translation -- there is no native-numeral evaluation prompt variant.
    ``translation_prompt_native_numerals`` defaults to
    :data:`~data_designer_nemotron_bharat_translation.prompts.GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS`
    but can be overridden, e.g. to keep a customized ``translation_prompt``
    consistent with its native-numeral counterpart -- overriding
    ``translation_prompt`` without also overriding
    ``translation_prompt_native_numerals`` while ``native_numeral_probability``
    is set emits a ``UserWarning``, since native-numeral rows would
    otherwise silently diverge from the customized ASCII prompt. Only
    ``"generic"`` ``translation_mode`` supports native numerals at all --
    ``bodhan-ai/indic-translate``'s minimal prompt has no native-numeral variant, so
    ``translation_mode='bodhan'`` forces ``native_numeral_probability`` to
    0.0 (with a ``UserWarning``). ``translation_prompt_native_numerals``
    must contain a literal ``{native_numbers_doc}`` placeholder (single
    braces) for the per-language legend, and ``target_language`` (and
    ``intermediate_language``, if set) must have an entry in
    :data:`~data_designer_nemotron_bharat_translation.prompts.NATIVE_NUMBERS_DOC`
    -- both raise ``ValueError`` at config construction time when
    ``native_numeral_probability`` is set and violated. Which style was
    drawn for a row is recorded in ``<name>_numeral_style`` (``"native"``
    or ``"none"``).
    """

    column_type: Literal["document-translation"] = "document-translation"
    model_alias: str
    evaluator_model_alias: str
    text_column: str = "text"
    target_language: str
    source_language: str

    #: Selects the default ``translation_prompt``/``language_descriptions``:
    #: ``"generic"`` (default) for general-purpose instruction-tuned
    #: translator models; ``"bodhan"`` for the ``bodhan-ai/indic-translate``
    #: specialist. Only affects the defaults below -- explicitly setting
    #: either field overrides this regardless of mode.
    translation_mode: Literal["generic", "bodhan"] = "generic"

    language_descriptions: dict[str, str] | None = None
    translation_prompt: str | None = None
    evaluation_prompt: str = GENERIC_EVALUATION_PROMPT
    rewrite_prompt: str = GENERIC_REWRITE_PROMPT
    max_rewrites: int = 0
    rewrite_threshold: int = 75
    intermediate_language: str | None = None
    max_rewrites_intermediate: int = 0
    rewrite_threshold_intermediate: int = 75
    max_correction_steps: int = 2

    #: When set, the translate step runs through overlapping token windows instead of one shot.
    enable_windowing: bool = False
    #: Chunk boundary the source text is split on.
    window_separator: str = "\n"
    #: Model used to measure token windows.
    window_tokenizer_model: str = "google/gemma-4-31b-it"
    #: Max tokens per window (a window always holds >= 2 chunks, even over budget).
    window_tokens: int = Field(default=2048, ge=1)
    #: Chunks shared between consecutive windows.
    window_overlap: int = Field(default=1, ge=0)
    #: Cap on how many of one document's windows are translated/merged concurrently (async path only).
    window_max_concurrent: int = Field(default=8, ge=1)
    #: Segment-extraction model alias. Defaults to ``model_alias`` when unset.
    window_merge_model_alias: str | None = None
    window_merge_prompt: str = MERGE_PROMPT
    #: Whether to keep and write out the per-window trace data. Only relevant when ``enable_windowing`` is set.
    keep_window_traces: bool = True

    #: A 4-letter ISO 15924 script code (e.g. "Mtei") to deterministically remap the final
    #: translation into after translate-evaluate-rewrite. See the class docstring.
    target_language_script: str | None = None
    #: Whether script remapping leaves fenced code blocks, inline code, and URLs untouched.
    #: Only relevant when target_language_script is set. See the class docstring.
    remap_preserves_protected_spans: bool = True

    #: Per-row chance [0, 1] of using native-script numerals instead of ASCII digits in prose.
    #: See the class docstring.
    native_numeral_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Native-numeral counterpart of translation_prompt. Defaults to GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS.
    translation_prompt_native_numerals: str | None = None

    @model_validator(mode="after")
    def _validate_target_language_script(self) -> "DocumentTranslationConfig":
        if self.target_language_script is None:
            return self
        if not re.fullmatch(r"[A-Za-z]{4}", self.target_language_script):
            raise ValueError(
                f"target_language_script {self.target_language_script!r} must be a 4-letter ISO 15924 "
                "script code (e.g. 'Mtei')."
            )
        self.target_language_script = self.target_language_script[0].upper() + self.target_language_script[1:].lower()
        _, source_script, _ = parse_bcp47(self.target_language)
        if not can_remap_script(source_script, self.target_language_script):
            supported = ", ".join(f"{src!r} -> {dst!r}" for src, dst in sorted(SCRIPT_REMAPPERS))
            raise ValueError(
                f"No script remapper for {source_script!r} -> {self.target_language_script!r} "
                f"(target_language={self.target_language!r}). Supported: {supported}."
            )
        return self

    @property
    def remapped_target_language(self) -> str | None:
        """The BCP-47 code for the remapped script, or ``None`` when ``target_language_script`` is unset."""
        if self.target_language_script is None:
            return None
        lang, _script, region = parse_bcp47(self.target_language)
        return f"{lang}-{self.target_language_script}-{region}"

    @model_validator(mode="after")
    def _apply_translation_mode_defaults(self) -> "DocumentTranslationConfig":
        self.translation_prompt, self.language_descriptions, overridden = _apply_translation_mode(
            self.translation_mode, self.translation_prompt, self.language_descriptions
        )
        if self.translation_mode == "bodhan" and overridden:
            warnings.warn(
                "translation_mode='bodhan' works best with the default bodhan-ai/indic-translate "
                "configuration -- leave translation_prompt and language_descriptions unset.",
                stacklevel=2,
            )

        # bodhan-ai/indic-translate is a narrow translation specialist that cannot follow the
        # multi-step rewrite_prompt self-correction instructions -- force the
        # rewrite loop off rather than let it burn extra calls for no benefit.
        if self.translation_mode == "bodhan" and (self.max_rewrites != 0 or self.max_rewrites_intermediate != 0):
            warnings.warn(
                "translation_mode='bodhan' does not support the rewrite loop. "
                "Forcing max_rewrites and max_rewrites_intermediate to 0.",
                stacklevel=2,
            )
            self.max_rewrites = 0
            self.max_rewrites_intermediate = 0

        # bodhan-ai/indic-translate's minimal prompt contract has no native-numeral instructions,
        # so it can't honor native_numeral_probability -- force it off rather than silently ignore it.
        if self.translation_mode == "bodhan" and self.native_numeral_probability != 0.0:
            warnings.warn(
                "translation_mode='bodhan' has no native-numeral prompt variant. "
                "Forcing native_numeral_probability to 0.0.",
                stacklevel=2,
            )
            self.native_numeral_probability = 0.0

        # Comparing against the resolved default (not just "is not None") matters here for the
        # same reason as _apply_translation_mode's was_overridden: ConfigurableTask.__init__
        # re-validates the config a second time, by which point an unset field already holds
        # its resolved default -- "is not None" alone would misread that as a user override.
        translation_prompt_overridden = self.translation_prompt != _TRANSLATION_MODE_PROMPTS[self.translation_mode]
        translation_native_numerals_overridden = (
            self.translation_prompt_native_numerals is not None
            and self.translation_prompt_native_numerals != GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS
        )
        if self.translation_prompt_native_numerals is None:
            self.translation_prompt_native_numerals = GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS

        if self.native_numeral_probability > 0.0:
            # A customized translation_prompt with translation_prompt_native_numerals left at
            # its generic default would silently diverge for native-numeral rows.
            if translation_prompt_overridden and not translation_native_numerals_overridden:
                warnings.warn(
                    "translation_prompt is customized but translation_prompt_native_numerals is not -- "
                    "rows drawn for native numerals will use the generic native-numeral prompt instead "
                    "of your custom translation_prompt. Pass translation_prompt_native_numerals to keep "
                    "them consistent.",
                    stacklevel=2,
                )
            if "{native_numbers_doc}" not in self.translation_prompt_native_numerals:
                raise ValueError(
                    "translation_prompt_native_numerals must contain a literal '{native_numbers_doc}' "
                    "placeholder to receive the per-language numeral legend."
                )
            for code in filter(None, [self.target_language, self.intermediate_language]):
                if code not in NATIVE_NUMBERS_DOC:
                    raise ValueError(
                        f"native_numeral_probability is set but {code!r} has no entry in "
                        f"NATIVE_NUMBERS_DOC. Covered languages: {sorted(NATIVE_NUMBERS_DOC)}."
                    )

        # The rewrite loop rewrites the whole (already merged) document in one shot -- for a
        # windowed document that's exactly the giant single-call translation windowing exists
        # to avoid, so force it off rather than let it undo windowing's benefit.
        if self.enable_windowing and (self.max_rewrites != 0 or self.max_rewrites_intermediate != 0):
            warnings.warn(
                "enable_windowing is set -- the rewrite loop operates on the whole merged "
                "document, which defeats the purpose of windowing. Forcing max_rewrites and "
                "max_rewrites_intermediate to 0.",
                stacklevel=2,
            )
            self.max_rewrites = 0
            self.max_rewrites_intermediate = 0
        return self

    @staticmethod
    def get_column_emoji() -> str:
        return "🌐"

    def get_model_aliases(self) -> list[str]:
        aliases = [self.model_alias, self.evaluator_model_alias]
        if self.enable_windowing:
            aliases.append(self.window_merge_model_alias or self.model_alias)
        return list(dict.fromkeys(aliases))

    @property
    def required_columns(self) -> list[str]:
        return [self.text_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return [
            f"{self.name}_evaluation",
            f"{self.name}_overall_score",
            f"{self.name}_translation_history",
            f"{self.name}_evaluation_history",
            f"{self.name}_window_traces",
            f"{self.name}_intermediate",
            f"{self.name}_intermediate_evaluation",
            f"{self.name}_intermediate_overall_score",
            f"{self.name}_intermediate_translation_history",
            f"{self.name}_intermediate_evaluation_history",
            f"{self.name}_intermediate_window_traces",
            f"{self.name}_numeral_style",
        ]
