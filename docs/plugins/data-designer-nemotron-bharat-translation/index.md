# data-designer-nemotron-bharat-translation

A document translation column generator plugin for Nemotron Bharat's Indic
pretraining data pipelines.

## Installation

```bash
uv add data-designer data-designer-nemotron-bharat-translation
```

## Column types

### `document-translation`

Translates a column into a single fixed target language, scores the result
with an MQM-based evaluator, and optionally self-corrects via a rewrite
loop. Supports an optional two-stage pivot through an intermediate language
(e.g. translating into native script first, then romanizing).

Translation itself can run two ways, controlled by `enable_windowing`:

- **`False` (default)**: one model call translates the whole document.
- **`True`**: the document is split into overlapping token-bounded windows
  (sized by the `window_*` fields below), each window is translated
  independently, then each window's own non-overlapping segment is
  extracted (with its neighbours shown for context, so extraction calls run
  concurrently) and the segments are stacked back into one document. This
  is for long documents that degrade whole-document translation quality
  (omissions, splicing, timeouts). Evaluation always runs on the whole
  (merged) document either way -- only the initial translation step is
  windowed. The rewrite loop is incompatible with windowing (it would
  rewrite the whole merged document in one shot, defeating the point of
  windowing it) -- setting `enable_windowing=True` alongside nonzero
  `max_rewrites`/`max_rewrites_intermediate` forces both to 0 and emits a
  `UserWarning`.

  When `keep_window_traces` is also `True` (the default), one dict per
  window -- `window` (source text), `translated_window` (that window's raw
  translation before extraction), `extracted_window`/`extracted_translated_window`
  (its own non-overlapping source/translated segment after extraction,
  both `null` for a window that contributes nothing to the merged
  document) -- is written to `<name>_window_traces`/`<name>_intermediate_window_traces`.
  Both are `None` when `enable_windowing` is `False`, or when
  `keep_window_traces` is `False` (windowing/merging still run either way --
  only whether the trace data is kept changes).

When `target_language_script` is set (a 4-letter ISO 15924 script code,
e.g. `Mtei`), the final stage's translation is deterministically remapped
into that script once the translate-evaluate-rewrite procedure finishes --
e.g. translate Manipuri into Bengali script (`target_language="mni-Beng-IN"`)
and remap into Meetei Mayek (`target_language_script="Mtei"`), since no
translation model in this pipeline produces Meetei Mayek directly. Only
the script pairs in `script_remap.SCRIPT_REMAPPERS` are supported
(currently Bengali -> Meetei Mayek only); an unsupported pair raises a
validation error when the config is constructed. The pre-remap
translation/evaluation are pushed onto `<name>_translation_history`/
`<name>_evaluation_history` (same as a superseded rewrite attempt), the
translation is remapped, and a fresh evaluation scores the remapped text
against the same source text -- that becomes `<name>`/`<name>_evaluation`/
`<name>_overall_score`. Remapping only ever applies to the final stage;
the intermediate stage (if any) is unaffected. The remapper is
character-level and has no notion of code vs. prose, so by default
(`remap_preserves_protected_spans=True`) fenced code blocks, inline code, and
URLs are left untouched during remapping -- only the surrounding prose is
remapped -- so that source-script characters inside a code literal or URL
(which the translation step already promised to preserve byte-for-byte)
aren't silently corrupted. Set `remap_preserves_protected_spans=False` to remap
the whole text unconditionally.

`native_numeral_probability` (default `0.0`, matching the source
prototype's `NATIVE_NUMERAL_PCT = 0.5` pattern but opted out by default) is
the per-row chance of translating with `translation_prompt_native_numerals`
(numbers in prose written in the target language's native numeral script
instead of ASCII digits) instead of the plain `translation_prompt` --
drawn independently per row, but shared between the intermediate and
final stage within that row. Digits inside code, formulas, URLs, and
LaTeX stay ASCII either way. Evaluation always uses `evaluation_prompt`,
regardless of which style was drawn for translation -- there is no
native-numeral evaluation prompt variant. Only `"generic"`
`translation_mode` supports native numerals at all --
`bodhan-ai/indic-translate`'s minimal prompt has no native-numeral
variant, so `translation_mode="bodhan"` forces
`native_numeral_probability` to `0.0` (with a `UserWarning`).
Requires `target_language` (and `intermediate_language`, if set) to have
an entry in `prompts.NATIVE_NUMBERS_DOC`; an uncovered language raises a
validation error when the config is constructed.
`translation_prompt_native_numerals` (defaulting to
`GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS`) lets you override the
native-numeral prompt the same way `translation_prompt` overrides the
plain one; it must keep the literal `{native_numbers_doc}` placeholder
(resolved via `str.replace()` before Jinja rendering) or config
construction raises a validation error. Overriding `translation_prompt`
without also overriding `translation_prompt_native_numerals` emits a
`UserWarning`, since the native-numeral row would otherwise silently fall
back to the generic native-numeral prompt instead of your custom one.
Which style was drawn for a row is recorded in `<name>_numeral_style`
(`"native"` or `"none"`), shared between the intermediate and final
stage.

| Field | Required | Description |
| --- | --- | --- |
| `text_column` | No (default `text`) | Source text column. |
| `source_language` | Yes | BCP-47 source language code, fixed for the whole column. |
| `target_language` | Yes | BCP-47 target language code, fixed for the whole column. |
| `model_alias` | Yes | Translator model alias. |
| `evaluator_model_alias` | Yes | Evaluator model alias. Must be set explicitly, even to the same value as `model_alias`. |
| `translation_mode` | No (default `generic`) | `generic` or `bodhan`. Selects the default `translation_prompt`/`language_descriptions` (see below); explicitly setting either field overrides the mode's default for that field only. Overriding either field in `bodhan` mode emits a `UserWarning`, since `bodhan-ai/indic-translate` works best with its own default prompt/descriptions. |
| `translation_prompt` | No (default depends on `translation_mode`: `GENERIC_TRANSLATION_PROMPT` or `BODHAN_TRANSLATION_PROMPT`) | Translation prompt template; rendered with `text`, `source_language_description`, `target_language_description`. |
| `evaluation_prompt` | No (default `GENERIC_EVALUATION_PROMPT`) | MQM-based evaluation prompt template. |
| `rewrite_prompt` | No (default `GENERIC_REWRITE_PROMPT`) | Self-correction rewrite prompt template. |
| `language_descriptions` | No (default depends on `translation_mode`: `GENERIC_LANGUAGE_DESCRIPTIONS` or `BODHAN_LANGUAGE_DESCRIPTIONS`) | Maps BCP-47 codes to natural-language descriptions used in prompts. |
| `intermediate_language` | No | Fixed BCP-47 code for an optional pivot stage. |
| `max_rewrites` | No (default 0) | Number of self-correction rewrite attempts. |
| `rewrite_threshold` | No (default 75) | MQM score below which a rewrite is triggered. |
| `max_rewrites_intermediate` | No (default 0) | Rewrite attempts for the intermediate stage. |
| `rewrite_threshold_intermediate` | No (default 75) | MQM score threshold for the intermediate stage. |
| `max_correction_steps` | No (default 2) | Correction rounds allowed per generation call. |
| `enable_windowing` | No (default `false`) | Enables windowed translation for long documents (see the `window_*` fields below). |
| `window_separator` | No (default `\n`) | Chunk boundary the source text is split on. |
| `window_tokenizer_model` | No (default `google/gemma-4-31b-it`) | Model used to measure token windows. |
| `window_tokens` | No (default 2048) | Max tokens per window (always >= 2 chunks). |
| `window_overlap` | No (default 1) | Chunks shared between consecutive windows. |
| `window_max_concurrent` | No (default 8) | Cap on windows translated/merged concurrently (async path). |
| `window_merge_model_alias` | No | Merge (segment-extraction) model alias; defaults to `model_alias`. |
| `window_merge_prompt` | No (default `MERGE_PROMPT`) | Segment-extraction prompt template. |
| `keep_window_traces` | No (default `true`) | Whether to keep and write out the per-window trace data (see above). Only relevant when `enable_windowing` is `true`. |
| `target_language_script` | No | 4-letter ISO 15924 script code to deterministically remap the final translation into (see above). Only `"Beng"` (implied by `target_language`) -> `"Mtei"` is currently supported. |
| `remap_preserves_protected_spans` | No (default `true`) | Whether script remapping leaves fenced code blocks, inline code, and URLs untouched (see above). Only relevant when `target_language_script` is set. |
| `native_numeral_probability` | No (default `0.0`) | Per-row chance (0 to 1) of using native-script numerals instead of ASCII digits in prose (see above). |
| `translation_prompt_native_numerals` | No (default `GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS`) | Translation prompt used for rows drawn into the native-numeral style; must keep the `{native_numbers_doc}` placeholder. |

For the full plugin authoring guide, see the
[main repository docs](https://nvidia-nemo.github.io/DataDesignerPlugins/authoring/).
