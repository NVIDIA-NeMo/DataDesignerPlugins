# data-designer-nemotron-bharat-translation

A document translation column generator plugin for Nemotron Bharat's Indic
pretraining data pipelines. It registers one `data_designer.plugins` entry
point, `document-translation`, that translates a text column into a fixed
target language, scores the result with an MQM-based evaluator, and
optionally self-corrects via a rewrite loop.

## Features

- **Translate + MQM evaluate + self-correct.** Each row is translated, then
  scored against an MQM rubric; while the score stays below
  `rewrite_threshold`, up to `max_rewrites` self-correction passes rerun
  translate+evaluate, keeping every superseded attempt in
  `<name>_translation_history`/`<name>_evaluation_history`.
- **Two translation modes.** `generic` targets general-purpose
  instruction-tuned models (e.g. Gemma) with a detailed rule-based prompt;
  `bodhan` is a minimal single-turn prompt tuned to
  [`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate)'s
  own documented reproducibility contract.
- **Optional intermediate-language pivot.** `intermediate_language` runs the
  whole translate-evaluate-rewrite procedure twice -- source language into
  the intermediate language, then that into the target language -- useful
  when a model translates better into a script it was tuned for before a
  second hop (e.g. Manipuri: English -> Bengali script -> Meetei Mayek).
- **Windowed translation for long documents.** `enable_windowing` splits the
  source into overlapping token-bounded windows, translates each
  independently, then extracts and stacks each window's own
  non-overlapping segment back into one document -- for documents long
  enough to degrade whole-document translation quality (omissions,
  splicing, timeouts). Evaluation always runs on the merged document.
- **Deterministic script remap.** `target_language_script` remaps the final
  translation into a different script than the translation model itself
  produces (currently Bengali -> Meetei Mayek only), then re-evaluates the
  remapped text against the source -- for scripts neither
  `google/gemma-4-31B-it` nor `bodhan-ai/indic-translate` can reliably
  generate directly, like Meetei Mayek. `remap_preserves_protected_spans`
  (default `True`) keeps fenced code blocks, inline code, and URLs out of
  the remap, since the character-level remapper has no notion of code vs.
  prose.
- **Native-numeral prose.** `native_numeral_probability` draws, per row, a
  chance of writing numbers in prose using the target language's native
  numeral script instead of ASCII digits (digits inside code, formulas,
  URLs, and LaTeX always stay ASCII). Which style was drawn is recorded in
  `<name>_numeral_style`.

## Supported languages and model selection

The plugin targets all 22 constitutionally scheduled Indic languages, plus
English, as BCP-47 `language-Script-Region` codes (see
`language_codes.TARGET_LANGUAGE_CODES`). The table below is the suggested
model selection used for Nemotron Bharat's own data preparation runs --
your own config can mix models differently via `model_alias`/
`evaluator_model_alias`/`translation_mode`.

| Language | Code | Translator | Evaluator | Notes |
| --- | --- | --- | --- | --- |
| Bengali | `bn-Beng-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Gujarati | `gu-Gujr-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Hindi | `hi-Deva-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Kannada | `kn-Knda-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Malayalam | `ml-Mlym-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Marathi | `mr-Deva-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Nepali | `ne-Deva-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Odia | `or-Orya-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Punjabi | `pa-Guru-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Tamil | `ta-Taml-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Telugu | `te-Telu-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Urdu | `ur-Arab-IN` | `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | `translation_mode="generic"` |
| Assamese | `as-Beng-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"` |
| Kashmiri | `ks-Arab-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"` |
| Sanskrit | `sa-Deva-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"` |
| Sindhi | `sd-Deva-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"` |
| Bodo | `brx-Deva-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"` |
| Dogri | `doi-Deva-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"` |
| Konkani | `gom-Deva-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"` |
| Maithili | `mai-Deva-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"` |
| Santali | `sat-Olck-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"`, `enable_windowing=True`, `window_tokens=2048` |
| Manipuri | `mni-Beng-IN` -> `mni-Mtei-IN` | `bodhan-ai/indic-translate` | `google/gemma-4-31B-it` | `translation_mode="bodhan"`, translates into Bengali script, then `target_language_script="Mtei"` deterministically remaps into Meetei Mayek |

- The suggested model for the first 12 languages (Bengali through Urdu) is
  `google/gemma-4-31B-it`, used for both translation and evaluation, in
  `generic` mode.
- The next 8 (Assamese through Maithili) use `bodhan-ai/indic-translate`
  for translation and `google/gemma-4-31B-it` for evaluation --
  `bodhan-ai/indic-translate` is a narrow translation specialist tuned to
  these languages' terse single-turn prompt contract, so
  `translation_mode="bodhan"` selects its minimal prompt instead of the
  generic rule-based one.
- Santali additionally enables windowed translation, since it tends to
  produce documents long enough to degrade whole-document translation
  quality.
- Manipuri is the one language that needs the script-remap + intermediate
  routing described above: neither `google/gemma-4-31B-it` nor
  `bodhan-ai/indic-translate` reliably generates Meetei Mayek script
  directly, so `bodhan-ai/indic-translate` instead translates into
  Bengali script and the result is deterministically remapped.

## Installation

```bash
uv add data-designer data-designer-nemotron-bharat-translation
```

## Usage

This package provides one Data Designer column generator plugin,
automatically discovered by Data Designer once installed. The four
examples below cover the four model-selection cases from the table above
-- adjust `model_alias`/`evaluator_model_alias` to whatever aliases your
`model_configs` register.

### `generic` mode (first 12 languages, e.g. Hindi)

```python
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer_nemotron_bharat_translation.config import DocumentTranslationConfig

builder = DataDesignerConfigBuilder()
builder.add_column(
    DocumentTranslationConfig(
        name="translation",
        text_column="text",
        source_language="en-Latn-IN",
        target_language="hi-Deva-IN",
        translation_mode="generic",
        model_alias="gemma",
        evaluator_model_alias="gemma",
    )
)
```

### `bodhan` mode (next 8 languages, e.g. Assamese)

```python
builder.add_column(
    DocumentTranslationConfig(
        name="translation",
        text_column="text",
        source_language="en-Latn-IN",
        target_language="as-Beng-IN",
        translation_mode="bodhan",
        model_alias="bodhan",
        evaluator_model_alias="gemma",
    )
)
```

### Santali -- `bodhan` mode + windowed translation

```python
builder.add_column(
    DocumentTranslationConfig(
        name="translation",
        text_column="text",
        source_language="en-Latn-IN",
        target_language="sat-Olck-IN",
        translation_mode="bodhan",
        model_alias="bodhan",
        evaluator_model_alias="gemma",
        enable_windowing=True,
        window_tokens=2048,
        window_merge_model_alias="gemma",
    )
)
```

### Manipuri -- `bodhan` mode + deterministic script remap

```python
builder.add_column(
    DocumentTranslationConfig(
        name="translation",
        text_column="text",
        source_language="en-Latn-IN",
        target_language="mni-Beng-IN",
        translation_mode="bodhan",
        model_alias="bodhan",
        evaluator_model_alias="gemma",
        target_language_script="Mtei",
    )
)
```

See [Supported languages and model selection](#supported-languages-and-model-selection)
above for `translation_mode`/`enable_windowing`/`target_language_script`
choices per language, and `docs/index.md` in this package for the full
field reference.

For the full plugin authoring guide, see the
[main repository docs](https://nvidia-nemo.github.io/DataDesignerPlugins/authoring/).

Plugin documentation for the repository site lives in this package's `docs/`
directory.
