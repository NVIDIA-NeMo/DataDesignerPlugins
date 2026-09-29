# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default translation, evaluation, rewrite, and merge prompts for the translation plugins.

``GENERIC_TRANSLATION_PROMPT`` is a plain-text translation prompt (no JSON
wrapper) that directly translates a document into a target language, with no
user-turn/persona simulation, meant for a general-purpose instruction-tuned
model (e.g. Gemma). Like ``GENERIC_EVALUATION_PROMPT``/``GENERIC_REWRITE_PROMPT``,
it's a Jinja2 template the generator renders itself (see
``translation_prompt`` on ``DocumentTranslationConfig``), with ``text``,
``source_language_description``, and ``target_language_description`` in
context -- the latter two hold the natural-language *description* of the
fixed ``source_language``/``target_language`` config values (e.g. "Indian
English in Latin script", not "en-Latn-IN"), and (for the intermediate
stage) the generator renders with the stage-specific description, so a
single template, reused for both the intermediate and final stage, still
gets the right wording for each. The model's response is used as plain
translated text via ``TextResponseRecipe``.

``BODHAN_TRANSLATION_PROMPT`` is a minimal alternative for
``bodhan-ai/indic-translate``, a narrow translation specialist rather than a
general-purpose instruction-tuned model: its model card calls for a single
terse user turn, no system message, and no source-language mention, since
anything more (including ``GENERIC_TRANSLATION_PROMPT``'s longer rule list)
measurably hurts its translation quality. It carries the same
``{{ text }}``/``{{ target_language_description }}`` template contract as
``GENERIC_TRANSLATION_PROMPT`` and is a drop-in substitute via
``translation_prompt``.

``GENERIC_EVALUATION_PROMPT`` is a standard MQM-based evaluation prompt.
Rendered with ``source_language_description``, ``target_language_description``,
``text``, and ``translation_result.translated_text``; the model's JSON
response is parsed into ``TranslationEvaluation`` via
``PydanticResponseRecipe`` and written out as a single dict-valued
``<name>_evaluation`` column.

``GENERIC_REWRITE_PROMPT`` is a plain-text rewrite prompt (no JSON wrapper)
used for the self-correction pass. Rendered with
``source_language_description``, ``target_language_description``,
``text_source`` (the original document), ``text`` (the
current translation being corrected), and the evaluation fields
``overall_score``/``analysis``/``error_analysis``/``justification`` (from
the current translation's own evaluation). See ``generator.py`` for the
exact rewrite loop and its history-column bookkeeping.

``MERGE_PROMPT`` is windowed translation's merge-phase prompt (see
``Translator`` in ``translator.py``): it extracts one window's own
non-overlapping segment, shown alongside its neighbours' translations for
context, so segment extraction calls are independent and can run
concurrently. Rendered with ``source_text`` (the window plus its
neighbours' span), ``windows_block``, ``segment_source`` (the exact portion
to return), ``source_language``, ``target_language``. Per-window
translation itself reuses this module's
``GENERIC_TRANSLATION_PROMPT``/``BODHAN_TRANSLATION_PROMPT`` via
``translator.py`` -- the same ``translation_mode``/``translation_prompt``
fields on ``DocumentTranslationConfig`` select between the two, windowed or
not.

``GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS`` is a native-numeral variant
of ``GENERIC_TRANSLATION_PROMPT`` (numbers in natural-language prose
written in the target language's native numeral script instead of ASCII
digits -- digits inside protected spans like code, formulas, URLs, and
LaTeX stay ASCII either way). Like the plain prompt, it's a static
template -- but since the numeral legend depends on ``target_language`` (a
full BCP-47 ``language-Script-Region`` code), it carries a literal
``{native_numbers_doc}`` placeholder (single braces, distinct from the
``{{ }}`` Jinja placeholders) instead of baking one in. Callers resolve it
with a plain ``str.replace()``, e.g.
``GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS.replace("{native_numbers_doc}",
native_numbers_doc(target_language))``, then pass the result via
``translation_prompt`` -- see ``NATIVE_NUMBERS_DOC``/``native_numbers_doc()``
for the per-language legend lookup below. This is the default for
``DocumentTranslationConfig.translation_prompt_native_numerals``, used for
rows drawn into the native-numeral style when ``native_numeral_probability``
is set (see ``config.py``); a caller can override it to keep a customized
``translation_prompt`` consistent with its native-numeral counterpart.
Evaluation always uses the plain ``evaluation_prompt`` regardless of which
style was drawn for translation -- there is no native-numeral evaluation
prompt variant. Only the languages covered by ``NATIVE_NUMBERS_DOC`` below
have a legend; an uncovered code raises ``ValueError``.
"""

GENERIC_TRANSLATION_PROMPT = """Translate the following document from {{ source_language_description }} into {{ target_language_description }}.

Rules:
- Prioritize natural sentence flow and readability. Translate prose into native-sounding {{ target_language_description }}; avoid literal, word-for-word translations that feel forced.
- Do not write the English meaning of words in brackets.
- You can use common technical loanwords if they are standard in {{ target_language_description }} to ensure the text sounds professional and modern.
- Translate every span of prose regardless of the source script. Translate the entire document, including every part of any multi-part questions, lists, or sub-tasks, without truncation.
- Preserve formatting, layout, headings, bullets, and numbering exactly as in the source.
- Preserve all protected spans, technical syntax, code, formulas, URLs, file paths, and LaTeX commands byte-for-byte. Translate only the internal natural-language labels.
- Preserve the identity of named entities and relationships between them. Do not substitute them with similar-sounding words or translate them morpheme-by-morpheme.
- Maintain all numbers, units, directions, and quantitative relationships exactly. Use ASCII digits (0-9) with grouping conventions appropriate for the target locale.
- Translate idioms into natural {{ target_language_description }} equivalents. Ensure grammatical consistency (e.g., gender and number agreement) throughout the document.
- Return ONLY the translated document text. Do not provide a confidence score, JSON markers, code fences, or any conversational preamble.

Source document:
{{ text }}"""

BODHAN_TRANSLATION_PROMPT = """Translate the following text into {{ target_language_description }}:

{{ text }}"""

GENERIC_EVALUATION_PROMPT = """You are a rigorous professional translation auditor. Your task is to evaluate the quality of a translation based on the provided source text from {{ source_language_description }} into {{ target_language_description }}.

### 1. Evaluation Framework: MQM (Multidimensional Quality Metrics)
- **Faithfulness**: Meaning preservation, lack of hallucinations, and accuracy of entities/numbers.
- **Fluency**: Grammar, spelling, punctuation, and naturalness in the target language.
- **Terminology**: Correct technical, academic, or domain-specific term choice.
- **Style**: Adherence to tone and consistency (e.g., avoiding unrequested code-switching or transliteration into English).
- **Format Preservation**: Structure, markup, tables, code fences, LaTeX, and layout preservation.

### 2. Constraints (Strictly Enforced)
- **Analysis Length**: The 'analysis' section must be concise and strictly under 100 words.
- **Justification Length**: The 'justification' field in the JSON must be exactly 1 or 2 sentences focusing on the specific justification for the score band.
- **Output Format**: Return ONLY a valid JSON object. No conversational filler.

### 3. Scoring Rubric (Strict Category Enforcement)
- **90-100**: Professional quality; NO major or critical errors; perfect format preservation.
- **70-89**: Good; contains ONLY minor errors (e.g., typos, slight awkwardness).
- **40-69**: Fair; contains AT LEAST ONE major error (e.g., entity swap, number drift, polarity flip, or unrequested language mixing).
- **0-39**: Poor; critical errors, gross inaccuracies, or destroyed technical formatting/LaTeX.

IMPORTANT:
- Translating any span that is not actually written in {{ source_language_description }} (e.g., text already in {{ target_language_description }}, a third language, or a protected span like code/URLs/LaTeX) into {{ target_language_description }} is a Faithfulness error -- only genuine {{ source_language_description }} content should be translated.
- Omitting, summarizing, condensing, or truncating any part of the source text -- even if the resulting text is itself fluent and accurate -- is a Major or Critical error depending on how much content is missing. The candidate translation must cover the ENTIRE source text; a shorter "summary" or "the gist of it" is not an acceptable substitute for a complete translation, regardless of how well-written it is.

### 4. Input Data
**Source Text**:
{{ text }}

**Candidate Translation**:
{{ translation_result.translated_text }}

### 5. Expected Output Format
```json
{
  "analysis": "Brief internal analysis of error spans. Explicitly label errors as Minor, Major, or Critical.",
  "error_analysis": [
    {"text_span": "The text span that is problematic", "error_category": "Faithfulness/Fluency/Terminology/Format Preservation/Script Compliance", "severity": "Minor/Major/Critical", "explanation": "Why the error was flagged"}
  ],
  "justification": "1-2 sentence justification of why the score was assigned to the chosen band.",
  "overall_score": 0-100
}
```"""

GENERIC_REWRITE_PROMPT = """You are an expert {{ target_language_description }} translator and editor. You are given a source document in {{ source_language_description }}, an existing {{ target_language_description }} translation of it, and a quality audit that identified specific errors in that translation. Produce an improved {{ target_language_description }} translation that fixes every identified error while preserving everything that was already correct.

Rules:
- Fix every error listed in the audit below, addressing the flagged Faithfulness, Fluency, Terminology, Style, Format Preservation, and Script Compliance issues.
- Do not introduce new errors and do not alter spans of the existing translation that were already correct.
- Prioritize natural sentence flow and readability. Produce native-sounding {{ target_language_description }}; avoid literal, word-for-word translations that feel forced.
- Do not write the English meaning of words in brackets.
- You can use common technical loanwords if they are standard in {{ target_language_description }} to ensure the text sounds professional and modern.
- Re-translate the entire document, including every part of any multi-part questions, lists, or sub-tasks, without truncation.
- Preserve formatting, layout, headings, bullets, and numbering exactly as in the source.
- Preserve all protected spans, technical syntax, code, formulas, URLs, file paths, and LaTeX commands byte-for-byte. Translate only the internal natural-language labels.
- Preserve the identity of named entities and relationships between them. Maintain all numbers, units, directions, and quantitative relationships exactly.
- Output ONLY the final corrected translation itself, as a clean standalone document. Do NOT explain, justify, annotate, or narrate any corrections; do NOT reference the audit or the flagged errors; do NOT include commentary such as "the correct translation is" or "X should be Y", and do NOT compare the old and new versions. Never enter a loop of repeated or self-correcting text.
- Return ONLY the corrected translated document text. Do not provide a confidence score, JSON markers, code fences, or any conversational preamble.

Source document ({{ source_language_description }}):
{{ text_source }}

Existing {{ target_language_description }} translation:
{{ text }}

Quality audit of the existing translation (errors to fix):
Overall score: {{ overall_score }}
Analysis: {{ analysis }}
Error spans: {{ error_analysis }}
Justification: {{ justification }}"""

MERGE_PROMPT = """You are extracting ONE clean segment of a {{ target_language }} translation and making necessary corrections to it.

A larger {{ source_language }} document was split into OVERLAPPING windows and each window was
translated on its own. Below are a few adjacent windows' shared source text, their translations,
and the exact portion of the source. That portion lies within ONE of the windows; the neighbouring
windows are shown only for context (the preceding one for how the run-up was translated, the
following one for the tail they share).
You must return the translation of this portion and make necessary corrections to it.

Rules:
- Return the {{ target_language }} translation of ONLY the portion under "# The portion to return".
- Make necessary corrections to the translation to make it correct and consistent with the surrounding context of the source text.
- Do NOT include the translation of any other part, and do NOT drop, summarise or invent content.
- Keep the line structure of the portion: one translated line per source line, same order.
- Preserve formatting, markup, LaTeX, numbers and named entities exactly as in the source.
- Return ONLY the translated portion. No preamble, no explanation, no code fences.

# {{ source_language }} source (the text the windows cover)
{{ source_text }}

# Window translations (in document order)
{{ windows_block }}

# The portion to return, in {{ source_language }}
{{ segment_source }}
"""

GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS = """Translate the following document from {{ source_language_description }} into {{ target_language_description }}.

Rules:
- Prioritize natural sentence flow and readability. Translate prose into native-sounding {{ target_language_description }}; avoid literal, word-for-word translations that feel forced.
- Do not write the English meaning of words in brackets.
- You can use common technical loanwords if they are standard in {{ target_language_description }} to ensure the text sounds professional and modern.
- Translate every span of prose regardless of the source script. Translate the entire document, including every part of any multi-part questions, lists, or sub-tasks, without truncation.
- Preserve formatting, layout, headings, bullets, and numbering exactly as in the source.
- Preserve all protected spans, technical syntax, code, formulas, URLs, file paths, and LaTeX commands byte-for-byte. Translate only the internal natural-language labels.
- Preserve the identity of named entities and relationships between them. Do not substitute them with similar-sounding words or translate them morpheme-by-morpheme.
- Maintain all numbers, units, directions, and quantitative relationships exactly. Write all numbers appearing in natural-language prose using the native {{ target_language_description }} numeral script (NOT ASCII/Western 0-9 digits), with grouping conventions appropriate for the target locale.
- It is allowed (and expected) to use ASCII digits (0-9) whenever the document contains equations, LaTeX, or code blocks — such digits must stay ASCII.
- Keep digits inside protected spans (code, formulas, URLs, file paths, LaTeX, identifiers, version numbers) as-is in ASCII. NEVER use native numerals inside computer code, code snippets, or commands — code must always use ASCII digits.
- Keep ALL mathematical and arithmetic expressions in ASCII digits, whether or not they are marked up (LaTeX, code fences, or bare inline text). Any expression combining numbers with operators, relations, or variables (e.g. "2 + 2 = 4", "x > 5", "3/4", "10%", "5 × 10^3") is a formula: keep its digits and symbols ASCII. Use native numerals only for standalone quantities in natural-language prose (e.g. counts, dates, measurements written as words-plus-number).
- Translate idioms into natural {{ target_language_description }} equivalents. Ensure grammatical consistency (e.g., gender and number agreement) throughout the document.
- Return ONLY the translated document text. Do not provide a confidence score, JSON markers, code fences, or any conversational preamble.

{native_numbers_doc}

Source document:
{{ text }}"""

# Keyed by full BCP-47 language-Script-Region codes -- the same form as
# TARGET_LANGUAGE_CODES / config.target_language (see language_codes.py) --
# rather than bare 2-letter language codes, so callers can look up a legend
# directly from config.target_language with no separate parsing step.
NATIVE_NUMBERS_DOC: dict[str, str] = {
    "bn-Beng-IN": "Native numeral legend for Bengali (Western digit -> native digit): 0->০, 1->১, 2->২, 3->৩, 4->৪, 5->৫, 6->৬, 7->৭, 8->৮, 9->৯.",
    "gu-Gujr-IN": "Native numeral legend for Gujarati (Western digit -> native digit): 0->૦, 1->૧, 2->૨, 3->૩, 4->૪, 5->૫, 6->૬, 7->૭, 8->૮, 9->૯.",
    "hi-Deva-IN": "Native numeral legend for Hindi (Western digit -> native digit): 0->०, 1->१, 2->२, 3->३, 4->४, 5->५, 6->६, 7->७, 8->८, 9->९.",
    "kn-Knda-IN": "Native numeral legend for Kannada (Western digit -> native digit): 0->೦, 1->೧, 2->೨, 3->೩, 4->೪, 5->೫, 6->೬, 7->೭, 8->೮, 9->೯.",
    "ml-Mlym-IN": "Native numeral legend for Malayalam (Western digit -> native digit): 0->൦, 1->൧, 2->൨, 3->൩, 4->൪, 5->൫, 6->൬, 7->൭, 8->൮, 9->൯. Malayalam also has historic (additive) symbols for 10->൰, 100->൱, 1000->൲ and fractions ¼->൳, ½->൴, ¾->൵.",
    "mr-Deva-IN": "Native numeral legend for Marathi (Western digit -> native digit): 0->०, 1->१, 2->२, 3->३, 4->४, 5->५, 6->६, 7->७, 8->८, 9->९.",
    "ne-Deva-IN": "Native numeral legend for Nepali (Western digit -> native digit): 0->०, 1->१, 2->२, 3->३, 4->४, 5->५, 6->६, 7->७, 8->८, 9->९.",
    "or-Orya-IN": "Native numeral legend for Odia (Western digit -> native digit): 0->୦, 1->୧, 2->୨, 3->୩, 4->୪, 5->୫, 6->୬, 7->୭, 8->୮, 9->୯.",
    "pa-Guru-IN": "Native numeral legend for Punjabi (Western digit -> native digit): 0->੦, 1->੧, 2->੨, 3->੩, 4->੪, 5->੫, 6->੬, 7->੭, 8->੮, 9->੯.",
    "ta-Taml-IN": "Native numeral legend for Tamil (Western digit -> native digit): 0->௦, 1->௧, 2->௨, 3->௩, 4->௪, 5->௫, 6->௬, 7->௭, 8->௮, 9->௯. Tamil also has historic (additive) symbols for 10->௰, 100->௱, 1000->௲.",
    "te-Telu-IN": "Native numeral legend for Telugu (Western digit -> native digit): 0->౦, 1->౧, 2->౨, 3->౩, 4->౪, 5->౫, 6->౬, 7->౭, 8->౮, 9->౯.",
    "ur-Arab-IN": "Native numeral legend for Urdu (Western digit -> native digit): 0->۰, 1->۱, 2->۲, 3->۳, 4->۴, 5->۵, 6->۶, 7->۷, 8->۸, 9->۹.",
}


def native_numbers_doc(target_language: str) -> str:
    """Legend mapping Western digits 0-9 to ``target_language``'s native numerals.

    Args:
        target_language: A full BCP-47 ``language-Script-Region`` code
            (e.g. ``"hi-Deva-IN"`` -- the same form as
            ``DocumentTranslationConfig.target_language``) with an entry in
            :data:`NATIVE_NUMBERS_DOC`.

    Returns:
        The native-numeral legend text for ``target_language``.

    Raises:
        ValueError: If ``target_language`` has no entry in
            :data:`NATIVE_NUMBERS_DOC`. There is no all-languages
            fallback -- ``target_language`` is a single fixed value for the
            whole column (see ``config.py``), so the legend for that one
            target language can always be resolved up front.
    """
    if target_language not in NATIVE_NUMBERS_DOC:
        raise ValueError(
            f"Language code {target_language!r} not found in NATIVE_NUMBERS_DOC. "
            f"Available languages: {sorted(NATIVE_NUMBERS_DOC)}"
        )
    return NATIVE_NUMBERS_DOC[target_language]
