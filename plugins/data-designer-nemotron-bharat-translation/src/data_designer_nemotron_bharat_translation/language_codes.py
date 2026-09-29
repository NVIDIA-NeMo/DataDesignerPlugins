# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse ``language-Script-Region`` BCP-47-style codes into natural phrasing.

The target/source/intermediate language columns used by the translation
plugins are expected to hold codes of the form ``language-Script-Region``,
e.g. ``en-Latn-IN`` (Indian English in Latin script) or ``hi-Deva-IN``
(Hindi in Devanagari script -- no region adjective, since Hindi is native to
India; see :data:`NATIVE_REGION`). These helpers render natural-sounding
descriptions from those codes for use in translation/evaluation prompts,
instead of showing the raw code.
"""

from __future__ import annotations

import re

BCP47_PATTERN = re.compile(r"^([a-zA-Z]{2,3})-([A-Za-z]{4})-([A-Za-z]{2})$")

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "mr": "Marathi",
    "te": "Telugu",
    "ta": "Tamil",
    "gu": "Gujarati",
    "ur": "Urdu",
    "kn": "Kannada",
    "or": "Odia",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ne": "Nepali",
    "as": "Assamese",
    "ks": "Kashmiri",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    # No ISO 639-1 two-letter code exists for these -- three-letter only.
    "brx": "Bodo",
    "doi": "Dogri",
    "gom": "Konkani",
    "mai": "Maithili",
    "mni": "Manipuri",
    "sat": "Santali",
}

SCRIPT_NAMES: dict[str, str] = {
    "Latn": "Latin script",
    "Deva": "Devanagari script",
    "Beng": "Bengali script",
    "Taml": "Tamil script",
    "Telu": "Telugu script",
    "Gujr": "Gujarati script",
    "Arab": "Arabic script",
    "Knda": "Kannada script",
    "Orya": "Odia script",
    "Mlym": "Malayalam script",
    "Guru": "Gurmukhi script",
    "Mtei": "Meetei Mayek script",
    "Olck": "Ol Chiki script",
}

REGION_ADJECTIVES: dict[str, str] = {
    "IN": "Indian",
    "US": "American",
}

# Languages that are native to a given region get no region adjective in
# describe_target_language() -- e.g. "hi-Deva-IN" is just "Hindi", not
# "Indian Hindi", since Hindi is indigenous to India. Non-native languages
# still get the adjective (e.g. "en-Latn-IN" -> "Indian English", to
# distinguish it from British/American English).
NATIVE_REGION: dict[str, str] = {
    "hi": "IN",
    "bn": "IN",
    "gu": "IN",
    "kn": "IN",
    "ml": "IN",
    "mr": "IN",
    "ne": "IN",
    "or": "IN",
    "pa": "IN",
    "ta": "IN",
    "te": "IN",
    "ur": "IN",
    "as": "IN",
    "ks": "IN",
    "sa": "IN",
    "sd": "IN",
    "brx": "IN",
    "doi": "IN",
    "gom": "IN",
    "mai": "IN",
    "mni": "IN",
    "sat": "IN",
}

# All 22 constitutionally scheduled Indic languages this pipeline targets,
# plus English, each in its default native script, region India, using the
# ISO 639-1 two-letter code where one exists and the ISO 639-2/3
# three-letter code otherwise (Bodo, Dogri, Konkani, Maithili, Manipuri,
# Santali). Manipuri carries two codes: the Bengali-script intermediate
# stage and the Meetei Mayek final stage (see
# DocumentTranslationConfig.target_language_script) -- both need a description.
TARGET_LANGUAGE_CODES: list[str] = [
    "en-Latn-IN",
    "bn-Beng-IN",
    "gu-Gujr-IN",
    "hi-Deva-IN",
    "kn-Knda-IN",
    "ml-Mlym-IN",
    "mr-Deva-IN",
    "ne-Deva-IN",
    "or-Orya-IN",
    "pa-Guru-IN",
    "ta-Taml-IN",
    "te-Telu-IN",
    "ur-Arab-IN",
    "as-Beng-IN",
    "ks-Arab-IN",
    "sa-Deva-IN",
    "sd-Deva-IN",
    "brx-Deva-IN",
    "doi-Deva-IN",
    "gom-Deva-IN",
    "mai-Deva-IN",
    "mni-Beng-IN",
    "mni-Mtei-IN",
    "sat-Olck-IN",
]


def parse_bcp47(code: str) -> tuple[str, str, str]:
    """Splits a BCP-47 ``language-Script-Region`` code into its parts, normalizing case.

    Args:
        code: A BCP-47-style code in ``language-Script-Region`` form, e.g.
            ``"en-Latn-IN"``.

    Returns:
        A ``(language, script, region)`` tuple, e.g. ``("en", "Latn", "IN")``.

    Raises:
        ValueError: If ``code`` isn't in the expected
            ``language-Script-Region`` form.
    """
    match = BCP47_PATTERN.match(code or "")
    if not match:
        raise ValueError(f"{code!r} is not in the expected 'language-Script-Region' form (e.g. 'en-Latn-IN').")
    lang, script, region = match.groups()
    return lang.lower(), script[0].upper() + script[1:].lower(), region.upper()


def describe_target_language(code: str) -> str:
    """Describes a BCP-47 ``language-Script-Region`` code in natural language.

    The region adjective is omitted when the language is native to that
    region -- e.g. ``"hi-Deva-IN"`` becomes ``"Hindi in Devanagari script"``,
    not ``"Indian Hindi in Devanagari script"``, since Hindi is indigenous to
    India (see :data:`NATIVE_REGION`). Non-native languages still get the
    adjective, e.g. ``"en-Latn-IN"`` -> ``"Indian English in Latin script"``.

    Args:
        code: A BCP-47-style code in ``language-Script-Region`` form, e.g.
            ``"en-Latn-IN"``.

    Returns:
        A natural-language description of the language and script.

    Raises:
        ValueError: If ``code`` isn't in the expected
            ``language-Script-Region`` form.
    """
    lang, _script, region = parse_bcp47(code)
    language_name = LANGUAGE_NAMES.get(lang, lang)
    script_name = describe_script(code)
    region_adjective = None if NATIVE_REGION.get(lang) == region else REGION_ADJECTIVES.get(region, region)
    if region_adjective is None:
        return f"{language_name} in {script_name}"
    return f"{region_adjective} {language_name} in {script_name}"


def describe_script(code: str) -> str:
    """Describes only the script portion of a BCP-47 code.

    Args:
        code: A BCP-47-style code in ``language-Script-Region`` form, e.g.
            ``"hi-Deva-IN"``.

    Returns:
        A natural-language description of the script, e.g.
        ``"Devanagari script"``.

    Raises:
        ValueError: If ``code`` isn't in the expected
            ``language-Script-Region`` form.
    """
    _lang, script, _region = parse_bcp47(code)
    return SCRIPT_NAMES.get(script, f"{script} script")


# Precomputed descriptions for TARGET_LANGUAGE_CODES, via
# describe_target_language(). This is the default value for
# DocumentTranslationConfig.language_descriptions, so configs that only
# ever use these codes are unaffected by that field's existence. Pass your
# own map (e.g. extending a copy of this one) to describe languages
# LANGUAGE_NAMES/NATIVE_REGION don't cover, without having to add entries
# to those tables.
GENERIC_LANGUAGE_DESCRIPTIONS: dict[str, str] = {code: describe_target_language(code) for code in TARGET_LANGUAGE_CODES}

# bodhan-ai/indic-translate's own naming contract for source/target
# languages, for use as DocumentTranslationConfig.language_descriptions when
# BODHAN_TRANSLATION_PROMPT is the translation_prompt. bodhan-ai/indic-translate
# expects plain language names, not GENERIC_LANGUAGE_DESCRIPTIONS' "<Language>
# in <Script> script" phrasing -- qualified with "(<Script> script)" only for
# the languages that have more than one script in its own table (Kashmiri,
# Sindhi, Manipuri). Covers all 22 constitutionally scheduled languages plus
# English. Includes "mni-Mtei-IN" even though bodhan-ai/indic-translate is
# never asked to translate into Meetei Mayek directly -- the Manipuri route
# always targets bodhan-ai/indic-translate at "mni-Beng-IN" (Bengali script)
# and deterministically remaps from there (see
# DocumentTranslationConfig.target_language_script) -- but the evaluator
# model still needs a description for "mni-Mtei-IN" to score the remapped
# result.
BODHAN_LANGUAGE_DESCRIPTIONS: dict[str, str] = {
    "en-Latn-IN": "English",
    "hi-Deva-IN": "Hindi",
    "bn-Beng-IN": "Bengali",
    "mr-Deva-IN": "Marathi",
    "te-Telu-IN": "Telugu",
    "ta-Taml-IN": "Tamil",
    "gu-Gujr-IN": "Gujarati",
    "ur-Arab-IN": "Urdu",
    "kn-Knda-IN": "Kannada",
    "or-Orya-IN": "Odia",
    "ml-Mlym-IN": "Malayalam",
    "pa-Guru-IN": "Punjabi",
    "ne-Deva-IN": "Nepali",
    "as-Beng-IN": "Assamese",
    "ks-Arab-IN": "Kashmiri (Perso-Arabic script)",
    "sa-Deva-IN": "Sanskrit",
    "sd-Deva-IN": "Sindhi (Devanagari script)",
    "brx-Deva-IN": "Bodo",
    "doi-Deva-IN": "Dogri",
    "gom-Deva-IN": "Konkani",
    "mai-Deva-IN": "Maithili",
    "mni-Beng-IN": "Manipuri (Bengali script)",
    "mni-Mtei-IN": "Manipuri (Meetei Mayek script)",
    "sat-Olck-IN": "Santali",
}
