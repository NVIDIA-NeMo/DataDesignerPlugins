# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer_nemotron_bharat_translation.language_codes import (
    BODHAN_LANGUAGE_DESCRIPTIONS,
    GENERIC_LANGUAGE_DESCRIPTIONS,
    TARGET_LANGUAGE_CODES,
    describe_script,
    describe_target_language,
)


def test_generic_language_descriptions_cover_all_target_codes() -> None:
    assert set(GENERIC_LANGUAGE_DESCRIPTIONS) == set(TARGET_LANGUAGE_CODES)


def test_low_resource_language_codes_all_describe_cleanly() -> None:
    assert GENERIC_LANGUAGE_DESCRIPTIONS["ks-Arab-IN"] == "Kashmiri in Arabic script"
    assert GENERIC_LANGUAGE_DESCRIPTIONS["sat-Olck-IN"] == "Santali in Ol Chiki script"


def test_target_language_codes_has_no_duplicates() -> None:
    assert len(TARGET_LANGUAGE_CODES) == len(set(TARGET_LANGUAGE_CODES))


def test_manipuri_has_both_intermediate_and_final_script_descriptions() -> None:
    assert GENERIC_LANGUAGE_DESCRIPTIONS["mni-Beng-IN"] == "Manipuri in Bengali script"
    assert GENERIC_LANGUAGE_DESCRIPTIONS["mni-Mtei-IN"] == "Manipuri in Meetei Mayek script"


def test_bodhan_language_descriptions_use_plain_names() -> None:
    assert BODHAN_LANGUAGE_DESCRIPTIONS["en-Latn-IN"] == "English"
    assert BODHAN_LANGUAGE_DESCRIPTIONS["sat-Olck-IN"] == "Santali"
    assert BODHAN_LANGUAGE_DESCRIPTIONS["brx-Deva-IN"] == "Bodo"


def test_bodhan_language_descriptions_qualify_multi_script_languages() -> None:
    assert BODHAN_LANGUAGE_DESCRIPTIONS["ks-Arab-IN"] == "Kashmiri (Perso-Arabic script)"
    assert BODHAN_LANGUAGE_DESCRIPTIONS["sd-Deva-IN"] == "Sindhi (Devanagari script)"
    assert BODHAN_LANGUAGE_DESCRIPTIONS["mni-Beng-IN"] == "Manipuri (Bengali script)"


def test_bodhan_language_descriptions_includes_meetei_mayek_for_remap_evaluation() -> None:
    # bodhan-ai/indic-translate is never asked to translate into Meetei Mayek directly -- the
    # Manipuri route always targets it at the Bengali-script intermediate
    # stage and deterministically remaps from there -- but the evaluator
    # model still needs a description to score the remapped result.
    assert BODHAN_LANGUAGE_DESCRIPTIONS["mni-Mtei-IN"] == "Manipuri (Meetei Mayek script)"


def test_bodhan_language_descriptions_cover_all_22_languages() -> None:
    assert BODHAN_LANGUAGE_DESCRIPTIONS["hi-Deva-IN"] == "Hindi"
    assert set(TARGET_LANGUAGE_CODES) <= set(BODHAN_LANGUAGE_DESCRIPTIONS)


def test_describe_target_language_omits_adjective_for_native_region() -> None:
    assert describe_target_language("hi-Deva-IN") == "Hindi in Devanagari script"


def test_describe_target_language_keeps_adjective_for_non_native_region() -> None:
    assert describe_target_language("en-Latn-IN") == "Indian English in Latin script"


def test_describe_target_language_falls_back_to_raw_code_for_unknown_language() -> None:
    assert describe_target_language("zz-Latn-IN") == "Indian zz in Latin script"


def test_describe_script_returns_script_only() -> None:
    assert describe_script("hi-Deva-IN") == "Devanagari script"


@pytest.mark.parametrize("code", ["", "not-a-code", "hi-Deva", "hindi-Deva-IN", None])
def test_describe_target_language_rejects_malformed_codes(code: str | None) -> None:
    with pytest.raises(ValueError, match="language-Script-Region"):
        describe_target_language(code)


@pytest.mark.parametrize("code", ["", "not-a-code", None])
def test_describe_script_rejects_malformed_codes(code: str | None) -> None:
    with pytest.raises(ValueError, match="language-Script-Region"):
        describe_script(code)
