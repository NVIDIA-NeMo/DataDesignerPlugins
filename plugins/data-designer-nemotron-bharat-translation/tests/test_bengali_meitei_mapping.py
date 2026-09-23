# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer_nemotron_bharat_translation.bengali_meitei_mapping import (
    BENGALI_DROP,
    BENGALI_MULTI,
    BENGALI_NUKTA,
    transliterate,
)


class TestTransliterate:
    def test_empty_string_passes_through(self) -> None:
        assert transliterate("") == ""

    def test_consonant_and_matra(self) -> None:
        # ক (KA) + া (AA matra) -> ꯀ + ꯥ
        assert transliterate("কা") == "ꯀꯥ"

    def test_many_to_one_sibilants(self) -> None:
        # শ/ষ/স all collapse to the same Meetei sibilant.
        assert transliterate("শ") == transliterate("ষ") == transliterate("স")

    def test_independent_vowel_expands_to_two_code_points(self) -> None:
        for bengali, meitei in BENGALI_MULTI.items():
            assert transliterate(bengali) == meitei

    def test_digits(self) -> None:
        assert transliterate("০১২৩৪৫৬৭৮৯") == "꯰꯱꯲꯳꯴꯵꯶꯷꯸꯹"

    def test_danda(self) -> None:
        assert transliterate("।") == "꯫"

    def test_dropped_characters_are_removed(self) -> None:
        for char in BENGALI_DROP:
            assert transliterate(char) == ""
        assert transliterate("কঃ") == "ꯀ"

    def test_nukta_variants_map_to_the_same_target(self) -> None:
        targets = {transliterate(sequence) for sequence in BENGALI_NUKTA}
        assert targets == {"ꯔ", "ꯌ"}

    def test_non_bengali_text_passes_through_unchanged(self) -> None:
        assert transliterate("abc123!?") == "abc123!?"

    def test_mixed_bengali_and_non_bengali(self) -> None:
        assert transliterate("hello কা world") == "hello ꯀꯥ world"
