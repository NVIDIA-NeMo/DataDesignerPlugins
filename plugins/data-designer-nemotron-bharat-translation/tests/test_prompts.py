# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from jinja2 import Template

from data_designer_nemotron_bharat_translation.prompts import (
    GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS,
    NATIVE_NUMBERS_DOC,
    native_numbers_doc,
)


class TestNativeNumbersDoc:
    def test_returns_the_legend_for_a_covered_language(self) -> None:
        assert native_numbers_doc("hi-Deva-IN") == NATIVE_NUMBERS_DOC["hi-Deva-IN"]
        assert "0->०" in native_numbers_doc("hi-Deva-IN")
        assert "9->९" in native_numbers_doc("hi-Deva-IN")

    def test_raises_for_an_uncovered_language(self) -> None:
        with pytest.raises(ValueError, match="not found in NATIVE_NUMBERS_DOC"):
            native_numbers_doc("en-Latn-IN")

    def test_covers_all_12_documented_languages(self) -> None:
        assert set(NATIVE_NUMBERS_DOC) == {
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
        }


class TestNativeNumeralPrompts:
    def test_translation_prompt_carries_the_native_numbers_doc_placeholder(self) -> None:
        assert "{native_numbers_doc}" in GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS

    def test_resolving_native_numbers_doc_leaves_jinja_placeholders_intact(self) -> None:
        resolved = GENERIC_TRANSLATION_PROMPT_NATIVE_NUMERALS.replace(
            "{native_numbers_doc}", native_numbers_doc("hi-Deva-IN")
        )
        assert "{native_numbers_doc}" not in resolved
        assert native_numbers_doc("hi-Deva-IN") in resolved
        rendered = Template(resolved).render(
            text="hello", source_language_description="English", target_language_description="Hindi"
        )
        assert "hello" in rendered
        assert native_numbers_doc("hi-Deva-IN") in rendered
