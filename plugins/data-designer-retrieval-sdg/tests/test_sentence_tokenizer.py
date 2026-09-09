# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dependency-free sentence tokenizer adapter."""

import pytest

from data_designer_retrieval_sdg.sentence_tokenizer import split_sentences


def test_split_sentences_preserves_abbreviations_decimals_and_closing_quotes() -> None:
    text = 'Dr. Ada paid $3.50. She said "Done." Then she left.'
    assert split_sentences(text) == ["Dr. Ada paid $3.50.", 'She said "Done."', "Then she left."]


def test_split_sentences_supports_unicode_terminators_without_spaces() -> None:
    assert split_sentences("最初の文です。次の文です！最後です？") == ["最初の文です。", "次の文です！", "最後です？"]


def test_split_sentences_supports_common_unicode_terminators_with_spaces() -> None:
    text = "هل هذا صحيح؟ نعم۔ फिर मिलेंगे। Այո։ እሺ።"
    assert split_sentences(text) == ["هل هذا صحيح؟", "نعم۔", "फिर मिलेंगे।", "Այո։", "እሺ።"]


def test_split_sentences_preserves_inline_ellipsis_without_spacing() -> None:
    assert split_sentences("Wait…what happened? Next.") == ["Wait…what happened?", "Next."]


def test_split_sentences_strong_terminator_overrides_abbreviation() -> None:
    assert split_sentences("Is she a Dr.? Yes.") == ["Is she a Dr.?", "Yes."]


def test_split_sentences_checks_complete_mixed_terminator_run() -> None:
    assert split_sentences("終わりです.。次です。") == ["終わりです.。", "次です。"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("「最初の文です。」次の文です。", ["「最初の文です。」", "次の文です。"]),
        ("『最初の文です。』次の文です。", ["『最初の文です。』", "次の文です。"]),
        ("（最初の文です。）次の文です。", ["（最初の文です。）", "次の文です。"]),
        ("〝最初の文です。〞次の文です。", ["〝最初の文です。〞", "次の文です。"]),
    ],
)
def test_split_sentences_keeps_unicode_closers_with_sentence(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The U.S. Army arrived.", ["The U.S. Army arrived."]),
        ("The U.S.S.R. Delegation arrived. It left.", ["The U.S.S.R. Delegation arrived.", "It left."]),
        ("Sentence A. Bob replied.", ["Sentence A.", "Bob replied."]),
    ],
)
def test_split_sentences_distinguishes_multi_initial_abbreviations(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What?! Really... Yes.", ["What?!", "Really...", "Yes."]),
        ("Visit example.com. Then email a.b@example.com.", ["Visit example.com.", "Then email a.b@example.com."]),
        ("The value is 3.14. Next value.", ["The value is 3.14.", "Next value."]),
        ('He shouted "Go!", then left.', ['He shouted "Go!", then left.']),
    ],
)
def test_split_sentences_handles_punctuation_context(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_split_sentences_handles_large_input_without_recursion() -> None:
    sentence = f"{'x' * 200_000}."
    assert split_sentences(sentence) == [sentence]
