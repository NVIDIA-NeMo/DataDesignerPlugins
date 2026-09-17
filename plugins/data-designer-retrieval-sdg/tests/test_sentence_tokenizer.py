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


@pytest.mark.parametrize(
    "second",
    [
        "It was difficult.",
        "He disagreed.",
        "She left.",
        "They returned.",
        "We left.",
        "This helped.",
        "The trip ended.",
        "Everyone agreed.",
    ],
)
@pytest.mark.parametrize(
    "first",
    ["They moved to the U.S.", "They returned from the U.K.", "We met John Smith Jr.", "We met John Smith Sr."],
)
def test_split_sentences_ends_acronyms_and_name_suffixes(first: str, second: str) -> None:
    assert split_sentences(f"{first} {second}") == [first, second]


@pytest.mark.parametrize("starter", ["Bob", "Sarah", "Meetings", "Customers", "Everyone", "Then"])
@pytest.mark.parametrize(
    "first",
    [
        "We hired Acme Inc.",
        "The answer was no.",
        "Bring paper, pens, etc.",
        "Sentence A.",
        "We met John Smith Jr.",
        "We met John Smith Sr.",
    ],
)
def test_split_sentences_does_not_require_a_sentence_starter_allowlist(first: str, starter: str) -> None:
    second = f"{starter} left."
    assert split_sentences(f"{first} {second}") == [first, second]


@pytest.mark.parametrize(
    "text",
    [
        "We visited St. Paul yesterday.",
        "The church is in St. Louis.",
        "We met Dr. Smith today.",
        "Prof. Ada Lovelace joined us.",
        "The U.S. Army arrived.",
        "The U.S.S.R. Delegation arrived.",
        "The U.K. Parliament voted.",
        "We met John Smith Jr. yesterday.",
        "John Smith Jr., our host, arrived.",
        "Choose a city, e.g. London or Paris.",
        "We mean the capital, i.e. Paris.",
        "The case was Smith vs. Jones.",
        "The meeting begins at 3 p.m. sharp.",
        "Use Fig. 2 for reference.",
        "We hired Acme Inc. for the job.",
        "Dr. J. Smith arrived.",
        "J. R. R. Tolkien wrote books.",
        "J. A. Smith arrived.",
        "J. I. Packer wrote books.",
        "J. Smith arrived.",
        "A. Smith arrived.",
        "John B. Smith arrived.",
    ],
)
def test_split_sentences_keeps_abbreviations_in_continuations(text: str) -> None:
    assert split_sentences(text) == [text]


@pytest.mark.parametrize("label", ["Sentence", "sentence", "Option", "Appendix", "Section", "Vitamin"])
def test_split_sentences_distinguishes_letter_labels_from_name_initials(label: str) -> None:
    assert split_sentences(f"{label} A. Bob replied.") == [f"{label} A.", "Bob replied."]


@pytest.mark.parametrize("text", ["", " \t\n", "\u2003\n"])
def test_split_sentences_skips_empty_input(text: str) -> None:
    assert split_sentences(text) == []


@pytest.mark.parametrize("separator", [" ", "\t", "\n", "\u2003"])
def test_split_sentences_keeps_source_text_and_handles_boundary_whitespace(separator: str) -> None:
    sentences = ['She said "We moved to the U.S."', "This was good.", "「はい。」", "We met Dr. Smith."]
    text = separator + separator.join(sentences) + separator
    assert split_sentences(text) == sentences


@pytest.mark.parametrize("first", ["Is she a Dr.?", "The U.S.!", "Ask Mr.?!", "The U.S.！"])
def test_split_sentences_strong_terminators_override_all_abbreviation_roles(first: str) -> None:
    assert split_sentences(f"{first} Yes.") == [first, "Yes."]


@pytest.mark.parametrize(
    "repeated",
    ["Dr. Smith arrived. ", "x" * 10_000 + ". ", "." * 10_000 + " "],
    ids=["sentences", "long-word", "punctuation-run"],
)
def test_split_sentences_preserves_all_text_in_repeated_inputs(repeated: str) -> None:
    text = repeated * 20
    sentences = split_sentences(text)
    assert "".join(text.split()) == "".join("".join(sentences).split())
