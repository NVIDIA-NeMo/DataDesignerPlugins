# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sentence-tokenizer adapter used by the retrieval SDG chunker.

This module intentionally isolates the temporary, dependency-free tokenizer.
When a patched NLTK release is available, its implementation can be swapped in
here without changing the chunking pipeline or its callers.
"""

from __future__ import annotations

import re
import unicodedata

_ASCII_SENTENCE_TERMINATORS = frozenset(".!?")
_UNICODE_SENTENCE_TERMINATORS = frozenset("…。！？؟۔։।॥።")
_SENTENCE_TERMINATORS = _ASCII_SENTENCE_TERMINATORS | _UNICODE_SENTENCE_TERMINATORS
_NO_SPACE_SENTENCE_TERMINATORS = frozenset("。！？")
_ASCII_SENTENCE_CLOSERS = frozenset("\"'")
_UNICODE_SENTENCE_CLOSER_CATEGORIES = frozenset({"Pe", "Pf"})
_NONTERMINAL_ABBREVIATIONS = frozenset(
    {
        "a.m.",
        "approx.",
        "co.",
        "corp.",
        "dept.",
        "dr.",
        "e.g.",
        "est.",
        "etc.",
        "fig.",
        "i.e.",
        "inc.",
        "jr.",
        "ltd.",
        "mr.",
        "mrs.",
        "ms.",
        "no.",
        "p.m.",
        "prof.",
        "sr.",
        "st.",
        "vs.",
    }
)
_NAME_TITLE_ABBREVIATIONS = frozenset({"dr.", "jr.", "mr.", "mrs.", "ms.", "prof.", "sr."})
_MAX_PERIOD_TOKEN_LENGTH = 32
_MAX_CONTEXT_LOOKAHEAD = 64


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without external models or downloads.

    The scanner recognizes common English abbreviations and decimal numbers,
    keeps closing quotes and brackets with their sentence, and supports common
    Unicode sentence terminators. Its work is bounded per input character.

    Args:
        text: A single paragraph of text.

    Returns:
        Stripped, non-empty sentences in source order.
    """
    sentences: list[str] = []
    sentence_start = 0
    index = 0

    while index < len(text):
        character = text[index]
        if character not in _SENTENCE_TERMINATORS:
            index += 1
            continue

        boundary_end = index + 1
        while boundary_end < len(text) and text[boundary_end] in _SENTENCE_TERMINATORS:
            boundary_end += 1
        terminator_end = boundary_end
        while boundary_end < len(text) and _is_sentence_closer(text[boundary_end]):
            boundary_end += 1

        has_boundary_spacing = boundary_end == len(text) or text[boundary_end].isspace()
        has_no_space_boundary = any(marker in _NO_SPACE_SENTENCE_TERMINATORS for marker in text[index:terminator_end])
        has_strong_terminator = any(marker != "." for marker in text[index:terminator_end])
        nonterminal_period = (
            character == "." and not has_strong_terminator and _period_is_nonterminal(text, index, boundary_end)
        )
        if (has_boundary_spacing or has_no_space_boundary) and not nonterminal_period:
            sentence = text[sentence_start:boundary_end].strip()
            if sentence:
                sentences.append(sentence)
            sentence_start = boundary_end
            while sentence_start < len(text) and text[sentence_start].isspace():
                sentence_start += 1
            index = sentence_start
            continue

        index = boundary_end

    remainder = text[sentence_start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _is_sentence_closer(character: str) -> bool:
    """Return whether a character closes quoted or bracketed text."""
    return (
        character in _ASCII_SENTENCE_CLOSERS or unicodedata.category(character) in _UNICODE_SENTENCE_CLOSER_CATEGORIES
    )


def _previous_period_token(text: str, period_index: int) -> str:
    """Return the bounded word-like token ending at a period.

    Args:
        text: Text containing the candidate sentence boundary.
        period_index: Index of the candidate period.

    Returns:
        A case-folded token containing letters and periods. The lookup is
        deliberately bounded so adversarial input remains linear to scan.
    """
    token_start = period_index
    lower_bound = max(0, period_index + 1 - _MAX_PERIOD_TOKEN_LENGTH)
    while token_start > lower_bound and (text[token_start - 1].isalpha() or text[token_start - 1] == "."):
        token_start -= 1
    return text[token_start : period_index + 1].casefold()


def _period_is_nonterminal(text: str, period_index: int, boundary_end: int) -> bool:
    """Determine whether a period belongs to a number or abbreviation.

    Args:
        text: Text containing the candidate sentence boundary.
        period_index: Index of the candidate period.
        boundary_end: Index immediately after trailing punctuation and closers.

    Returns:
        ``True`` when the period should not end a sentence.
    """
    if period_index > 0 and period_index + 1 < len(text):
        if text[period_index - 1].isdigit() and text[period_index + 1].isdigit():
            return True

    next_word = _next_context_word(text, boundary_end)
    if not next_word:
        return False

    token = _previous_period_token(text, period_index)
    is_abbreviation = token in _NONTERMINAL_ABBREVIATIONS
    is_initial = re.fullmatch(r"[a-z]\.", token) is not None
    is_unlisted_multi_initial = (
        token not in _NONTERMINAL_ABBREVIATIONS and re.fullmatch(r"(?:[a-z]\.){2,}", token) is not None
    )
    if not (is_abbreviation or is_initial or is_unlisted_multi_initial):
        return False
    if next_word[0].islower() or next_word[0].isdigit():
        return True
    return token in _NAME_TITLE_ABBREVIATIONS or is_unlisted_multi_initial


def _next_context_word(text: str, boundary_end: int) -> str:
    """Return a bounded lookahead word following a candidate boundary.

    Args:
        text: Text containing the candidate sentence boundary.
        boundary_end: Index immediately after trailing punctuation and closers.

    Returns:
        The next alphanumeric word, or an empty string when none occurs within
        the bounded lookahead window.
    """
    index = boundary_end
    lookahead_end = min(len(text), boundary_end + _MAX_CONTEXT_LOOKAHEAD)
    while index < lookahead_end and not text[index].isalnum():
        index += 1

    word_start = index
    while index < lookahead_end and text[index].isalnum():
        index += 1
    return text[word_start:index]
