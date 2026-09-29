# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic high-precision retrieval-query checks."""

from __future__ import annotations

import re
import unicodedata

from data_designer_retrieval_sdg.models import QuerySurface

_SOURCE_RELATIVE_PATTERNS = (
    re.compile(
        r"\b(?:this|the|that|given|above|below|following|preceding)\s+"
        r"(?:page|slide|document|chart|table|figure|image|diagram|graph|plot|section|excerpt|text)\b"
    ),
    re.compile(r"\b(?:shown|displayed|depicted|illustrated|pictured|listed|presented)\s+(?:here|above|below)\b"),
    re.compile(
        r"\b(?:shown|displayed|depicted|illustrated|pictured|listed|presented)\s+"
        r"(?:in|on)\s+(?:(?:this|the|that|given)\s+)?"
        r"(?:data|page|slide|document|chart|table|figure|image|diagram|graph|plot)\b"
    ),
    re.compile(r"\b(?:above|below|here)\s+(?:shows?|lists?|depicts?|illustrates?|displays?|presents?)\b"),
    re.compile(r"\b(?:is|are|was|were)\s+(?:shown|displayed|depicted|illustrated|pictured|listed|presented)\b"),
    re.compile(r"\b(?:shown|displayed|depicted|illustrated|pictured|listed|presented)\s*[?!.]*$"),
)
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_CHART_CONTEXT_PATTERN = re.compile(r"\b(?:charts?|graphs?)\b")
_BAR_EXTENT_COMPARISON_PATTERN = re.compile(
    r"\b(?:which|whose)\b[^?!.]{0,120}\bbars?\s+"
    r"(?:is|are|extends?|extend)\s+(?:the\s+)?"
    r"(?:longer|shorter|taller|further|farther)\b"
)
_ABSENCE_ANSWER_PATTERN = re.compile(
    r"\bnot\s+(?:specified|provided|stated|mentioned|available)\s+in\s+"
    r"(?:the\s+)?(?:provided\s+|supplied\s+)?(?:material|source|context|document|text|image|page)\b"
)


def normalize_query_text(value: str) -> str:
    """Normalize Unicode, case, and whitespace for deterministic comparison."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def query_rejection_reason(question: str, query_surface: QuerySurface | None = None) -> str | None:
    """Check query wording without access to the source or answer.

    Args:
        question: Generated retrieval query in its original language.
        query_surface: Optional declared query form.

    Returns:
        A stable high-precision rejection reason, or None. This check supplements
        the multilingual model judge; it does not certify query quality.
    """
    normalized_question = normalize_query_text(question)
    if any(pattern.search(normalized_question) for pattern in _SOURCE_RELATIVE_PATTERNS):
        return "source_relative_query"
    if _CHART_CONTEXT_PATTERN.search(normalized_question) and _BAR_EXTENT_COMPARISON_PATTERN.search(
        normalized_question
    ):
        return "visual_mark_reading_query"
    if query_surface in {"instruction", "keyword"} and normalized_question.endswith("?"):
        return "query_surface_mismatch"
    return None


def deterministic_rejection_reason(
    question: str,
    answer: str,
    query_surface: QuerySurface | None = None,
) -> str | None:
    """Return a high-precision rejection reason, if one applies."""
    rejection = query_rejection_reason(question, query_surface)
    if rejection is not None:
        return rejection
    normalized_question = normalize_query_text(question)

    normalized_answer = normalize_query_text(answer).strip(" .,:;!?")
    if _ABSENCE_ANSWER_PATTERN.search(normalized_answer):
        return "answer_does_not_resolve_query"
    answer_tokens = _TOKEN_PATTERN.findall(normalized_answer)
    if not normalized_answer:
        return "empty_answer"
    if len(answer_tokens) >= 4 and _contains_token_sequence(normalized_question, normalized_answer):
        return "answer_repeated_in_query"
    if len(answer_tokens) == 1 and any(character.isdigit() for character in normalized_answer):
        if re.search(rf"(?<!\w){re.escape(normalized_answer)}(?!\w)", normalized_question):
            return "numeric_answer_repeated_in_query"
    return None


def _contains_token_sequence(normalized_text: str, normalized_sequence: str) -> bool:
    """Return whether a normalized token sequence occurs at word boundaries."""
    sequence_tokens = _TOKEN_PATTERN.findall(normalized_sequence)
    if not sequence_tokens:
        return False
    pattern = r"(?<!\w)" + r"\W+".join(re.escape(token) for token in sequence_tokens) + r"(?!\w)"
    return re.search(pattern, normalized_text) is not None
