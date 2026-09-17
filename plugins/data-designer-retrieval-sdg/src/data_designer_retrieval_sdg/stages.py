# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic selection and immutable query/answer assembly."""

from __future__ import annotations

from collections import Counter
from typing import Any

from data_designer.config import custom_column_generator

from data_designer_retrieval_sdg.models import (
    GenerationDiagnostic,
    QueryQualityEvaluations,
    QuestionAnswerPair,
    RetrievalAnswers,
    RetrievalQuery,
)
from data_designer_retrieval_sdg.retrieval.quality import query_rejection_reason


@custom_column_generator(required_columns=["deduplicated_queries", "source_blind_evaluations"])
def select_retrieval_queries(row: dict[str, Any]) -> dict[str, Any]:
    """Select queries that pass deterministic and independent model checks.

    Args:
        row: Data Designer row containing deduplicated queries and judgements.

    Returns:
        Row with a query_selection audit containing selected and rejected queries.
        Missing or duplicate judgements fail closed for the affected query.
    """
    queries = [RetrievalQuery.model_validate(item) for item in row["deduplicated_queries"]]
    raw_judgements = row.get("source_blind_evaluations") if queries else {"evaluations": []}
    judgements = QueryQualityEvaluations.model_validate(raw_judgements)
    counts = Counter(item.candidate_index for item in judgements.evaluations)
    by_index = {item.candidate_index: item for item in judgements.evaluations}
    selected, rejected = [], []
    for index, query in enumerate(queries):
        reason = query_rejection_reason(query.question, query.query_surface)
        judgement = by_index.get(index)
        if reason is None and counts[index] != 1:
            reason = "missing_or_duplicate_query_judgement"
        if reason is None and judgement is not None and not judgement.passes:
            reason = "query_quality_rejected: " + judgement.reason
        if reason is not None:
            rejected.append(_diagnostic(index, query, reason))
        else:
            selected.append({"original_index": index, "query": query.model_dump(), "judgement": judgement.model_dump()})
    return {**row, "query_selection": {"selected": selected, "rejected": rejected}}


@custom_column_generator(
    required_columns=["query_selection", "answer_generation"],
    side_effect_columns=["query_quality_evaluations", "generation_diagnostics"],
)
def assemble_retrieval_pairs(row: dict[str, Any]) -> dict[str, Any]:
    """Join answers to immutable queries and reindex surviving judgements.

    Args:
        row: Data Designer row containing selected queries and generated answers.

    Returns:
        Row with canonical QA pairs, aligned query judgements and rejection audit.
        Missing/duplicate answers are quarantined; unknown answer indexes fail closed.

    Raises:
        ValueError: If an answer refers to a nonexistent selected query.
    """
    selection = row["query_selection"]
    selected = selection["selected"]
    payload = row.get("answer_generation")
    answers = RetrievalAnswers.model_validate(payload).answers if selected else []
    if any(answer.candidate_index >= len(selected) for answer in answers):
        raise ValueError("answer candidate index is out of range")
    counts = Counter(answer.candidate_index for answer in answers)
    by_index = {answer.candidate_index: answer for answer in answers}
    pairs, evaluations = [], []
    rejected = list(selection["rejected"])
    for index, item in enumerate(selected):
        query = RetrievalQuery.model_validate(item["query"])
        if counts[index] != 1:
            rejected.append(_diagnostic(item["original_index"], query, "missing_or_duplicate_answer"))
            continue
        pair = QuestionAnswerPair.model_validate(
            {**query.model_dump(), **by_index[index].model_dump(exclude={"candidate_index"})}
        )
        evaluations.append({**item["judgement"], "candidate_index": len(pairs)})
        pairs.append(pair.model_dump())
    return {
        **row,
        "deduplicated_qa_pairs": pairs,
        "query_quality_evaluations": {"evaluations": evaluations},
        "generation_diagnostics": rejected,
    }


def _diagnostic(index: int, query: RetrievalQuery, reason: str) -> dict[str, Any]:
    """Build a rejection record without inventing an answer or positive unit."""
    return GenerationDiagnostic(
        candidate_index=index,
        question=query.question,
        query_surface=query.query_surface,
        rejection_reason=reason,
    ).model_dump()
