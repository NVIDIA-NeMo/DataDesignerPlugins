# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer_retrieval_sdg.models import (
    ArtifactItem,
    DocumentArtifacts,
    HopContext,
    QAEvaluation,
    QAEvaluationCriterion,
    QAOverallEvaluation,
    QAPairEvaluations,
    QuestionAnswerPair,
    QuestionAnswerPairs,
)


def test_artifact_item_round_trip() -> None:
    item = ArtifactItem(text="concept", description="a concept", importance="high")
    assert item.text == "concept"
    data = item.model_dump()
    assert ArtifactItem.model_validate(data) == item


def test_document_artifacts_defaults() -> None:
    artifacts = DocumentArtifacts()
    assert artifacts.key_concepts == []
    assert artifacts.technical_terms == []


def test_question_answer_pair() -> None:
    pair = QuestionAnswerPair(
        question="What?",
        answer="This.",
        question_complexity=4,
        query_type="multi_hop",
        reasoning_type="factual",
        segment_ids=[1, 3],
        hop_count=2,
        hop_contexts=[
            HopContext(hop_number=1, segment_ids=[1], summary="first"),
            HopContext(hop_number=2, segment_ids=[3], summary="second"),
        ],
    )
    assert pair.query_type == "multi_hop"
    assert len(pair.hop_contexts) == 2


def test_question_answer_pairs_container() -> None:
    pairs = QuestionAnswerPairs(
        pairs=[
            QuestionAnswerPair(
                question="Q1",
                answer="A1",
                question_complexity=4,
                query_type="structural",
                reasoning_type="relational",
                segment_ids=[2],
                hop_count=1,
                hop_contexts=[],
            )
        ]
    )
    assert len(pairs.pairs) == 1


def test_qa_evaluation_round_trip() -> None:
    evl = QAEvaluation(
        relevance=QAEvaluationCriterion(score=8, justification="relevant"),
        accuracy=QAEvaluationCriterion(score=9, justification="accurate"),
        context_support=QAEvaluationCriterion(score=7, justification="supported"),
        clarity=QAEvaluationCriterion(score=8, justification="clear"),
        overall=QAOverallEvaluation(score=8.0, assessment="good"),
        improvements="none",
    )
    data = evl.model_dump()
    assert QAEvaluation.model_validate(data).overall.score == 8.0


def test_qa_pair_evaluations() -> None:
    evals = QAPairEvaluations(evaluations=[])
    assert evals.evaluations == []
