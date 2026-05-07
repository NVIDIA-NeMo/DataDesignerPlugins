# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for structured LLM outputs in the retriever SDG pipeline.

These models define the schemas for artifact extraction, QA generation,
and quality evaluation columns.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Artifact extraction models
# ---------------------------------------------------------------------------


class ArtifactItem(BaseModel):
    """A single artifact item with text, description, and importance."""

    text: str = Field(description="The artifact text or name")
    description: str = Field(description="Detailed description of the artifact")
    importance: str = Field(description="Why this artifact is important")


class DocumentArtifacts(BaseModel):
    """Semantic artifacts extracted from a document."""

    key_concepts: list[ArtifactItem] = Field(default_factory=list, description="Key concepts in the document")
    relationships: list[ArtifactItem] = Field(default_factory=list, description="Relationships between concepts")
    themes: list[ArtifactItem] = Field(default_factory=list, description="Main themes")
    entities: list[ArtifactItem] = Field(default_factory=list, description="Entities mentioned")
    processes: list[ArtifactItem] = Field(default_factory=list, description="Processes described")
    insights: list[ArtifactItem] = Field(default_factory=list, description="Key insights")
    technical_terms: list[ArtifactItem] = Field(default_factory=list, description="Technical terms")
    contextual_factors: list[ArtifactItem] = Field(default_factory=list, description="Contextual factors")


# ---------------------------------------------------------------------------
# QA generation models
# ---------------------------------------------------------------------------


class HopContext(BaseModel):
    """Context for a single hop in a multi-hop question."""

    hop_number: int = Field(description="The hop number (1-indexed)")
    segment_ids: list[int] = Field(description="Segment IDs for this hop")
    summary: str = Field(description="Summary of the supporting segments for this hop")


class QuestionAnswerPair(BaseModel):
    """A single question-answer pair with metadata."""

    question: str = Field(
        description=("The question requiring understanding of contexts without explicitly referencing them"),
    )
    answer: str = Field(
        description=("Comprehensive answer from the contexts without explicitly referencing them"),
    )
    question_complexity: int = Field(description="Numeric score from min_complexity to 5")
    query_type: Literal["multi_hop", "structural", "contextual"] = Field(
        description="Type of query, one of multi_hop, structural, or contextual",
    )
    reasoning_type: Literal["factual", "relational", "inferential", "temporal", "procedural", "visual", "causal"] = (
        Field(
            description=(
                "Type of reasoning required, one of factual, relational, inferential, "
                "temporal, procedural, visual, or causal"
            ),
        )
    )
    segment_ids: list[int] = Field(
        description="List of segment IDs that are source material for this question",
    )
    hop_count: int = Field(
        description=("Number of hops (min_hops to max_hops) for multi_hop questions, or 1 for non-multi-hop"),
    )
    hop_contexts: list[HopContext] = Field(description="Array of hop detail objects")


class QuestionAnswerPairs(BaseModel):
    """Collection of question-answer pairs."""

    pairs: list[QuestionAnswerPair] = Field(description="List of question-answer pairs")


# ---------------------------------------------------------------------------
# QA evaluation models
# ---------------------------------------------------------------------------


class QAEvaluationCriterion(BaseModel):
    """Evaluation criterion with score and justification."""

    score: int = Field(description="Score from 1-10")
    justification: str = Field(description="Brief justification for the score")


class QAOverallEvaluation(BaseModel):
    """Overall evaluation with score and assessment."""

    score: float = Field(description="Overall score from 1-10")
    assessment: str = Field(description="Final assessment of the QA pair")


class QAEvaluation(BaseModel):
    """Evaluation of a single QA pair."""

    relevance: QAEvaluationCriterion = Field(description="Relevance of question to context")
    accuracy: QAEvaluationCriterion = Field(description="Factual accuracy of answer")
    context_support: QAEvaluationCriterion = Field(
        description="How well answer is supported by context",
    )
    clarity: QAEvaluationCriterion = Field(description="Clarity and unambiguity of question")
    overall: QAOverallEvaluation = Field(description="Overall evaluation")
    improvements: str = Field(description="Suggestions for improving this QA pair")


class QAPairEvaluations(BaseModel):
    """Evaluations for all QA pairs in a document."""

    evaluations: list[QAEvaluation] = Field(
        description="List of evaluations, one per QA pair, in the same order as the QA pairs",
    )
