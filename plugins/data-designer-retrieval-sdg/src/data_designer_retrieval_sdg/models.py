# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for structured LLM outputs in the retriever SDG pipeline.

These models define the schemas for artifact extraction, QA generation,
and quality evaluation columns.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

QuerySurface = Literal["question", "instruction", "keyword"]
EvidenceModality = Literal["text_only", "image_grounded"]

# ---------------------------------------------------------------------------
# Artifact extraction models
# ---------------------------------------------------------------------------


class ArtifactItem(BaseModel):
    """A single artifact item with text, description, and importance."""

    text: str = Field(description="The artifact text or name")
    description: str = Field(description="Detailed description of the artifact")
    importance: str = Field(description="Why this artifact is important")


class RetrievalProfile(BaseModel):
    """Advisory source suitability, distinct from candidate grounding."""

    text_retrieval_useful: bool = Field(description="Readable content supports a scoped information need")
    visual_retrieval_useful: bool = Field(description="Non-decorative visual relationships support useful queries")
    visual_content_types: list[Literal["chart", "table", "diagram", "layout", "photograph"]]
    suggested_query_operations: list[str] = Field(description="Source-specific, verifiable retrieval operations")
    reason: str = Field(min_length=1)


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
    retrieval_profile: RetrievalProfile | None = Field(
        default=None, description="Source suitability; absence means unknown, not unsuitable"
    )


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

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        description=("The question requiring understanding of contexts without explicitly referencing them"),
    )
    answer: str = Field(
        description=("Comprehensive answer from the contexts without explicitly referencing them"),
    )
    evidence: str = Field(
        min_length=1,
        description="Concise source evidence supporting the answer",
    )
    query_surface: QuerySurface = Field(
        description="Actual query form: natural question, search instruction, or compact keyword query",
    )
    positive_unit_ids: list[str] = Field(
        min_length=1,
        description="Stable retrieval-unit identifiers that should be positives for the query",
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

    @field_validator("positive_unit_ids")
    @classmethod
    def validate_positive_unit_ids(cls, values: list[str]) -> list[str]:
        """Reject blank or duplicate positive-unit identifiers."""
        if any(not value.strip() for value in values):
            raise ValueError("positive_unit_ids must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("positive_unit_ids must be unique")
        return values


class QuestionAnswerPairs(BaseModel):
    """Collection of question-answer pairs."""

    pairs: list[QuestionAnswerPair] = Field(description="List of question-answer pairs")


class RetrievalQuery(BaseModel):
    """A retrieval query generated before its answer or evidence."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1, description="Retrieval query in the declared surface; need not be interrogative"
    )
    query_surface: QuerySurface
    question_complexity: int = Field(ge=1, le=5)
    query_type: Literal["multi_hop", "structural", "contextual"] = Field(
        description="Query structure: ONLY multi_hop, structural, or contextual. Never use factual or visual here."
    )
    reasoning_type: Literal["factual", "relational", "inferential", "temporal", "procedural", "visual", "causal"] = (
        Field(description="Reasoning operation, independent of query_type")
    )


class RetrievalQueries(BaseModel):
    """Queries only; an empty list is valid when the source is unsuitable."""

    queries: list[RetrievalQuery]


class RetrievalAnswer(BaseModel):
    """Evidence for an immutable query identified by its selected-list index."""

    model_config = ConfigDict(extra="forbid")

    candidate_index: int = Field(ge=0)
    answer: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    positive_unit_ids: list[str] = Field(min_length=1)
    segment_ids: list[int]
    hop_count: int = Field(ge=1)
    hop_contexts: list[HopContext]


class RetrievalAnswers(BaseModel):
    """Answerable selected queries; unsupported queries may be omitted."""

    answers: list[RetrievalAnswer] = Field(description="Omit unanswerable queries; an empty list is valid")


class GenerationDiagnostic(BaseModel):
    """A candidate rejected before grounding, indexed in deduplicated queries."""

    candidate_index: int = Field(ge=0)
    question: str
    query_surface: QuerySurface
    rejection_reason: str


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


class NativeTextEvidence(BaseModel):
    """A judge-selected verbatim quote from one positive unit's supplied text."""

    unit_id: str = Field(min_length=1, description="Declared positive unit containing the quoted native text")
    quote: str = Field(
        min_length=1,
        description=(
            "A short verbatim supplied-text span supporting the answer, never generated evidence or pixels. "
            "Do not reproduce decorative dot leaders or other repeated formatting; select informative exact spans. "
            "Use separate quotes for separate spans, never insert ellipses or rewrite the source wording."
        ),
    )


class SourceAssessment(BaseModel):
    """An answer-blind reading of the query's declared positive source units."""

    model_config = ConfigDict(extra="forbid")

    candidate_index: int = Field(ge=0)
    positive_unit_ids: list[str] = Field(min_length=1)
    answerable: bool
    independent_answer: str | None = Field(
        description=(
            "Answer independently read from the source, or null when it cannot be established. "
            "Encode numeric answers as strings, never bare JSON numbers."
        )
    )
    supporting_evidence: str | None = Field(description="Source-bound support for the independent answer, or null")
    uncertainty_reasons: list[str] = Field(description="Material unreadable, ambiguous or unsupported information")
    positive_source_relevant: bool
    evidence_modality: EvidenceModality = Field(
        description=(
            "text_only requires supplied text to establish every material fact AND relationship without pixels. "
            "Matching labels alone do not establish layout, containment, direction or label-value alignment. "
            "Use image_grounded when pixels are necessary to establish those relationships."
        )
    )
    source_language_preserved: bool
    native_text_evidence: list[NativeTextEvidence]

    @field_validator("positive_unit_ids")
    @classmethod
    def validate_positive_unit_ids(cls, values: list[str]) -> list[str]:
        """Reject blank or duplicate source identifiers."""
        if any(not value.strip() for value in values) or len(values) != len(set(values)):
            raise ValueError("assessment positive_unit_ids must be nonblank and unique")
        return values

    @property
    def passes_source_assessment(self) -> bool:
        """Require a supported independent answer, not a confident abstention."""
        return (
            self.answerable
            and self.positive_source_relevant
            and self.source_language_preserved
            and not self.uncertainty_reasons
            and bool(self.independent_answer and self.independent_answer.strip())
            and bool(self.supporting_evidence and self.supporting_evidence.strip())
        )


class SourceAssessments(BaseModel):
    """Independent readings of all surviving candidates in a source row."""

    assessments: list[SourceAssessment]


class QAEvaluation(BaseModel):
    """Evaluation of a single QA pair."""

    candidate_index: int = Field(ge=0)
    relevance: QAEvaluationCriterion = Field(description="Relevance of question to context")
    accuracy: QAEvaluationCriterion = Field(description="Factual accuracy of answer")
    context_support: QAEvaluationCriterion = Field(
        description="How well answer is supported by context",
    )
    clarity: QAEvaluationCriterion = Field(description="Clarity and unambiguity of question")
    overall: QAOverallEvaluation = Field(description="Overall evaluation")
    improvements: str = Field(description="Suggestions for improving this QA pair")
    answer_grounded: bool
    answer_resolves_query: bool = Field(
        description="The answer supplies the requested information, not merely a statement that it is absent"
    )
    unsupported_claims: list[str]
    positive_source_relevant: bool
    evidence_modality: EvidenceModality = Field(
        description=(
            "text_only requires support in the positive unit's supplied text field, not generated answer/evidence. "
            "If that field is empty and pixels support the answer, use image_grounded, even for printed text."
        )
    )
    answer_not_revealed_by_query: bool
    source_language_preserved: bool
    native_text_evidence: list[NativeTextEvidence] = Field(
        default_factory=list,
        description=(
            "For text_only, verbatim supplied-text quotes covering every positive and every material answer claim. "
            "Do not quote pixels, generated answers or claimed evidence. Empty for unsupported or image-only support."
        ),
    )

    @property
    def passes_grounding(self) -> bool:
        """Return whether all grounding-specific criteria pass."""
        return (
            self.answer_grounded
            and self.answer_resolves_query
            and not self.unsupported_claims
            and self.positive_source_relevant
            and self.answer_not_revealed_by_query
            and self.source_language_preserved
        )


class QAPairEvaluations(BaseModel):
    """Evaluations for all QA pairs in a document."""

    evaluations: list[QAEvaluation] = Field(
        description="List of evaluations, one per QA pair, in the same order as the QA pairs",
    )


class QueryQualityEvaluation(BaseModel):
    """Source-blind judgement of one generated retrieval query."""

    candidate_index: int = Field(ge=0)
    reason: str = Field(
        min_length=1,
        description=(
            "Before deciding, identify the requested fact and check its subject, event, period and jurisdiction. "
            "For each material anchor, identify words in the query supplying it or state that it is missing. "
            "Explain why an anchor is unnecessary for a genuinely evergreen or general need. "
            "Do not substitute a plausible imagined context for a missing anchor."
        ),
    )
    standalone_query: bool
    plausible_information_need: bool
    retrieval_discriminative: bool = Field(
        description=(
            "False if ANY material subject, event, period, jurisdiction or comparison scope is missing. "
            "A recurring ranking or time-varying comparison needs an explicit period even without words like latest. "
            "A government policy or regulated price statistic needs an identified jurisdiction, not just a year. "
            "Query language does not supply a country; a company name plus date does not identify an event. "
            "Financial facts need the reporting entity or transaction. Counterparty names, even several, "
            "do not by themselves identify an issuer, client or transaction. "
            "Do not infer missing scope. Genuinely evergreen or general conceptual needs require no artificial anchors."
        )
    )
    query_surface_correct: bool
    source_language_preserved: bool

    @property
    def passes(self) -> bool:
        """Return whether every source-blind criterion passes."""
        return (
            self.standalone_query
            and self.plausible_information_need
            and self.retrieval_discriminative
            and self.query_surface_correct
            and self.source_language_preserved
        )


class QueryQualityEvaluations(BaseModel):
    """Source-blind judgements for a deduplicated candidate list."""

    evaluations: list[QueryQualityEvaluation] = Field(default_factory=list)
