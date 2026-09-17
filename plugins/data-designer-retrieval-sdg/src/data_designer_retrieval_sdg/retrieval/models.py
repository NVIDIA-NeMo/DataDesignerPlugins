# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts for modality-neutral retrieval data generation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_designer_retrieval_sdg.models import (
    DocumentArtifacts,
    EvidenceModality,
    GenerationDiagnostic,
    QAPairEvaluations,
    QueryQualityEvaluations,
    QuerySurface,
    QuestionAnswerPair,
    SourceAssessments,
)

SplitName = Literal["train", "validation", "evaluation"]
ViewName = Literal["text", "image", "image_and_text"]


class StrictModel(BaseModel):
    """Immutable model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RetrievalUnit(StrictModel):
    """One independently retrievable source unit."""

    unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = ""
    images: list[str] = Field(default_factory=list, max_length=1)
    source_uri: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    segment_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_content(self) -> RetrievalUnit:
        """Require text or image content."""
        if not self.text.strip() and not self.images:
            raise ValueError("a retrieval unit must contain text or at least one image")
        if any(not image.strip() for image in self.images):
            raise ValueError("retrieval-unit image paths must not be blank")
        return self


class RetrievalSource(StrictModel):
    """Convenience input for a single retrieval unit."""

    document_id: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    text: str = ""
    images: list[str] = Field(default_factory=list, max_length=1)
    language: str = Field(default="source", min_length=1)
    source_uri: str | None = None
    page_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_content(self) -> RetrievalSource:
        """Require text or image content."""
        if not self.text.strip() and not self.images:
            raise ValueError("a retrieval source must contain text or at least one image")
        if any(not image.strip() for image in self.images):
            raise ValueError("retrieval-source image paths must not be blank")
        return self

    def to_seed_record(self) -> dict[str, Any]:
        """Return a canonical Data Designer seed row."""
        source_uri = self.source_uri or self.document_id
        unit = RetrievalUnit(
            unit_id=self.unit_id,
            document_id=self.document_id,
            text=self.text,
            images=self.images,
            source_uri=source_uri,
            page_number=self.page_number,
            segment_id=1,
        )
        display_text = self.text if self.text.strip() else "[image-only retrieval unit]"
        return {
            "file_name": [source_uri],
            "source_id": self.unit_id,
            "document_id": self.document_id,
            "unit_id": self.unit_id,
            "text": self.text,
            "chunks": [
                {
                    "chunk_id": 1,
                    "text": self.text,
                    "doc_id": self.document_id,
                    "doc_path": source_uri,
                }
            ],
            "sections_structured": [f"Segment 1 [Unit: {self.unit_id}] [Doc: {self.document_id}]: {display_text}"],
            "retrieval_units": [unit.model_dump(mode="json")],
            "images": list(self.images),
            "language": self.language,
            "source_uri": source_uri,
            "page_number": self.page_number,
            "bundle_id": "",
            "bundle_members": [source_uri],
            "is_multi_doc": False,
        }


class GeneratedRetrievalRecord(BaseModel):
    """One completed shared-pipeline row ready for recipe export."""

    source_id: str = Field(min_length=1)
    retrieval_units: list[RetrievalUnit] = Field(min_length=1)
    deduplicated_qa_pairs: list[QuestionAnswerPair]
    query_quality_evaluations: QueryQualityEvaluations
    qa_evaluations: QAPairEvaluations
    source_assessments: SourceAssessments | None = None
    document_artifacts: DocumentArtifacts | None = None
    generation_diagnostics: list[GenerationDiagnostic] = Field(default_factory=list)
    language: str = Field(default="source", min_length=1)
    model_config = ConfigDict(extra="ignore", frozen=True, str_strip_whitespace=True)

    @model_validator(mode="after")
    def validate_evaluation_indexes(self) -> GeneratedRetrievalRecord:
        """Reject duplicate or out-of-range explicit evaluation indexes."""
        candidate_count = len(self.deduplicated_qa_pairs)
        query_indexes = [item.candidate_index for item in self.query_quality_evaluations.evaluations]
        if len(query_indexes) != len(set(query_indexes)):
            raise ValueError("query evaluations contain duplicate candidate indexes")
        grounding_indexes = [item.candidate_index for item in self.qa_evaluations.evaluations]
        if len(grounding_indexes) != len(set(grounding_indexes)):
            raise ValueError("grounding evaluations contain duplicate candidate indexes")
        source_indexes = (
            [item.candidate_index for item in self.source_assessments.assessments]
            if self.source_assessments is not None
            else []
        )
        if len(source_indexes) != len(set(source_indexes)):
            raise ValueError("source assessments contain duplicate candidate indexes")
        for label, indexes in (
            ("query", query_indexes),
            ("grounding", grounding_indexes),
            ("source assessment", source_indexes),
        ):
            if any(index >= candidate_count for index in indexes):
                raise ValueError(f"{label} evaluation candidate index is out of range")
        return self


class SplitRatios(StrictModel):
    """Document-level train, validation, and evaluation ratios."""

    train: float = Field(default=0.8, ge=0.0, le=1.0)
    validation: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluation: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> SplitRatios:
        """Require ratios to sum to one."""
        if abs(self.train + self.validation + self.evaluation - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return self


class CandidateDiagnostic(StrictModel):
    """Auditable acceptance decision for one generated candidate."""

    source_id: str
    candidate_index: int
    question: str
    answer: str
    positive_unit_ids: list[str]
    query_surface: QuerySurface
    evidence_modality: EvidenceModality | None
    accepted: bool
    rejection_reason: str | None = None


class ExportSummary(StrictModel):
    """Summary and paths for a completed recipe-oriented export."""

    output_dir: str
    run_manifest_path: str
    source_record_count: int
    retrieval_unit_count: int
    accepted_candidate_count: int
    rejected_candidate_count: int
    document_split_counts: dict[str, int]
    candidate_split_counts: dict[str, int]
    query_surface_counts: dict[str, int]
    evidence_modality_counts: dict[str, int]
    warnings: list[str]
