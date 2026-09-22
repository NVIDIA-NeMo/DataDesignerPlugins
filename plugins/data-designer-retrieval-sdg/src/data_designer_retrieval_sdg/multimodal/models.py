# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-independent contracts for the retrieval-first multimodal workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_designer_retrieval_sdg.retrieval.models import SplitRatios
from data_designer_retrieval_sdg.retrieval.query_groups import QueryProvenance


class StrictModel(BaseModel):
    """Reject unknown configuration and response fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class ModelSettings(StrictModel):
    """Explicit OpenAI-compatible model settings; only a credential variable name is stored."""

    model: str = Field(min_length=1)
    endpoint: str = Field(pattern=r"^https?://")
    credential_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    temperature: float = Field(default=0.6, ge=0, le=2)
    max_tokens: int = Field(default=8192, gt=0)
    timeout: int = Field(default=600, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class QueryInstruction(StrictModel):
    """A requested information need; matching its style is diagnostic, not a quality gate."""

    name: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    query_type: str | None = None
    format: Literal["question", "instruction", "keyword"] | None = None
    modality: Literal["text", "image", "text_and_image"] | None = None
    persona: str | None = None
    answerability: str | None = None


DEFAULT_INSTRUCTIONS = (
    QueryInstruction(name="text", instruction="Ask for substantive facts or reasoning supported by source text."),
    QueryInstruction(name="figure", instruction="Ask about evidence in a chart or diagram, if present."),
    QueryInstruction(name="table", instruction="Ask about substantive evidence in a table, if present."),
)


class GenerationContext(StrictModel):
    """Explicit source-unit membership for a request, independent of query split groups."""

    context_id: str = Field(min_length=1)
    unit_ids: list[str] = Field(min_length=1)
    language: str = Field(default="source", min_length=1)

    @model_validator(mode="after")
    def unique_units(self) -> GenerationContext:
        """Reject repeated or empty source identities."""
        if len(set(self.unit_ids)) != len(self.unit_ids) or any(not key.strip() for key in self.unit_ids):
            raise ValueError("context unit IDs must be unique and nonempty")
        return self


class MultimodalSDGConfig(StrictModel):
    """Bounded generation and portable export configuration; no dataset adapters or model defaults."""

    sources_file: Path
    output_dir: Path
    dataset_id: str = Field(min_length=1)
    generator: ModelSettings
    judge: ModelSettings
    contexts_file: Path | None = None
    concurrency: int = Field(default=8, ge=1, le=128)
    batch_size: int = Field(default=30, ge=1, le=128)
    max_units_per_context: int = Field(default=8, ge=1)
    context_strategy: Literal["unit", "document", "sections"] = "unit"
    section_size: int = Field(default=5, ge=1)
    combination_iterations: int = Field(default=0, ge=0, le=100)
    summary_embedding_model: str | None = None
    summary_embedding_revision: str | None = None
    summary_embedding_device: str = "cpu"
    summary_embedding_endpoint: str | None = Field(default=None, pattern=r"^https?://")
    summary_embedding_credential_env: str = Field(default="NVIDIA_API_KEY", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    summary_embedding_extra_body: dict[str, Any] = Field(default_factory=dict)
    persona: str = "A reader seeking specific evidence and useful information from this corpus."
    max_context_chars: int = Field(default=100000, ge=1)
    related_contexts_per_context: int = Field(default=0, ge=0, le=2)
    related_summary_similarity: float = Field(default=0.2, gt=0, le=1)
    judge_summaries: bool = True
    summary_quality_threshold: int = Field(default=4, ge=1, le=5)
    summary_count: int | None = Field(default=None, gt=0)
    summary_fraction: float | None = Field(default=None, gt=0, le=1)
    summary_near_duplicate_threshold: float | None = Field(default=None, gt=0, le=1)
    instructions_per_context: int | None = Field(default=None, gt=0)
    missing_response_attempts: int = Field(default=3, ge=1, le=3)
    require_verbatim_quotes: bool = False
    instructions: list[QueryInstruction] = Field(default_factory=lambda: list(DEFAULT_INSTRUCTIONS), min_length=1)
    relevance_threshold: int = Field(default=4, ge=1, le=5)
    self_sufficiency_threshold: int = Field(default=4, ge=1, le=5)
    ratios: SplitRatios = Field(default_factory=SplitRatios)
    seed: int = 42
    group_near_duplicates: bool = False
    resume: bool = False

    @model_validator(mode="after")
    def unique_instructions(self) -> MultimodalSDGConfig:
        """Require stable, unambiguous requested slot identities."""
        if len({item.name for item in self.instructions}) != len(self.instructions):
            raise ValueError("instruction names must be unique")
        if self.summary_count is not None and self.summary_fraction is not None:
            raise ValueError("summary_count and summary_fraction are mutually exclusive")
        if self.instructions_per_context is not None and self.instructions_per_context > len(self.instructions):
            raise ValueError("instructions_per_context exceeds the instruction pool")
        if self.combination_iterations and self.context_strategy != "sections":
            raise ValueError("semantic combinations require context_strategy=sections")
        if self.combination_iterations and not self.summary_embedding_model:
            raise ValueError("summary_embedding_model is required for semantic combinations")
        if self.summary_embedding_endpoint and self.summary_embedding_revision:
            raise ValueError("summary_embedding_revision applies only to local embeddings")
        if {"model", "input", "encoding_format"}.intersection(self.summary_embedding_extra_body):
            raise ValueError("summary_embedding_extra_body cannot override model, input or encoding_format")
        if self.summary_embedding_extra_body and not self.summary_embedding_endpoint:
            raise ValueError("summary_embedding_extra_body requires summary_embedding_endpoint")
        if self.combination_iterations and self.related_contexts_per_context:
            raise ValueError("semantic combinations and lexical related contexts are mutually exclusive")
        return self


class ContextSummary(StrictModel):
    """Context enrichment for query generation, never a replacement for source evidence."""

    summary: str
    visual_evidence: str


class Grade(StrictModel):
    """One independently recorded summary quality criterion."""

    grade: int = Field(ge=1, le=5)
    explanation: str


class SummaryJudgment(StrictModel):
    """Four reference summary grades; all must meet the configured threshold."""

    information_richness: Grade
    persona_relevance: Grade
    query_generation_potential: Grade
    conceptual_clarity: Grade


class Description(StrictModel):
    """A concise description used only for summary planning."""

    description: str


class VisualDescription(Description):
    """Visual enrichment kept separate from original source text."""

    has_visual_content: bool


class SelfSufficiencyJudgment(StrictModel):
    """Reference self-sufficiency judgment with supplied source context."""

    self_sufficiency: int = Field(ge=1, le=5)
    reasoning: str


class QueryMetadata(StrictModel):
    """Query-only answer-leak and observed-label assessment."""

    actual_query_type: Literal[
        "open-ended", "compare-contrast", "enumerative", "numerical", "boolean", "extractive", "multi-hop"
    ]
    actual_query_format: Literal["question", "instruction", "keyword"]
    has_answer: bool
    classification_reasoning: str = ""
    classification_confidence: float | None = Field(default=None, ge=0, le=1)


class QuerySlot(StrictModel):
    """One generated query or explicit abstention."""

    # Data Designer validates JSON Schema before Python models are reconstructed.
    # Encode the relation here too, so native correction receives the actual error.
    model_config = ConfigDict(
        json_schema_extra={
            "if": {"properties": {"slot": {}, "query": {}, "evidence_modality": {"const": "none"}}},
            "then": {"properties": {"slot": {}, "evidence_modality": {}, "query": {"type": "null"}}},
            "else": {
                "properties": {
                    "slot": {},
                    "evidence_modality": {},
                    "query": {"type": "string", "minLength": 1, "pattern": "\\S"},
                }
            },
        }
    )

    slot: int = Field(ge=0)
    query: str | None = Field(description="A nonempty query, or null for an unsupported slot; never an empty string")
    evidence_modality: Literal["text", "image", "text_and_image", "none"] = Field(
        description="Use none exactly when query is null; otherwise identify the actual supporting evidence"
    )

    @model_validator(mode="after")
    def consistent_abstention(self) -> QuerySlot:
        """Reject inconsistent slots so bounded structured-response retries can regenerate them."""
        if self.query is None and self.evidence_modality != "none":
            raise ValueError("Abstention must declare evidence_modality=none")
        if self.query is not None and (not self.query or self.evidence_modality == "none"):
            raise ValueError("Generated query must be nonempty and declare evidence")
        return self


class QueryBatch(StrictModel):
    """Addressable outcomes validated against the requested slots by the runner."""

    queries: list[QuerySlot]


class QueryJudgment(StrictModel):
    """Stable combined output of context-aware sufficiency and query-only metadata checks."""

    self_sufficiency: int = Field(ge=1, le=5)
    has_answer: bool
    observed_type: str = Field(min_length=1)
    observed_format: Literal["question", "instruction", "keyword"]
    reasoning: str


class RelevanceJudgment(StrictModel):
    """Source-reading relevance check, without an invented generated answer."""

    relevance: int = Field(ge=1, le=5)
    reasoning: str


class Support(StrictModel):
    """Source-local silver relevance; a grade is not an embedding similarity score."""

    unit_id: str = Field(min_length=1)
    grade: Literal[1, 2]
    modality: Literal["text", "image", "text_and_image"]
    quote: str
    visual_evidence: str
    contribution: str = Field(min_length=1)


class Localization(StrictModel):
    """All relevant units within the supplied context, not exhaustive corpus judgments."""

    supports: list[Support]


class Candidate(StrictModel):
    """Auditable terminal candidate, including rejected queries and their actual judgments."""

    query_id: str
    context_id: str
    query: str
    source_unit_ids: list[str]
    language: str
    requested_instruction: QueryInstruction
    query_judgment: QueryJudgment
    relevance_judgment: RelevanceJudgment
    localization: Localization | None
    accepted: bool
    rejection_reasons: list[str]
    evidence_modality: Literal["text", "image", "text_and_image", "none"] = "none"
    quote_verification: dict[str, bool] = Field(default_factory=dict)
    provenance: QueryProvenance = Field(default_factory=QueryProvenance)
