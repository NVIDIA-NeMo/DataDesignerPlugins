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
    context_strategy: Literal["unit", "document"] = "unit"
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
        return self


class ContextSummary(StrictModel):
    """Context enrichment for query generation, never a replacement for source evidence."""

    summary: str
    visual_evidence: str


class SummaryJudgment(StrictModel):
    """Source-grounded summary fidelity and usefulness, independent of query labels."""

    fidelity: int = Field(ge=1, le=5)
    usefulness: int = Field(ge=1, le=5)
    reasoning: str


class QuerySlot(StrictModel):
    """One generated query or explicit abstention."""

    slot: int = Field(ge=0)
    query: str | None
    evidence_modality: Literal["text", "image", "text_and_image", "none"]


class QueryBatch(StrictModel):
    """Addressable outcomes validated against the requested slots by the runner."""

    queries: list[QuerySlot]


class QueryJudgment(StrictModel):
    """Query-only standalone/answer-leak checks and observed, non-gating style labels."""

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
