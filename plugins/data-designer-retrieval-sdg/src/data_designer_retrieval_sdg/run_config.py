# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for retrieval SDG generation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import data_designer.config as dd
from data_designer.config.base import ConfigBase
from data_designer.engine.storage.artifact_storage import ResumeMode
from pydantic import ConfigDict, Field, model_validator

from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

DEFAULT_CHAT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
DEFAULT_PROVIDER = "nvidia"
DEFAULT_ARTIFACT_PATH = Path("./artifacts")
DEFAULT_BUFFER_SIZE = 200
DEFAULT_MAX_ARTIFACTS_PER_TYPE = 2
DEFAULT_NUM_PAIRS = 7
DEFAULT_MIN_HOPS = 1
DEFAULT_MAX_HOPS = 3
DEFAULT_MIN_COMPLEXITY = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.9
DEFAULT_QUERY_COUNTS: dict[str, int] = {"multi_hop": 3, "structural": 2, "contextual": 2}
DEFAULT_REASONING_COUNTS: dict[str, int] = {
    "factual": 1,
    "relational": 1,
    "inferential": 1,
    "temporal": 1,
    "procedural": 1,
    "causal": 1,
    "visual": 1,
}

NonNegativeInt = Annotated[int, Field(ge=0)]
_QUERY_COUNT_KEYS = frozenset(DEFAULT_QUERY_COUNTS)
_REASONING_COUNT_KEYS = frozenset(DEFAULT_REASONING_COUNTS)


def _validate_count_distribution(
    name: str,
    counts: dict[str, int],
    expected_keys: frozenset[str],
    num_pairs: int,
) -> None:
    """Validate one complete question-distribution mapping."""
    actual_keys = set(counts)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(f"{name} keys must match the expected set; missing={missing}, unexpected={unexpected}")

    total = sum(counts.values())
    if total != num_pairs:
        raise ValueError(f"{name} must sum to num_pairs ({num_pairs}); got {total}")


def _is_sensitive_key(key: str) -> bool:
    """Return whether a serialized configuration key can contain a credential."""
    normalized = key.lower().replace("-", "_")
    if normalized.endswith("_env"):
        return False
    exact_sensitive_keys = {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "password",
        "proxy_authorization",
        "refresh_token",
        "secret",
        "token",
    }
    sensitive_suffixes = (
        "_access_token",
        "_api_key",
        "_password",
        "_refresh_token",
        "_secret",
    )
    return normalized in exact_sensitive_keys or normalized.endswith(sensitive_suffixes)


def _redact_sensitive_values(value: Any) -> Any:
    """Return a recursively redacted copy of serialized configuration data."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, nested_value in value.items():
            if _is_sensitive_key(str(key)) and nested_value is not None:
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = _redact_sensitive_values(nested_value)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    return value


class GenerationPipelineConfig(ConfigBase):
    """Explicit settings for the four-column retrieval generation pipeline."""

    model_config = ConfigDict(frozen=True)

    max_artifacts_per_type: int = Field(default=DEFAULT_MAX_ARTIFACTS_PER_TYPE, ge=1)
    num_pairs: int = Field(default=DEFAULT_NUM_PAIRS, ge=1)
    query_counts: dict[str, NonNegativeInt] = Field(default_factory=lambda: dict(DEFAULT_QUERY_COUNTS))
    min_hops: int = Field(default=DEFAULT_MIN_HOPS, ge=1)
    max_hops: int = Field(default=DEFAULT_MAX_HOPS, ge=1)
    reasoning_counts: dict[str, NonNegativeInt] = Field(default_factory=lambda: dict(DEFAULT_REASONING_COUNTS))
    min_complexity: int = Field(default=DEFAULT_MIN_COMPLEXITY, ge=1, le=5)
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0)
    max_parallel_requests_for_gen: int | None = Field(default=None, ge=1)
    artifact_extraction_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1)
    artifact_extraction_provider: str = Field(default=DEFAULT_PROVIDER, min_length=1)
    qa_generation_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1)
    qa_generation_provider: str = Field(default=DEFAULT_PROVIDER, min_length=1)
    quality_judge_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1)
    quality_judge_provider: str = Field(default=DEFAULT_PROVIDER, min_length=1)
    embed_model: str = Field(default=DEFAULT_EMBED_MODEL, min_length=1)
    embed_provider: str = Field(default=DEFAULT_PROVIDER, min_length=1)

    @model_validator(mode="after")
    def validate_pipeline_settings(self) -> GenerationPipelineConfig:
        """Reject inconsistent hop and question-distribution settings."""
        if self.min_hops > self.max_hops:
            raise ValueError("min_hops must be less than or equal to max_hops")
        _validate_count_distribution("query_counts", self.query_counts, _QUERY_COUNT_KEYS, self.num_pairs)
        _validate_count_distribution("reasoning_counts", self.reasoning_counts, _REASONING_COUNT_KEYS, self.num_pairs)
        return self

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``build_qa_generation_pipeline``."""
        return self.model_dump(mode="python")


class GenerationRunConfig(ConfigBase):
    """Complete typed input for one resumable retrieval generation run."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    seed_source: DocumentChunkerSeedSource
    output_dir: Path
    artifact_path: Path = DEFAULT_ARTIFACT_PATH
    dataset_name: str | None = None
    buffer_size: int = Field(default=DEFAULT_BUFFER_SIZE, ge=1)
    resume: ResumeMode = Field(default=ResumeMode.NEVER, validate_default=True)
    model_providers: list[dd.ModelProvider] | None = None
    pipeline: GenerationPipelineConfig = Field(default_factory=GenerationPipelineConfig)
    num_records: int | None = Field(default=None, ge=1)

    def to_redacted_dict(self) -> dict[str, Any]:
        """Serialize all effective settings without exposing credential values."""
        serialized = self.model_dump(mode="json")
        return _redact_sensitive_values(serialized)
