# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import data_designer.config as dd
import pytest
from pydantic import ValidationError

from data_designer_retrieval_sdg import (
    GenerationPipelineConfig as PublicGenerationPipelineConfig,
)
from data_designer_retrieval_sdg import (
    GenerationRunConfig as PublicGenerationRunConfig,
)
from data_designer_retrieval_sdg.run_config import GenerationPipelineConfig, GenerationRunConfig
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


def test_generation_config_models_are_exported_from_package_root() -> None:
    assert PublicGenerationPipelineConfig is GenerationPipelineConfig
    assert PublicGenerationRunConfig is GenerationRunConfig


def test_generation_pipeline_config_has_canonical_defaults() -> None:
    config = GenerationPipelineConfig()

    assert config.num_pairs == 7
    assert config.min_hops == 1
    assert config.max_hops == 3
    assert config.min_complexity == 2
    assert config.artifact_extraction_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert config.qa_generation_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert config.quality_judge_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert config.embed_model == "nvidia/nemotron-3-embed-1b"


def test_generation_configs_reject_unknown_fields_and_schema_versions(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GenerationPipelineConfig.model_validate({"num_pair": 10})

    with pytest.raises(ValidationError, match="literal_error"):
        GenerationRunConfig(
            schema_version=2,
            seed_source=DocumentChunkerSeedSource(path=str(tmp_path)),
            output_dir=tmp_path / "output",
        )


def test_generation_pipeline_config_rejects_invalid_hop_range() -> None:
    with pytest.raises(ValidationError, match="min_hops must be less than or equal to max_hops"):
        GenerationPipelineConfig(min_hops=4, max_hops=3)


@pytest.mark.parametrize("min_complexity", [0, 6])
def test_generation_pipeline_config_rejects_complexity_outside_prompt_scale(min_complexity: int) -> None:
    with pytest.raises(ValidationError):
        GenerationPipelineConfig(min_complexity=min_complexity)


@pytest.mark.parametrize(
    ("field", "counts"),
    [
        ("query_counts", {"multi-hop": 3, "structural": 2, "contextual": 2}),
        (
            "reasoning_counts",
            {
                "factual": 1,
                "relational": 1,
                "inferential": 1,
                "temporal": 1,
                "procedural": 1,
                "causal": 1,
                "spatial": 1,
            },
        ),
    ],
)
def test_generation_pipeline_config_rejects_unknown_distribution_keys(
    field: str,
    counts: dict[str, int],
) -> None:
    with pytest.raises(ValidationError, match=f"{field} keys must match the expected set"):
        GenerationPipelineConfig.model_validate({field: counts})


@pytest.mark.parametrize(
    ("field", "counts"),
    [
        ("query_counts", {"multi_hop": 2, "structural": 2, "contextual": 2}),
        (
            "reasoning_counts",
            {
                "factual": 2,
                "relational": 1,
                "inferential": 1,
                "temporal": 1,
                "procedural": 1,
                "causal": 1,
                "visual": 1,
            },
        ),
    ],
)
def test_generation_pipeline_config_requires_distributions_to_sum_to_num_pairs(
    field: str,
    counts: dict[str, int],
) -> None:
    with pytest.raises(ValidationError, match=f"{field} must sum to num_pairs"):
        GenerationPipelineConfig.model_validate({field: counts})


def test_generation_run_config_serialization_redacts_credentials(tmp_path: Path) -> None:
    config = GenerationRunConfig(
        seed_source=DocumentChunkerSeedSource(path=str(tmp_path)),
        output_dir=tmp_path / "output",
        model_providers=[
            dd.ModelProvider(
                name="custom",
                endpoint="https://example.invalid/v1",
                api_key="provider-secret",
                extra_headers={
                    "Authorization": "Bearer header-secret",
                    "X-Request-Id": "visible-request-id",
                },
                extra_body={
                    "access_token": "body-secret",
                    "api_key_env": "CUSTOM_API_KEY",
                    "max_tokens": 1024,
                },
            )
        ],
    )

    redacted = config.to_redacted_dict()
    provider = redacted["model_providers"][0]
    serialized = json.dumps(redacted)

    assert provider["api_key"] == "<redacted>"
    assert provider["extra_headers"]["Authorization"] == "<redacted>"
    assert provider["extra_headers"]["X-Request-Id"] == "visible-request-id"
    assert provider["extra_body"]["access_token"] == "<redacted>"
    assert provider["extra_body"]["api_key_env"] == "CUSTOM_API_KEY"
    assert provider["extra_body"]["max_tokens"] == 1024
    assert "provider-secret" not in serialized
    assert "header-secret" not in serialized
    assert "body-secret" not in serialized
