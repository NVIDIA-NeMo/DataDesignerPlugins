# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for retrieval pipeline defaults and provider construction."""

import inspect
import json
from pathlib import Path

import pytest

from data_designer_retrieval_sdg.pipeline import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    build_model_providers,
    build_qa_generation_pipeline,
)
from data_designer_retrieval_sdg.run_config import GenerationPipelineConfig


def test_defaults_match_nemotron_profile() -> None:
    assert DEFAULT_CHAT_MODEL == "nvidia/nemotron-3-ultra-550b-a55b"
    assert DEFAULT_EMBED_MODEL == "nvidia/nemotron-3-embed-1b"


def test_pipeline_builder_defaults_match_typed_config() -> None:
    config = GenerationPipelineConfig()
    parameters = inspect.signature(build_qa_generation_pipeline).parameters

    for field_name in (
        "max_artifacts_per_type",
        "num_pairs",
        "min_hops",
        "max_hops",
        "min_complexity",
        "similarity_threshold",
        "max_parallel_requests_for_gen",
        "artifact_extraction_model",
        "artifact_extraction_provider",
        "qa_generation_model",
        "qa_generation_provider",
        "quality_judge_model",
        "quality_judge_provider",
        "embed_model",
        "embed_provider",
    ):
        assert parameters[field_name].default == getattr(config, field_name)


def test_provider_builder_combines_distinct_aliases(tmp_path: Path) -> None:
    providers_file = tmp_path / "providers.json"
    providers_file.write_text(
        json.dumps(
            [
                {
                    "name": "local",
                    "endpoint": "http://localhost:8000/v1",
                    "provider_type": "openai",
                }
            ]
        ),
        encoding="utf-8",
    )

    all_providers, custom = build_model_providers(
        custom_provider_endpoint="https://gateway.example.invalid/v1",
        custom_provider_name="nvidia",
        custom_provider_api_key="NVIDIA_API_KEY",
        model_providers_file=providers_file,
    )

    assert all_providers is not None
    assert {provider.name for provider in custom} == {"local", "nvidia"}
    assert next(provider for provider in custom if provider.name == "nvidia").endpoint == (
        "https://gateway.example.invalid/v1"
    )


def test_provider_builder_rejects_conflicting_aliases(tmp_path: Path) -> None:
    providers_file = tmp_path / "providers.json"
    providers_file.write_text(
        json.dumps(
            [
                {
                    "name": "nvidia",
                    "endpoint": "https://first.example.invalid/v1",
                    "provider_type": "openai",
                    "api_key": "NVIDIA_API_KEY",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting model provider alias 'nvidia'"):
        build_model_providers(
            custom_provider_endpoint="https://second.example.invalid/v1",
            custom_provider_name="nvidia",
            custom_provider_api_key="NVIDIA_API_KEY",
            model_providers_file=providers_file,
        )
