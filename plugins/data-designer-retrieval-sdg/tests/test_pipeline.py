# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for retrieval pipeline defaults and provider construction."""

import inspect
import json
from pathlib import Path

import data_designer.config as dd
import pytest

import data_designer_retrieval_sdg.pipeline as pipeline_module
from data_designer_retrieval_sdg.pipeline import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    build_model_providers,
    build_qa_generation_pipeline,
)
from data_designer_retrieval_sdg.run_config import GenerationPipelineConfig


def test_defaults_match_canonical_nemotron_models() -> None:
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


def test_provider_builder_uses_builtins_before_default_file_is_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    builtin = dd.ModelProvider(
        name="builtin",
        endpoint="https://builtin.example.invalid/v1",
        provider_type="openai",
    )

    def missing_default_file() -> list[dd.ModelProvider]:
        raise FileNotFoundError

    monkeypatch.setattr(pipeline_module, "get_default_providers", missing_default_file)
    monkeypatch.setattr(pipeline_module, "get_builtin_model_providers", lambda: [builtin])

    all_providers, custom = build_model_providers(
        custom_provider_endpoint="https://custom.example.invalid/v1",
        custom_provider_name="custom",
    )

    assert all_providers is not None
    assert [provider.name for provider in all_providers] == ["builtin", "custom"]
    assert [provider.name for provider in custom] == ["custom"]


def test_inline_provider_overrides_matching_provider_file_alias(tmp_path: Path) -> None:
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

    _, custom = build_model_providers(
        custom_provider_endpoint="https://second.example.invalid/v1",
        custom_provider_name="nvidia",
        custom_provider_api_key="NVIDIA_API_KEY",
        model_providers_file=providers_file,
    )

    provider = next(item for item in custom if item.name == "nvidia")
    assert provider.endpoint == "https://second.example.invalid/v1"


def test_partial_inline_provider_override_preserves_unspecified_fields() -> None:
    configured = dd.ModelProvider(
        name="custom",
        endpoint="https://old.example.invalid/v1",
        provider_type="openai",
        api_key="CUSTOM_API_KEY",
    )

    _, custom = build_model_providers(
        model_providers=[configured],
        custom_provider_endpoint="https://new.example.invalid/v1",
        custom_provider_name="custom",
        custom_provider_fields={"endpoint", "name"},
    )

    provider = next(item for item in custom if item.name == "custom")
    assert provider.endpoint == "https://new.example.invalid/v1"
    assert provider.provider_type == "openai"
    assert provider.api_key == "CUSTOM_API_KEY"


def test_provider_builder_rejects_conflicting_aliases_within_one_file(tmp_path: Path) -> None:
    providers_file = tmp_path / "providers.json"
    providers_file.write_text(
        json.dumps(
            [
                {"name": "local", "endpoint": "https://first.example.invalid/v1"},
                {"name": "local", "endpoint": "https://second.example.invalid/v1"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting model provider alias 'local' within"):
        build_model_providers(model_providers_file=providers_file)
