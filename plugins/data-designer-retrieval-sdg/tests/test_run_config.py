# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import data_designer.config as dd
import pytest
import yaml
from pydantic import ValidationError
from pydantic_settings import CliSettingsSource

from data_designer_retrieval_sdg import (
    ConversionRunConfig as PublicConversionRunConfig,
)
from data_designer_retrieval_sdg import (
    GenerationPipelineConfig as PublicGenerationPipelineConfig,
)
from data_designer_retrieval_sdg import (
    GenerationRunConfig as PublicGenerationRunConfig,
)
from data_designer_retrieval_sdg.run_config import (
    ConversionRunConfig,
    GenerationPipelineConfig,
    GenerationRunConfig,
    dump_resolved_config,
    load_conversion_config,
    load_generation_config,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


def test_generation_config_models_are_exported_from_package_root() -> None:
    assert PublicGenerationPipelineConfig is GenerationPipelineConfig
    assert PublicGenerationRunConfig is GenerationRunConfig
    assert PublicConversionRunConfig is ConversionRunConfig


def test_generation_pipeline_config_has_canonical_defaults() -> None:
    config = GenerationPipelineConfig()
    chat_model = "nvidia/nemotron-3-ultra-550b-a55b"
    embed_model = "nvidia/nemotron-3-embed-1b"

    assert config.num_pairs == 7
    assert config.min_hops == 1
    assert config.max_hops == 3
    assert config.min_complexity == 2
    assert config.artifact_extraction_model == chat_model
    assert config.qa_generation_model == chat_model
    assert config.quality_judge_model == chat_model
    assert config.embed_model == embed_model


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
                    "X-Request-Id": "request-secret",
                    "X-Auth-Token": "custom-header-secret",
                    "Cookie": "session=cookie-secret",
                },
                extra_body={
                    "input_type": "query",
                    "truncate": "NONE",
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
    assert provider["extra_headers"]["X-Request-Id"] == "<redacted>"
    assert provider["extra_headers"]["X-Auth-Token"] == "<redacted>"
    assert provider["extra_headers"]["Cookie"] == "<redacted>"
    assert provider["extra_body"]["input_type"] == "query"
    assert provider["extra_body"]["truncate"] == "NONE"
    assert provider["extra_body"]["access_token"] == "<redacted>"
    assert provider["extra_body"]["api_key_env"] == "<redacted>"
    assert provider["extra_body"]["max_tokens"] == "<redacted>"
    assert "provider-secret" not in serialized
    assert "header-secret" not in serialized
    assert "request-secret" not in serialized
    assert "custom-header-secret" not in serialized
    assert "cookie-secret" not in serialized
    assert "body-secret" not in serialized


def test_extra_body_allowlist_requires_expected_value_types(tmp_path: Path) -> None:
    config = GenerationRunConfig(
        seed_source=DocumentChunkerSeedSource(path=str(tmp_path)),
        output_dir=tmp_path / "output",
        model_providers=[
            dd.ModelProvider(
                name="custom",
                endpoint="https://example.invalid/v1",
                provider_type="openai",
                extra_body={"input_type": {"credential": "nested-secret"}},
            )
        ],
    )

    provider = config.to_redacted_dict()["model_providers"][0]

    assert provider["extra_body"]["input_type"] == "<redacted>"
    assert "nested-secret" not in json.dumps(provider)


def test_pydantic_defaults_are_complete_and_validate() -> None:
    generation = load_generation_config()
    conversion = load_conversion_config()

    assert generation.config.seed_source.path == "."
    assert generation.config.seed_source.file_extensions == [".txt", ".md", ".text"]
    assert generation.config.output_dir == Path("generated")
    assert generation.config.pipeline.num_pairs == 7
    assert generation.config.pipeline.embed_model == "nvidia/nemotron-3-embed-1b"
    assert generation.sources == ()
    assert generation.override_paths == ()

    assert conversion.config.input_path == Path("generated/retrieval_sdg.jsonl")
    assert conversion.config.corpus_id == "retrieval_sdg"
    assert conversion.config.split_strategy == "random"
    assert conversion.sources == ()


def test_generation_config_precedence_is_model_then_file_then_programmatic_then_cli(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    config_path = tmp_path / "generation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed_source": {"path": str(tmp_path / "docs")},
                "pipeline": {"min_complexity": 1, "min_hops": 2},
            }
        ),
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    cli_settings_source = CliSettingsSource(
        GenerationRunConfig,
        root_parser=parser,
        cli_kebab_case=True,
    )
    cli_args = parser.parse_args(["--pipeline.min-complexity", "3", "--buffer-size", "32"])
    loaded = load_generation_config(
        config_path,
        cli_overrides={"pipeline": {"min_complexity": 2}, "seed_source": {"multi_doc": True}},
        cli_args=cli_args,
        cli_settings_source=cli_settings_source,
    )

    assert loaded.config.pipeline.min_complexity == 3
    assert loaded.config.pipeline.min_hops == 2
    assert loaded.config.buffer_size == 32
    assert loaded.config.seed_source.multi_doc is True
    assert [source.location for source in loaded.sources] == [str(config_path.resolve())]
    assert loaded.override_paths == ("pipeline.min_complexity", "seed_source.multi_doc", "buffer_size")


def test_file_loader_rejects_unknown_fields_in_files_and_programmatic_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("pipeline:\n  num_pair: 10\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_generation_config(config_path)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_generation_config(cli_overrides={"pipeline": {"unknown": 1}})


def test_explicit_provider_environment_references_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEST_PROVIDER_ENDPOINT", "https://example.invalid/v1")
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "provider-secret")
    config_path = tmp_path / "provider.yaml"
    config_path.write_text(
        """
model_providers:
  - name: custom
    endpoint: ${TEST_PROVIDER_ENDPOINT}
    provider_type: openai
    api_key: ${TEST_PROVIDER_API_KEY}
""".lstrip(),
        encoding="utf-8",
    )

    loaded = load_generation_config(config_path)

    assert loaded.config.model_providers is not None
    assert loaded.config.model_providers[0].endpoint == "https://example.invalid/v1"
    assert loaded.config.model_providers[0].api_key == "TEST_PROVIDER_API_KEY"
    assert loaded.environment_variables == ("TEST_PROVIDER_ENDPOINT", "TEST_PROVIDER_API_KEY")
    resolved = dump_resolved_config(loaded.config)
    assert "provider-secret" not in resolved
    assert "api_key: <redacted>" in resolved


def test_unset_explicit_provider_environment_reference_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "provider.yaml"
    config_path.write_text(
        """
model_providers:
  - name: custom
    endpoint: ${MISSING_PROVIDER_ENDPOINT}
    provider_type: openai
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MISSING_PROVIDER_ENDPOINT"):
        load_generation_config(config_path)


def test_conversion_run_config_validates_split_ratios() -> None:
    with pytest.raises(ValidationError, match="train_ratio plus val_ratio"):
        ConversionRunConfig(input_path="input.jsonl", corpus_id="corpus", train_ratio=0.9, val_ratio=0.2)


def test_conversion_config_to_kwargs_normalizes_paths(tmp_path: Path) -> None:
    config = ConversionRunConfig(
        input_path=tmp_path / "input.jsonl",
        corpus_id="corpus",
        output_dir=tmp_path / "output",
        groups_json=[tmp_path / "groups.json"],
    )

    kwargs = config.to_conversion_kwargs()

    assert kwargs["input_path"] == str(tmp_path / "input.jsonl")
    assert kwargs["output_dir"] == str(tmp_path / "output")
    assert kwargs["groups_json"] == [str(tmp_path / "groups.json")]
