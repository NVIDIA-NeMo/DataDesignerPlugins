# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import data_designer.config as dd
import pytest
import yaml
from data_designer.engine.storage.artifact_storage import ResumeMode

from data_designer_retrieval_sdg.run_artifacts import (
    finalize_generation_run_artifacts,
    write_conversion_run_artifacts,
    write_generation_run_artifacts,
)
from data_designer_retrieval_sdg.run_config import (
    ConfigSource,
    ConversionRunConfig,
    GenerationPipelineConfig,
    GenerationRunConfig,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


def _generation_config(tmp_path: Path, **updates: object) -> GenerationRunConfig:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    source = docs / "source.txt"
    if not source.exists():
        source.write_text("The initial source text.", encoding="utf-8")
    values: dict[str, object] = {
        "seed_source": DocumentChunkerSeedSource(path=str(docs), file_extensions=[".txt"]),
        "output_dir": tmp_path / "generated",
        "artifact_path": tmp_path / "artifacts",
        "dataset_name": "retrieval",
        "num_records": 1,
    }
    values.update(updates)
    return GenerationRunConfig(**values)


def test_generation_run_artifacts_are_complete_and_redacted(tmp_path: Path) -> None:
    config = _generation_config(
        tmp_path,
        model_providers=[
            dd.ModelProvider(
                name="custom",
                endpoint="https://example.invalid/v1",
                provider_type="openai",
                api_key="do-not-persist",
            )
        ],
    )
    source = ConfigSource(location="/configs/generation.yaml", sha256="a" * 64)

    artifacts = write_generation_run_artifacts(
        config,
        dataset_name="retrieval",
        num_records=1,
        producer_version="0.1.0",
        sources=[source],
        override_paths=["pipeline.num_pairs"],
        environment_variables=["NVIDIA_API_KEY"],
    )

    resolved_text = artifacts.resolved_config_path.read_text(encoding="utf-8")
    resolved = yaml.safe_load(resolved_text)
    provenance = json.loads(artifacts.provenance_path.read_text(encoding="utf-8"))
    assert "do-not-persist" not in resolved_text
    assert resolved["model_providers"][0]["api_key"] == "<redacted>"
    assert resolved["dataset_name"] == "retrieval"
    assert resolved["num_records"] == 1
    assert provenance["producer_version"] == "0.1.0"
    assert provenance["config_sources"] == [{"location": "/configs/generation.yaml", "sha256": "a" * 64}]
    assert provenance["override_paths"] == ["pipeline.num_pairs"]
    assert provenance["environment_variables"] == ["NVIDIA_API_KEY"]
    assert provenance["input_file_count"] == 1
    assert provenance["config_fingerprint"] == artifacts.config_fingerprint
    assert provenance["input_fingerprint"] == artifacts.input_fingerprint


def test_resume_refuses_changed_generation_settings(tmp_path: Path) -> None:
    initial = _generation_config(tmp_path, resume=ResumeMode.IF_POSSIBLE)
    write_generation_run_artifacts(
        initial,
        dataset_name="retrieval",
        num_records=1,
        producer_version="0.1.0",
    )
    changed = initial.model_copy(update={"pipeline": GenerationPipelineConfig(min_complexity=3)})

    with pytest.raises(ValueError, match="resolved generation settings changed"):
        write_generation_run_artifacts(
            changed,
            dataset_name="retrieval",
            num_records=1,
            producer_version="0.1.0",
        )


def test_resume_refuses_changed_source_corpus(tmp_path: Path) -> None:
    config = _generation_config(tmp_path, resume=ResumeMode.IF_POSSIBLE)
    write_generation_run_artifacts(
        config,
        dataset_name="retrieval",
        num_records=1,
        producer_version="0.1.0",
    )
    (tmp_path / "docs" / "source.txt").write_text("Changed source text.", encoding="utf-8")

    with pytest.raises(ValueError, match="source corpus changed"):
        write_generation_run_artifacts(
            config,
            dataset_name="retrieval",
            num_records=1,
            producer_version="0.1.0",
        )


def test_resume_refuses_changed_plugin_version(tmp_path: Path) -> None:
    config = _generation_config(tmp_path, resume=ResumeMode.IF_POSSIBLE)
    write_generation_run_artifacts(
        config,
        dataset_name="retrieval",
        num_records=1,
        producer_version="0.1.0",
    )

    with pytest.raises(ValueError, match="plugin version changed"):
        write_generation_run_artifacts(
            config,
            dataset_name="retrieval",
            num_records=1,
            producer_version="0.2.0",
        )


def test_resume_refuses_legacy_artifacts_without_provenance(tmp_path: Path) -> None:
    config = _generation_config(tmp_path, resume=ResumeMode.ALWAYS)
    (config.artifact_path / "retrieval").mkdir(parents=True)

    with pytest.raises(ValueError, match="existing artifacts have no config_provenance.json"):
        write_generation_run_artifacts(
            config,
            dataset_name="retrieval",
            num_records=1,
            producer_version="0.1.0",
        )


def test_operational_generation_settings_do_not_change_data_fingerprint(tmp_path: Path) -> None:
    first = _generation_config(tmp_path, dataset_name="first", buffer_size=10, log_level="INFO")
    second = first.model_copy(
        update={
            "output_dir": tmp_path / "another-output",
            "artifact_path": tmp_path / "another-artifacts",
            "dataset_name": "second",
            "buffer_size": 50,
            "log_level": "DEBUG",
        }
    )

    first_artifacts = write_generation_run_artifacts(
        first,
        dataset_name="first",
        num_records=1,
        producer_version="0.1.0",
    )
    second_artifacts = write_generation_run_artifacts(
        second,
        dataset_name="second",
        num_records=1,
        producer_version="0.1.0",
    )

    assert first_artifacts.config_fingerprint == second_artifacts.config_fingerprint
    assert first_artifacts.input_fingerprint == second_artifacts.input_fingerprint


def test_fresh_name_collision_preserves_prior_metadata_and_uses_resolved_name(tmp_path: Path) -> None:
    config = _generation_config(tmp_path, resume=ResumeMode.NEVER)
    first = write_generation_run_artifacts(
        config,
        dataset_name="retrieval",
        num_records=1,
        producer_version="0.1.0",
    )
    first_provenance = first.provenance_path.read_text(encoding="utf-8")
    (config.artifact_path / "retrieval").mkdir(parents=True)

    staged = write_generation_run_artifacts(
        config,
        dataset_name="retrieval",
        num_records=1,
        producer_version="0.1.0",
    )
    assert staged.provenance_path.parent.parent.name == ".pending"
    assert first.provenance_path.read_text(encoding="utf-8") == first_provenance

    finalized = finalize_generation_run_artifacts(
        staged,
        config,
        resolved_dataset_name="retrieval-1",
    )

    assert finalized.provenance_path.parent.name == "retrieval-1"
    assert first.provenance_path.exists()
    provenance = json.loads(finalized.provenance_path.read_text(encoding="utf-8"))
    assert provenance["dataset_name"] == "retrieval"
    assert provenance["resolved_dataset_name"] == "retrieval-1"


def test_conversion_run_artifacts_capture_exact_input(tmp_path: Path) -> None:
    input_path = tmp_path / "records.jsonl"
    input_path.write_text('{"record": 1}\n', encoding="utf-8")
    config = ConversionRunConfig(
        input_path=input_path,
        corpus_id="corpus",
        output_dir=tmp_path / "converted",
    )

    artifacts = write_conversion_run_artifacts(
        config,
        output_dir=tmp_path / "converted",
        producer_version="0.1.0",
        override_paths=["corpus_id"],
    )

    resolved = yaml.safe_load(artifacts.resolved_config_path.read_text(encoding="utf-8"))
    provenance = json.loads(artifacts.provenance_path.read_text(encoding="utf-8"))
    assert resolved["input_path"] == str(input_path)
    assert resolved["output_dir"] == str(tmp_path / "converted")
    assert provenance["override_paths"] == ["corpus_id"]
    assert provenance["input_file_count"] == 1
    assert artifacts.resolved_config_path.parent.name == ".retrieval_sdg_run"


def test_conversion_input_fingerprint_includes_group_file_contents(tmp_path: Path) -> None:
    input_path = tmp_path / "records.jsonl"
    input_path.write_text('{"record": 1}\n', encoding="utf-8")
    groups_path = tmp_path / "groups.json"
    groups_path.write_text('{"groups": []}\n', encoding="utf-8")
    config = ConversionRunConfig(
        input_path=input_path,
        corpus_id="corpus",
        output_dir=tmp_path / "converted",
        groups_json=[groups_path],
    )
    first = write_conversion_run_artifacts(
        config,
        output_dir=tmp_path / "converted",
        producer_version="0.1.0",
    )
    groups_path.write_text('{"groups": [["doc"]]}\n', encoding="utf-8")
    second = write_conversion_run_artifacts(
        config,
        output_dir=tmp_path / "converted",
        producer_version="0.1.0",
    )

    assert first.input_fingerprint != second.input_fingerprint
