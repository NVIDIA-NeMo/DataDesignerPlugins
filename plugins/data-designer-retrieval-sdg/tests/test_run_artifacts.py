# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import data_designer.config as dd
import yaml

from data_designer_retrieval_sdg.convert import ConversionResult
from data_designer_retrieval_sdg.run_artifacts import (
    write_conversion_run_artifacts,
    write_generation_run_artifacts,
)
from data_designer_retrieval_sdg.run_config import (
    ConfigSource,
    ConversionRunConfig,
    GenerationRunConfig,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


def _generation_config(tmp_path: Path) -> GenerationRunConfig:
    return GenerationRunConfig(
        seed_source=DocumentChunkerSeedSource(path=str(tmp_path / "docs"), file_extensions=[".txt"]),
        output_dir=tmp_path / "generated",
        artifact_path=tmp_path / "artifacts",
        dataset_name="retrieval",
        num_records=1,
        model_providers=[
            dd.ModelProvider(
                name="custom",
                endpoint="https://example.invalid/v1",
                provider_type="openai",
                api_key="do-not-persist",
            )
        ],
    )


def test_generation_run_artifacts_capture_completed_outputs_and_redact_secrets(tmp_path: Path) -> None:
    config = _generation_config(tmp_path)
    source = ConfigSource(location="/configs/generation.yaml", sha256="a" * 64)
    dataset_path = tmp_path / "artifacts" / "retrieval_resolved"
    output_path = tmp_path / "generated" / "retrieval_resolved.jsonl"

    artifacts = write_generation_run_artifacts(
        config,
        requested_dataset_name="retrieval",
        resolved_dataset_name="retrieval_resolved",
        dataset_path=dataset_path,
        output_path=output_path,
        requested_num_records=3,
        actual_num_records=2,
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
    assert resolved["dataset_name"] == "retrieval_resolved"
    assert resolved["num_records"] == 3
    assert artifacts.resolved_config_path.parent.name == "retrieval_resolved"
    assert provenance == {
        "schema_version": 1,
        "operation": "generation",
        "producer_version": "0.1.0",
        "requested_dataset_name": "retrieval",
        "resolved_dataset_name": "retrieval_resolved",
        "resume_mode": "never",
        "config_sources": [{"location": "/configs/generation.yaml", "sha256": "a" * 64}],
        "override_paths": ["pipeline.num_pairs"],
        "environment_variables": ["NVIDIA_API_KEY"],
        "output_paths": {
            "dataset": str(dataset_path.resolve()),
            "jsonl": str(output_path.resolve()),
        },
        "record_counts": {"requested": 3, "generated": 2},
    }


def test_generation_run_artifacts_do_not_read_or_hash_seed_files(tmp_path: Path) -> None:
    config = _generation_config(tmp_path)

    artifacts = write_generation_run_artifacts(
        config,
        requested_dataset_name="retrieval",
        resolved_dataset_name="retrieval",
        dataset_path=tmp_path / "artifacts" / "retrieval",
        output_path=tmp_path / "generated" / "retrieval.jsonl",
        requested_num_records=1,
        actual_num_records=1,
        producer_version="0.1.0",
    )

    provenance = json.loads(artifacts.provenance_path.read_text(encoding="utf-8"))
    assert "input_fingerprint" not in provenance
    assert "config_fingerprint" not in provenance


def test_generation_run_artifacts_use_each_resolved_dataset_name(tmp_path: Path) -> None:
    config = _generation_config(tmp_path)
    first = write_generation_run_artifacts(
        config,
        requested_dataset_name="retrieval",
        resolved_dataset_name="retrieval",
        dataset_path=tmp_path / "artifacts" / "retrieval",
        output_path=tmp_path / "generated" / "retrieval.jsonl",
        requested_num_records=1,
        actual_num_records=1,
        producer_version="0.1.0",
    )
    first_provenance = first.provenance_path.read_text(encoding="utf-8")

    second = write_generation_run_artifacts(
        config,
        requested_dataset_name="retrieval",
        resolved_dataset_name="retrieval_08-04-2026_120000",
        dataset_path=tmp_path / "artifacts" / "retrieval_08-04-2026_120000",
        output_path=tmp_path / "generated" / "retrieval_08-04-2026_120000.jsonl",
        requested_num_records=1,
        actual_num_records=1,
        producer_version="0.1.0",
    )

    assert first.provenance_path.read_text(encoding="utf-8") == first_provenance
    assert second.provenance_path.parent.name == "retrieval_08-04-2026_120000"


def test_conversion_run_artifacts_capture_completed_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "records.jsonl"
    config = ConversionRunConfig(
        input_path=input_path,
        corpus_id="corpus",
        output_dir=tmp_path / "converted",
    )
    result = ConversionResult(
        output_dir=tmp_path / "converted",
        train_file=tmp_path / "converted" / "train.jsonl",
        validation_file=tmp_path / "converted" / "validation.jsonl",
        corpus_dir=tmp_path / "converted" / "corpus",
        evaluation_dir=tmp_path / "converted" / "evaluation",
        training_examples=10,
        validation_examples=2,
        evaluation_queries=3,
    )

    artifacts = write_conversion_run_artifacts(
        config,
        result=result,
        producer_version="0.1.0",
        override_paths=["corpus_id"],
    )

    resolved = yaml.safe_load(artifacts.resolved_config_path.read_text(encoding="utf-8"))
    provenance = json.loads(artifacts.provenance_path.read_text(encoding="utf-8"))
    assert resolved["input_path"] == str(input_path)
    assert resolved["output_dir"] == str(tmp_path / "converted")
    assert provenance["override_paths"] == ["corpus_id"]
    assert provenance["output_paths"] == {
        "output_dir": str(result.output_dir.resolve()),
        "train_file": str(result.train_file.resolve()),
        "validation_file": str(result.validation_file.resolve()),
        "corpus_dir": str(result.corpus_dir.resolve()),
        "evaluation_dir": str(result.evaluation_dir.resolve()),
    }
    assert provenance["record_counts"] == {
        "training_examples": 10,
        "validation_examples": 2,
        "evaluation_queries": 3,
    }
    assert "input_fingerprint" not in provenance
    assert "config_fingerprint" not in provenance
