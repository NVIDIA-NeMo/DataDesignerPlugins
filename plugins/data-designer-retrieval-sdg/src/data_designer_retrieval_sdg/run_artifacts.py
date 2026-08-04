# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved configuration snapshots for completed SDG runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import yaml
from data_designer.engine.storage.artifact_storage import ResumeMode

from data_designer_retrieval_sdg.run_config import (
    ConfigSource,
    ConversionRunConfig,
    GenerationRunConfig,
)

if TYPE_CHECKING:
    from data_designer_retrieval_sdg.convert import ConversionResult

RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
CONFIG_PROVENANCE_FILENAME = "config_provenance.json"
GENERATION_METADATA_DIR = ".retrieval_sdg_runs"
CONVERSION_METADATA_DIR = ".retrieval_sdg_run"


@dataclass(frozen=True)
class RunArtifactPaths:
    """Paths written for one completed run."""

    resolved_config_path: Path
    provenance_path: Path


def _write_atomic(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 run metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _write_run_artifacts(
    metadata_dir: Path,
    config: GenerationRunConfig | ConversionRunConfig,
    provenance: dict[str, object],
) -> RunArtifactPaths:
    """Persist one redacted config and its completed-run provenance."""
    resolved_config_path = metadata_dir / RESOLVED_CONFIG_FILENAME
    provenance_path = metadata_dir / CONFIG_PROVENANCE_FILENAME
    _write_atomic(resolved_config_path, yaml.safe_dump(config.to_redacted_dict(), sort_keys=False))
    _write_atomic(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return RunArtifactPaths(
        resolved_config_path=resolved_config_path,
        provenance_path=provenance_path,
    )


def write_generation_run_artifacts(
    config: GenerationRunConfig,
    *,
    requested_dataset_name: str,
    resolved_dataset_name: str,
    dataset_path: Path,
    output_path: Path,
    requested_num_records: int,
    actual_num_records: int,
    producer_version: str,
    sources: Sequence[ConfigSource] = (),
    override_paths: Sequence[str] = (),
    environment_variables: Sequence[str] = (),
) -> RunArtifactPaths:
    """Persist a redacted snapshot after generation and export complete."""
    effective_config = config.model_copy(
        update={"dataset_name": resolved_dataset_name, "num_records": requested_num_records}
    )
    metadata_dir = config.artifact_path / GENERATION_METADATA_DIR / resolved_dataset_name
    provenance = {
        "schema_version": 1,
        "operation": "generation",
        "producer_version": producer_version,
        "requested_dataset_name": requested_dataset_name,
        "resolved_dataset_name": resolved_dataset_name,
        "resume_mode": ResumeMode(effective_config.resume).value,
        "config_sources": [source.to_dict() for source in sources],
        "override_paths": list(override_paths),
        "environment_variables": sorted(set(environment_variables)),
        "output_paths": {
            "dataset": str(dataset_path.resolve()),
            "jsonl": str(output_path.resolve()),
        },
        "record_counts": {
            "requested": requested_num_records,
            "generated": actual_num_records,
        },
    }
    return _write_run_artifacts(metadata_dir, effective_config, provenance)


def write_conversion_run_artifacts(
    config: ConversionRunConfig,
    *,
    result: ConversionResult,
    producer_version: str,
    sources: Sequence[ConfigSource] = (),
    override_paths: Sequence[str] = (),
) -> RunArtifactPaths:
    """Persist a redacted snapshot after conversion completes."""
    effective_config = config.model_copy(update={"output_dir": result.output_dir})
    metadata_dir = result.output_dir / CONVERSION_METADATA_DIR
    provenance = {
        "schema_version": 1,
        "operation": "conversion",
        "producer_version": producer_version,
        "config_sources": [source.to_dict() for source in sources],
        "override_paths": list(override_paths),
        "output_paths": {
            "output_dir": str(result.output_dir.resolve()),
            "train_file": str(result.train_file.resolve()) if result.train_file is not None else None,
            "validation_file": (str(result.validation_file.resolve()) if result.validation_file is not None else None),
            "corpus_dir": str(result.corpus_dir.resolve()) if result.corpus_dir is not None else None,
            "evaluation_dir": str(result.evaluation_dir.resolve()),
        },
        "record_counts": {
            "training_examples": result.training_examples,
            "validation_examples": result.validation_examples,
            "evaluation_queries": result.evaluation_queries,
        },
    }
    return _write_run_artifacts(metadata_dir, effective_config, provenance)
