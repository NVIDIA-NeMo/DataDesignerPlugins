# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable Python API for running retrieval synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import data_designer.config as dd
from data_designer.engine.resources.seed_reader import SeedReaderError
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface import DataDesigner

from data_designer_retrieval_sdg.pipeline import build_qa_generation_pipeline
from data_designer_retrieval_sdg.run_config import GenerationRunConfig
from data_designer_retrieval_sdg.seed_reader import DocumentChunkerSeedReader
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


@dataclass(frozen=True)
class GenerationResult:
    """Artifacts produced by a completed generation run."""

    output_path: Path
    dataset_path: Path
    dataset_name: str
    num_records: int
    requested_num_records: int
    producer_version: str


@dataclass(frozen=True)
class GenerationPreviewResult:
    """Summary of a completed generation preview."""

    num_seed_records: int
    num_preview_records: int


def _count_seed_records(seed_source: DocumentChunkerSeedSource) -> int:
    """Return the number of records produced by a seed source manifest."""
    reader = DocumentChunkerSeedReader()
    reader.attach(seed_source, PlaintextResolver())
    return reader.get_seed_dataset_size()


def _path_is_relative_to(path: Path, root: Path) -> bool:
    """Return whether *path* is contained by *root* after resolution."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_dataset_name(dataset_name: str, artifact_path: Path) -> str:
    """Validate a Data Designer dataset name used as an artifact path segment."""
    if not dataset_name:
        raise ValueError("--dataset-name must not be empty")
    if dataset_name in {".", ".."}:
        raise ValueError("--dataset-name must be a real path segment, not '.' or '..'")
    if any(ord(char) < 32 or ord(char) == 127 for char in dataset_name):
        raise ValueError("--dataset-name must not contain control characters")
    if any(separator in dataset_name for separator in ("/", "\\")):
        raise ValueError("--dataset-name must be a single path segment without path separators")

    dataset_path = Path(dataset_name)
    if dataset_path.is_absolute() or len(dataset_path.parts) != 1:
        raise ValueError("--dataset-name must be a single relative path segment")

    artifact_root = artifact_path.resolve()
    resolved_dataset_path = (artifact_root / dataset_name).resolve()
    if resolved_dataset_path == artifact_root or not _path_is_relative_to(resolved_dataset_path, artifact_root):
        raise ValueError("--dataset-name must resolve under --artifact-path")
    return dataset_name


def _resolve_dataset_name(seed_source: DocumentChunkerSeedSource, artifact_path: Path, dataset_name: str | None) -> str:
    """Return an explicit or source-derived dataset name after validation."""
    source_name = Path(str(seed_source.path)).name
    resolved_name = dataset_name if dataset_name is not None else source_name or "retrieval_sdg"
    return _validate_dataset_name(resolved_name, artifact_path)


def _producer_version() -> str:
    """Return the installed package version, including editable checkouts."""
    try:
        return version("data-designer-retrieval-sdg")
    except PackageNotFoundError:
        return "0+unknown"


def run_generation(config: GenerationRunConfig) -> GenerationResult:
    """Generate and export one retrieval SDG dataset.

    Args:
        config: Fully translated generation run configuration.

    Returns:
        Immutable metadata describing the exported data and Data Designer
        artifact paths.

    Raises:
        SeedReaderError: If the seed source cannot produce a manifest.
        ValueError: If the dataset name, record count, or run settings are
            invalid.
    """
    if config.buffer_size <= 0:
        raise ValueError("buffer_size must be greater than zero")

    dataset_name = _resolve_dataset_name(config.seed_source, config.artifact_path, config.dataset_name)
    num_records = config.num_records if config.num_records is not None else _count_seed_records(config.seed_source)
    if num_records <= 0:
        raise SeedReaderError("The seed source produced no records")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    data_designer = DataDesigner(artifact_path=config.artifact_path, model_providers=config.model_providers)
    data_designer.set_run_config(dd.RunConfig(disable_early_shutdown=True, buffer_size=config.buffer_size))

    config_builder = build_qa_generation_pipeline(
        seed_source=config.seed_source,
        start_index=0,
        end_index=num_records - 1,
        **config.pipeline.to_pipeline_kwargs(),
    )
    result = data_designer.create(
        config_builder,
        num_records=num_records,
        dataset_name=dataset_name,
        resume=ResumeMode(config.resume),
    )

    resolved_dataset_name = result.artifact_storage.resolved_dataset_name
    output_path = config.output_dir / f"{resolved_dataset_name}.jsonl"
    result.export(output_path, format="jsonl")
    actual_num_records = result.count_records()

    return GenerationResult(
        output_path=output_path,
        dataset_path=Path(result.artifact_storage.base_dataset_path),
        dataset_name=resolved_dataset_name,
        num_records=actual_num_records,
        requested_num_records=num_records,
        producer_version=_producer_version(),
    )


def preview_generation(config: GenerationRunConfig, num_records: int = 1) -> GenerationPreviewResult:
    """Run a non-persisted preview using the same translated generation config.

    Args:
        config: Fully translated generation run configuration.
        num_records: Number of preview records requested from Data Designer.

    Returns:
        Counts describing the seed dataset and preview request.

    Raises:
        SeedReaderError: If the seed source cannot produce a manifest.
        ValueError: If the requested preview or run settings are invalid.
    """
    if config.buffer_size <= 0:
        raise ValueError("buffer_size must be greater than zero")
    if num_records <= 0:
        raise ValueError("num_records must be greater than zero")

    total_records = config.num_records if config.num_records is not None else _count_seed_records(config.seed_source)
    if total_records <= 0:
        raise SeedReaderError("The seed source produced no records")

    data_designer = DataDesigner(artifact_path=config.artifact_path, model_providers=config.model_providers)
    data_designer.set_run_config(dd.RunConfig(disable_early_shutdown=True, buffer_size=config.buffer_size))
    config_builder = build_qa_generation_pipeline(
        seed_source=config.seed_source,
        start_index=0,
        end_index=min(config.buffer_size - 1, total_records - 1),
        **config.pipeline.to_pipeline_kwargs(),
    )
    preview_result = data_designer.preview(config_builder, num_records=num_records)
    preview_result.display_sample_record()
    return GenerationPreviewResult(num_seed_records=total_records, num_preview_records=num_records)
