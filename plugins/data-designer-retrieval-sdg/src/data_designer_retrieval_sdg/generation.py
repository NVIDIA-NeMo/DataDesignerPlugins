# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable Python API for running retrieval synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import data_designer.config as dd
from data_designer.engine.resources.seed_reader import SeedReaderError
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface import DataDesigner

from data_designer_retrieval_sdg.pipeline import build_model_providers, build_qa_generation_pipeline
from data_designer_retrieval_sdg.retrieval.source_file import (
    RetrievalSourcesFile,
    snapshot_retrieval_sources,
    validate_generated_source_coverage,
)
from data_designer_retrieval_sdg.run_artifacts import write_generation_run_artifacts
from data_designer_retrieval_sdg.run_config import ConfigSource, GenerationRunConfig, config_source_from_path
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
    resolved_config_path: Path | None = None
    provenance_path: Path | None = None


@dataclass(frozen=True)
class GenerationPreviewResult:
    """Summary of a completed generation preview."""

    num_seed_records: int
    num_preview_records: int


def _prepare_seed_source(
    source: DocumentChunkerSeedSource | RetrievalSourcesFile,
    artifact_path: Path,
) -> DocumentChunkerSeedSource | dd.LocalFileSeedSource:
    """Translate canonical input into native seeds without changing the pipeline."""
    if isinstance(source, RetrievalSourcesFile):
        snapshot = snapshot_retrieval_sources(source.path, artifact_path)
        return dd.LocalFileSeedSource(path=str(snapshot))
    return source


def _count_seed_records(seed_source: DocumentChunkerSeedSource | dd.LocalFileSeedSource) -> int:
    """Return the number of records produced by a seed source manifest."""
    if isinstance(seed_source, dd.LocalFileSeedSource):
        with Path(seed_source.path).open(encoding="utf-8") as stream:
            return sum(1 for _ in stream)
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


def _resolve_dataset_name(
    seed_source: DocumentChunkerSeedSource | RetrievalSourcesFile, artifact_path: Path, dataset_name: str | None
) -> str:
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


def run_generation(
    config: GenerationRunConfig,
    *,
    config_sources: Sequence[ConfigSource] = (),
    override_paths: Sequence[str] = (),
    environment_variables: Sequence[str] = (),
) -> GenerationResult:
    """Generate and export one retrieval SDG dataset.

    Args:
        config: Fully translated generation run configuration.
        config_sources: Hashed configuration files used to resolve *config*.
        override_paths: Dotted paths explicitly overridden after file loading.
        environment_variables: Names of explicit environment-backed values.

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

    model_providers, _ = build_model_providers(model_providers=config.model_providers)
    config = config.model_copy(update={"model_providers": model_providers})
    if isinstance(config.seed_source, RetrievalSourcesFile):
        source = config.seed_source.model_copy(update={"path": config.seed_source.path.resolve()})
        config = config.model_copy(update={"seed_source": source})
        config_sources = (*config_sources, config_source_from_path(source.path))
    dataset_name = _resolve_dataset_name(config.seed_source, config.artifact_path, config.dataset_name)
    seed_source = _prepare_seed_source(config.seed_source, config.artifact_path)
    num_records = config.num_records if config.num_records is not None else _count_seed_records(seed_source)
    if isinstance(seed_source, dd.LocalFileSeedSource) and num_records > _count_seed_records(seed_source):
        raise ValueError("num_records exceeds the available canonical retrieval sources")
    if num_records <= 0:
        raise SeedReaderError("The seed source produced no records")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    producer_version = _producer_version()
    data_designer = DataDesigner(artifact_path=config.artifact_path, model_providers=config.model_providers)
    data_designer.set_run_config(dd.RunConfig(disable_early_shutdown=True, buffer_size=config.buffer_size))

    config_builder = build_qa_generation_pipeline(
        seed_source=seed_source,
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
    if isinstance(seed_source, dd.LocalFileSeedSource):
        validate_generated_source_coverage(Path(seed_source.path), output_path, num_records)
        if actual_num_records != num_records:
            raise ValueError("Generated record count does not match the canonical source selection")
        config_sources = (*config_sources, config_source_from_path(seed_source.path))
    dataset_path = Path(result.artifact_storage.base_dataset_path)
    run_artifacts = write_generation_run_artifacts(
        config,
        requested_dataset_name=dataset_name,
        resolved_dataset_name=resolved_dataset_name,
        dataset_path=dataset_path,
        output_path=output_path,
        requested_num_records=num_records,
        actual_num_records=actual_num_records,
        producer_version=producer_version,
        sources=config_sources,
        override_paths=override_paths,
        environment_variables=environment_variables,
    )

    return GenerationResult(
        output_path=output_path,
        dataset_path=dataset_path,
        dataset_name=resolved_dataset_name,
        num_records=actual_num_records,
        requested_num_records=num_records,
        producer_version=producer_version,
        resolved_config_path=run_artifacts.resolved_config_path,
        provenance_path=run_artifacts.provenance_path,
    )


def preview_generation(config: GenerationRunConfig, num_records: int = 1) -> GenerationPreviewResult:
    """Preview the shared pipeline without persisting a generated dataset.

    Canonical inputs and image copies are retained under ``artifact_path`` so
    preview and generation use the same content-addressed source representation.

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

    model_providers, _ = build_model_providers(model_providers=config.model_providers)
    config = config.model_copy(update={"model_providers": model_providers})
    seed_source = _prepare_seed_source(config.seed_source, config.artifact_path)
    total_records = config.num_records if config.num_records is not None else _count_seed_records(seed_source)
    if isinstance(seed_source, dd.LocalFileSeedSource) and total_records > _count_seed_records(seed_source):
        raise ValueError("num_records exceeds the available canonical retrieval sources")
    if total_records <= 0:
        raise SeedReaderError("The seed source produced no records")

    data_designer = DataDesigner(artifact_path=config.artifact_path, model_providers=config.model_providers)
    data_designer.set_run_config(dd.RunConfig(disable_early_shutdown=True, buffer_size=config.buffer_size))
    config_builder = build_qa_generation_pipeline(
        seed_source=seed_source,
        start_index=0,
        end_index=min(config.buffer_size - 1, total_records - 1),
        **config.pipeline.to_pipeline_kwargs(),
    )
    preview_result = data_designer.preview(config_builder, num_records=num_records)
    preview_result.display_sample_record()
    return GenerationPreviewResult(num_seed_records=total_records, num_preview_records=num_records)
