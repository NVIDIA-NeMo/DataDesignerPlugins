# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved configuration snapshots and input fingerprints for SDG runs."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import yaml
from data_designer.engine.storage.artifact_storage import ResumeMode

from data_designer_retrieval_sdg.run_config import (
    ConfigSource,
    ConversionRunConfig,
    GenerationRunConfig,
)

RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
CONFIG_PROVENANCE_FILENAME = "config_provenance.json"
GENERATION_METADATA_DIR = ".retrieval_sdg_runs"
CONVERSION_METADATA_DIR = ".retrieval_sdg_run"
_ENV_VAR_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True)
class InputFingerprint:
    """Digest and file count for one normalized input selection."""

    sha256: str
    file_count: int


@dataclass(frozen=True)
class RunArtifactPaths:
    """Paths and fingerprints written for one resolved run."""

    resolved_config_path: Path
    provenance_path: Path
    config_fingerprint: str
    input_fingerprint: str


def _canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using stable canonical encoding."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hash_files(root: Path, relative_paths: Sequence[str]) -> InputFingerprint:
    """Hash selected relative paths and file bytes without ambiguous concatenation."""
    digest = hashlib.sha256()
    digest.update(f"file-count:{len(relative_paths)}\n".encode())
    for relative_path in relative_paths:
        normalized_path = PurePosixPath(relative_path).as_posix()
        encoded_path = normalized_path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        absolute_path = root / Path(relative_path)
        with absolute_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(len(chunk).to_bytes(8, "big"))
                digest.update(chunk)
        digest.update((0).to_bytes(8, "big"))
    return InputFingerprint(sha256=digest.hexdigest(), file_count=len(relative_paths))


def _path_matches_extensions(relative_path: str, extensions: list[str] | None) -> bool:
    """Mirror document-chunker extension matching without importing reader internals."""
    if not extensions:
        return True
    extension_set = {extension.lower() for extension in extensions}
    relative = PurePosixPath(relative_path)
    if relative.suffix.lower() in extension_set:
        return True
    return "" in extension_set and "." not in relative.name


def fingerprint_seed_input(config: GenerationRunConfig) -> InputFingerprint:
    """Hash the source files selected by the generation seed configuration."""
    root = Path(str(config.seed_source.path)).resolve()
    if not root.is_dir():
        raise ValueError(f"Seed source path does not exist or is not a directory: {root}")

    candidates: Iterable[Path] = root.rglob("*") if config.seed_source.recursive else root.iterdir()
    relative_paths = sorted(
        path.relative_to(root).as_posix()
        for path in candidates
        if path.is_file()
        and fnmatch.fnmatchcase(path.name, config.seed_source.file_pattern)
        and _path_matches_extensions(path.relative_to(root).as_posix(), config.seed_source.file_extensions)
    )
    if config.seed_source.num_files is not None:
        relative_paths = relative_paths[: config.seed_source.num_files]
    if not relative_paths:
        raise ValueError(f"No source files selected under {root}")

    fingerprint = _hash_files(root, relative_paths)
    if config.seed_source.multi_doc_manifest is None:
        return fingerprint

    manifest_path = Path(config.seed_source.multi_doc_manifest).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Multi-document manifest does not exist: {manifest_path}")
    digest = hashlib.sha256()
    digest.update(fingerprint.sha256.encode("ascii"))
    manifest_bytes = manifest_path.read_bytes()
    digest.update(len(manifest_bytes).to_bytes(8, "big"))
    digest.update(manifest_bytes)
    return InputFingerprint(sha256=digest.hexdigest(), file_count=fingerprint.file_count + 1)


def _discover_conversion_files(input_path: Path) -> tuple[Path, list[str]]:
    """Return the same unambiguous input format class used by conversion."""
    if input_path.is_file():
        if input_path.suffix.lower() not in {".json", ".jsonl", ".parquet"}:
            raise ValueError(f"Unsupported generated data file format: {input_path}")
        return input_path.parent, [input_path.name]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    jsonl_files = sorted(path for path in input_path.glob("*.jsonl") if path.is_file())
    legacy_json_files = sorted(path for path in input_path.glob("generated_batch*.json") if path.is_file())
    json_files = legacy_json_files or sorted(path for path in input_path.glob("*.json") if path.is_file())
    parquet_files = sorted(path for path in input_path.glob("*.parquet") if path.is_file())
    discovered = [files for files in (jsonl_files, json_files, parquet_files) if files]
    if len(discovered) > 1:
        raise ValueError(
            f"Mixed generated-data formats found in {input_path}; "
            "pass an exact input file or a directory containing only one format class."
        )
    if not discovered:
        raise ValueError(f"No generated JSONL, JSON, or parquet files found in {input_path}")
    return input_path, [path.name for path in discovered[0]]


def fingerprint_conversion_input(config: ConversionRunConfig) -> InputFingerprint:
    """Hash the exact generated records selected for conversion."""
    root, relative_paths = _discover_conversion_files(config.input_path.resolve())
    input_fingerprint = _hash_files(root, relative_paths)
    if not config.groups_json:
        return input_fingerprint

    digest = hashlib.sha256()
    digest.update(input_fingerprint.sha256.encode("ascii"))
    for group_path in sorted(path.resolve() for path in config.groups_json):
        if not group_path.is_file():
            raise ValueError(f"Dedup group file does not exist: {group_path}")
        encoded_path = str(group_path).encode("utf-8")
        group_bytes = group_path.read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(group_bytes).to_bytes(8, "big"))
        digest.update(group_bytes)
    return InputFingerprint(
        sha256=digest.hexdigest(),
        file_count=input_fingerprint.file_count + len(config.groups_json),
    )


def _generation_fingerprint_payload(config: GenerationRunConfig) -> dict[str, Any]:
    """Return only settings that can affect generated record content."""
    payload = config.to_redacted_dict()
    for key in ("output_dir", "artifact_path", "dataset_name", "buffer_size", "resume", "log_level"):
        payload.pop(key, None)
    return payload


def _conversion_fingerprint_payload(config: ConversionRunConfig) -> dict[str, Any]:
    """Return only settings that can affect converted record content."""
    payload = config.to_redacted_dict()
    for key in ("input_path", "output_dir"):
        payload.pop(key, None)
    return payload


def _credential_environment_variables(config: GenerationRunConfig) -> list[str]:
    """Record provider credential env-var names only when they are known env vars."""
    environment_variables: set[str] = set()
    for provider in config.model_providers or []:
        api_key = provider.api_key
        if isinstance(api_key, str) and _ENV_VAR_PATTERN.fullmatch(api_key) and api_key in os.environ:
            environment_variables.add(api_key)
    return sorted(environment_variables)


def _write_atomic(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 run metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _read_existing_provenance(path: Path) -> dict[str, Any]:
    """Read an existing provenance document or fail with an actionable error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot validate resume metadata at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Cannot validate resume metadata at {path}: expected a JSON object")
    return value


def _validate_resume(
    config: GenerationRunConfig,
    dataset_name: str,
    provenance_path: Path,
    *,
    producer_version: str,
    config_fingerprint: str,
    corpus_fingerprint: str,
) -> None:
    """Refuse resume when prior data-affecting settings or source bytes differ."""
    if ResumeMode(config.resume) == ResumeMode.NEVER:
        return

    if not provenance_path.exists():
        dataset_path = config.artifact_path / dataset_name
        if dataset_path.exists():
            raise ValueError(
                f"Resume refused for dataset {dataset_name!r}: existing artifacts have no "
                f"{CONFIG_PROVENANCE_FILENAME}. Use resume=never with a new dataset_name."
            )
        return

    existing = _read_existing_provenance(provenance_path)
    mismatches = []
    if existing.get("producer_version") != producer_version:
        mismatches.append("plugin version")
    if existing.get("config_fingerprint") != config_fingerprint:
        mismatches.append("resolved generation settings")
    if existing.get("input_fingerprint") != corpus_fingerprint:
        mismatches.append("source corpus")
    if mismatches:
        changed = " and ".join(mismatches)
        raise ValueError(
            f"Resume refused for dataset {dataset_name!r}: {changed} changed. Use resume=never with a new dataset_name."
        )


def write_generation_run_artifacts(
    config: GenerationRunConfig,
    *,
    dataset_name: str,
    num_records: int,
    producer_version: str,
    sources: Sequence[ConfigSource] = (),
    override_paths: Sequence[str] = (),
    environment_variables: Sequence[str] = (),
) -> RunArtifactPaths:
    """Validate resume identity and persist a redacted generation run snapshot."""
    effective_config = config.model_copy(update={"dataset_name": dataset_name, "num_records": num_records})
    input_fingerprint = fingerprint_seed_input(effective_config)
    config_fingerprint = _canonical_sha256(_generation_fingerprint_payload(effective_config))
    metadata_root = config.artifact_path / GENERATION_METADATA_DIR
    metadata_dir = metadata_root / dataset_name
    if ResumeMode(config.resume) == ResumeMode.NEVER and (
        metadata_dir.exists() or (config.artifact_path / dataset_name).exists()
    ):
        metadata_dir = metadata_root / ".pending" / f"{dataset_name}-{uuid.uuid4().hex}"
    resolved_config_path = metadata_dir / RESOLVED_CONFIG_FILENAME
    provenance_path = metadata_dir / CONFIG_PROVENANCE_FILENAME

    _validate_resume(
        effective_config,
        dataset_name,
        provenance_path,
        producer_version=producer_version,
        config_fingerprint=config_fingerprint,
        corpus_fingerprint=input_fingerprint.sha256,
    )

    provenance = {
        "schema_version": 1,
        "producer_version": producer_version,
        "dataset_name": dataset_name,
        "resume_mode": ResumeMode(effective_config.resume).value,
        "config_sources": [source.to_dict() for source in sources],
        "override_paths": list(override_paths),
        "environment_variables": sorted(
            set(environment_variables) | set(_credential_environment_variables(effective_config))
        ),
        "config_fingerprint": config_fingerprint,
        "input_fingerprint": input_fingerprint.sha256,
        "input_file_count": input_fingerprint.file_count,
    }
    _write_atomic(resolved_config_path, yaml.safe_dump(effective_config.to_redacted_dict(), sort_keys=False))
    _write_atomic(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return RunArtifactPaths(
        resolved_config_path=resolved_config_path,
        provenance_path=provenance_path,
        config_fingerprint=config_fingerprint,
        input_fingerprint=input_fingerprint.sha256,
    )


def finalize_generation_run_artifacts(
    artifacts: RunArtifactPaths,
    config: GenerationRunConfig,
    *,
    resolved_dataset_name: str,
) -> RunArtifactPaths:
    """Attach staged fresh-run metadata to Data Designer's resolved dataset name."""
    if ResumeMode(config.resume) != ResumeMode.NEVER:
        return artifacts

    metadata_root = config.artifact_path / GENERATION_METADATA_DIR
    target_dir = metadata_root / resolved_dataset_name
    current_dir = artifacts.resolved_config_path.parent
    if current_dir != target_dir:
        if target_dir.exists():
            raise ValueError(
                f"Cannot attach run metadata to resolved dataset {resolved_dataset_name!r}: {target_dir} already exists"
            )
        target_dir.mkdir(parents=True)
        resolved_config_path = target_dir / RESOLVED_CONFIG_FILENAME
        provenance_path = target_dir / CONFIG_PROVENANCE_FILENAME
        artifacts.resolved_config_path.replace(resolved_config_path)
        artifacts.provenance_path.replace(provenance_path)
        current_dir.rmdir()
        if current_dir.parent.name == ".pending" and not any(current_dir.parent.iterdir()):
            current_dir.parent.rmdir()
    else:
        resolved_config_path = artifacts.resolved_config_path
        provenance_path = artifacts.provenance_path

    provenance = _read_existing_provenance(provenance_path)
    provenance["resolved_dataset_name"] = resolved_dataset_name
    _write_atomic(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return RunArtifactPaths(
        resolved_config_path=resolved_config_path,
        provenance_path=provenance_path,
        config_fingerprint=artifacts.config_fingerprint,
        input_fingerprint=artifacts.input_fingerprint,
    )


def write_conversion_run_artifacts(
    config: ConversionRunConfig,
    *,
    output_dir: Path,
    producer_version: str,
    sources: Sequence[ConfigSource] = (),
    override_paths: Sequence[str] = (),
) -> RunArtifactPaths:
    """Persist a redacted conversion run snapshot and exact input fingerprint."""
    effective_config = config.model_copy(update={"output_dir": output_dir})
    input_fingerprint = fingerprint_conversion_input(effective_config)
    config_fingerprint = _canonical_sha256(_conversion_fingerprint_payload(effective_config))
    metadata_dir = output_dir / CONVERSION_METADATA_DIR
    resolved_config_path = metadata_dir / RESOLVED_CONFIG_FILENAME
    provenance_path = metadata_dir / CONFIG_PROVENANCE_FILENAME
    provenance = {
        "schema_version": 1,
        "producer_version": producer_version,
        "config_sources": [source.to_dict() for source in sources],
        "override_paths": list(override_paths),
        "config_fingerprint": config_fingerprint,
        "input_fingerprint": input_fingerprint.sha256,
        "input_file_count": input_fingerprint.file_count,
    }
    _write_atomic(resolved_config_path, yaml.safe_dump(effective_config.to_redacted_dict(), sort_keys=False))
    _write_atomic(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return RunArtifactPaths(
        resolved_config_path=resolved_config_path,
        provenance_path=provenance_path,
        config_fingerprint=config_fingerprint,
        input_fingerprint=input_fingerprint.sha256,
    )
