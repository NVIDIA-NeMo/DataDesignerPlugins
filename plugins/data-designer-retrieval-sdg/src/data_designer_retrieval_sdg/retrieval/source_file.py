# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File inputs for the shared retrieval generation runner."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from itertools import islice, zip_longest
from pathlib import Path
from typing import Literal

from data_designer.config.base import ConfigBase
from pydantic import ConfigDict

from data_designer_retrieval_sdg.retrieval.models import RetrievalSource


class RetrievalSourcesFile(ConfigBase):
    """Canonical JSONL input, with one retrieval source per nonblank line.

    Args:
        path: Source file. Relative image paths resolve against its directory.
        seed_type: Discriminator distinguishing canonical sources from chunking.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed_type: Literal["retrieval-sources"] = "retrieval-sources"
    path: Path


def load_retrieval_sources(path: str | Path) -> list[RetrievalSource]:
    """Read ordered sources and resolve their local image paths.

    Args:
        path: UTF-8 JSONL file containing canonical ``RetrievalSource`` objects.

    Returns:
        Validated sources with absolute image paths and unchanged identities.

    Raises:
        ValueError: If input is empty, malformed, or repeats a unit identifier.
        FileNotFoundError: If the source file or a referenced image is missing.
    """
    input_path = Path(path).resolve()
    sources: list[RetrievalSource] = []
    unit_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                source = RetrievalSource.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"Invalid retrieval source at line {line_number}") from exc
            if source.unit_id in unit_ids:
                raise ValueError(f"Duplicate retrieval unit_id at line {line_number}: {source.unit_id}")
            images = [(input_path.parent / image).resolve() for image in source.images]
            for image in images:
                if not image.is_file():
                    raise FileNotFoundError(f"Missing image for retrieval source at line {line_number}: {image}")
            sources.append(source.model_copy(update={"images": [str(image) for image in images]}))
            unit_ids.add(source.unit_id)
    if not sources:
        raise ValueError("The retrieval source file contains no records")
    return sources


def snapshot_retrieval_sources(path: str | Path, artifact_path: Path) -> Path:
    """Materialize content-addressed native seeds and image copies for safe resume.

    Args:
        path: Canonical retrieval source JSONL.
        artifact_path: Run artifact root owning reusable input snapshots.

    Returns:
        Native seed JSONL whose filename binds ordered records and image bytes.

    Raises:
        ValueError: If sources are invalid or a cached snapshot was modified.
        OSError: If reading inputs or storing the snapshot fails.
    """
    sources = load_retrieval_sources(path)
    root = artifact_path.resolve() / ".retrieval_sdg_inputs"
    records = []
    for source in sources:
        images = [_snapshot_image(Path(image), root / "assets") for image in source.images]
        copied = source.model_copy(update={"images": [str(image) for image in images]})
        records.append(copied.to_seed_record())
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records).encode("utf-8")
    snapshot = root / f"{hashlib.sha256(content).hexdigest()}.jsonl"
    _write_immutable(snapshot, content)
    return snapshot


def _snapshot_image(image: Path, destination: Path) -> Path:
    """Copy source pixels to a content-addressed asset without overwriting files."""
    content = image.read_bytes()
    target = destination / f"{hashlib.sha256(content).hexdigest()}{image.suffix.lower()}"
    _write_immutable(target, content)
    return target


def _write_immutable(path: Path, content: bytes) -> None:
    """Reuse identical snapshot bytes and fail rather than repair changed files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing retrieval input snapshot has changed: {path}")
        return
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=path.parent) as temporary_dir:
        temporary = Path(temporary_dir) / "content"
        temporary.write_bytes(content)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Existing retrieval input snapshot has changed: {path}") from None


def validate_generated_source_coverage(seed_path: Path, output_path: Path, num_records: int) -> None:
    """Require each selected canonical source to survive generation unchanged.

    Args:
        seed_path: Immutable native seed snapshot used by the generation run.
        output_path: Raw generated JSONL; rejected-query rows must remain present.
        num_records: Ordered seed prefix requested from Data Designer.

    Raises:
        ValueError: If source rows are missing, duplicated, reordered, or altered.
    """
    with seed_path.open(encoding="utf-8") as seeds, output_path.open(encoding="utf-8") as output:
        for index, (expected, observed) in enumerate(zip_longest(islice(seeds, num_records), output), 1):
            if expected is None or observed is None:
                raise ValueError(f"Generated source coverage differs at record {index}")
            seed = json.loads(expected)
            row = json.loads(observed)
            if not isinstance(row, dict) or any(
                row.get(key) != seed[key] for key in ("source_id", "retrieval_units", "language")
            ):
                raise ValueError(f"Generated source identity or content differs at record {index}")
