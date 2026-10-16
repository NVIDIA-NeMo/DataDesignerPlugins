# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-file loading for multimodal retrieval generation."""

from __future__ import annotations

from pathlib import Path

from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource


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
