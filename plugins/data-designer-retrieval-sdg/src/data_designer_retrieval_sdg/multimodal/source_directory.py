# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare canonical multimodal sources from a directory of local files."""

from __future__ import annotations

import fnmatch
import hashlib
import io
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from data_designer_retrieval_sdg.multimodal.source_models import (
    TEXT_EXTENSIONS,
    RetrievalSource,
    SourceDirectoryConfig,
)
from data_designer_retrieval_sdg.multimodal.text_preparation import load_text_token_budget, prepare_text_chunks


def discover_source_files(config: SourceDirectoryConfig) -> list[Path]:
    """Select supported files in sorted corpus-relative path order.

    Args:
        config: Directory, basename glob, extensions, recursion, and file limit.

    Returns:
        Absolute paths in deterministic order, limited after filtering.

    Raises:
        ValueError: The root is not a directory, no files match, or a selected
            entry is a symbolic link or nonregular file.
        OSError: A directory cannot be read.
    """
    root = config.path.resolve()
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist or is not a directory: {root}")
    paths = []
    for directory, subdirectories, names in os.walk(root, onerror=raise_directory_error, followlinks=False):
        parent = Path(directory)
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if config.recursive and not name.startswith(".") and not (parent / name).is_symlink()
        )
        for name in names:
            if name.startswith(".") or not fnmatch.fnmatchcase(name, config.file_pattern):
                continue
            path = parent / name
            if path.suffix.lower() not in config.file_extensions:
                continue
            paths.append(path)
    paths.sort(key=lambda path: path.relative_to(root).as_posix())
    if config.num_files is not None:
        paths = paths[: config.num_files]
    if not paths:
        raise ValueError(f"No supported files matched the source directory filters: {root}")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Source must be a regular file, not a symbolic link: {path}")
    return paths


def raise_directory_error(error: OSError) -> None:
    """Propagate discovery errors instead of silently omitting a subtree.

    Args:
        error: Filesystem error reported by directory discovery.

    Raises:
        OSError: The supplied discovery error.
    """
    raise error


def validate_directory_image(content: bytes, path: Path) -> None:
    """Decode a static supported image before any model requests.

    Args:
        content: Bytes read from the selected source file.
        path: Source path for actionable error messages.

    Raises:
        ValueError: The bytes are damaged, animated, or not PNG, JPEG, or WebP.
    """
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"} or getattr(image, "n_frames", 1) != 1:
                raise ValueError("expected one static PNG, JPEG, or WebP image")
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Invalid source image {path}: {exc}") from exc


def load_directory_sources(config: SourceDirectoryConfig) -> tuple[list[RetrievalSource], dict]:
    """Read files into canonical sources and a checksummed input manifest.

    Each text file becomes one document containing sentence chunks or one
    whole-file unit. Each image becomes its own document and unit. Matching names
    do not imply a text/image pairing or a page number; use explicit JSONL for
    those relationships.

    Args:
        config: Validated directory preparation settings.

    Returns:
        Ordered sources and their input manifest, containing file paths, raw
        byte hashes, byte counts, unit IDs, and optional tokenizer metadata.

    Raises:
        ValueError: Discovery fails, text is empty or not UTF-8, an image or
            tokenizer is invalid, or text exceeds its token budget. Selected
            files are never silently skipped.
        OSError: A selected source file cannot be read.
    """
    root = config.path.resolve()
    paths = discover_source_files(config)
    budget = load_text_token_budget(config) if any(path.suffix.lower() in TEXT_EXTENSIONS for path in paths) else None
    sources, files = [], []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        document_id = quote(relative, safe="/")
        content = path.read_bytes()
        if path.suffix.lower() in TEXT_EXTENSIONS:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Source text must be UTF-8: {path}") from exc
            if not text.strip():
                raise ValueError(f"Source text contains no nonwhitespace content: {path}")
            chunks = prepare_text_chunks(text, config, budget, path)
            records = [
                RetrievalSource(
                    document_id=document_id,
                    unit_id=f"{document_id}#chunk-{index:06d}",
                    text=chunk,
                    language=config.language,
                    source_uri=document_id,
                )
                for index, chunk in enumerate(chunks, 1)
            ]
        else:
            validate_directory_image(content, path)
            records = [
                RetrievalSource(
                    document_id=document_id,
                    unit_id=f"{document_id}#image",
                    images=[str(path)],
                    language=config.language,
                    source_uri=document_id,
                )
            ]
        sources.extend(records)
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "unit_ids": [record.unit_id for record in records],
            }
        )
    manifest = {"schema_version": 1, "files": files}
    if budget is not None:
        manifest["source_preparation"] = budget.metadata()
    return sources, manifest
