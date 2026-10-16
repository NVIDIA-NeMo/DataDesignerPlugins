# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed source configuration for the document-chunker plugin."""

from __future__ import annotations

from typing import Literal

from data_designer.config.base import ConfigBase
from data_designer.config.seed_source import FileSystemSeedSource
from pydantic import Field


class DocumentChunkerSeedSource(FileSystemSeedSource, ConfigBase):
    """Load text files, sentence-chunk them, and build structured sections.

    Subclasses :class:`FileSystemSeedSource` (so the framework owns
    directory discovery, glob matching, and DuckDB registration) and
    :class:`ConfigBase` (required by ``assert_valid_plugin``).  This
    config layers chunking and multi-document bundling parameters on top.

    Inherited fields:
        path: Directory containing source text files.
        file_pattern: Filename glob (basenames only).  Defaults to ``"*"``.
        recursive: Whether to descend into subdirectories.

    Args:
        file_extensions: Optional list of allowed file extensions (e.g.
            ``[".txt", ".md"]``).  Filtered after glob matching against
            ``file_pattern``.  ``None`` disables extension filtering.
        min_text_length: Minimum character count to keep a document.
        sentences_per_chunk: Sentences grouped into a single chunk.
        num_sections: Sections to organise chunks into per row.
        num_files: Cap on the number of files to load (``None`` = no cap).
        multi_doc: If true, group files into multi-document bundles
            (one row per bundle) instead of one row per file.
        bundle_size: Documents per automatic bundle.
        bundle_strategy: ``"sequential"`` / ``"doc_balanced"`` /
            ``"interleaved"``; controls how chunks across documents are
            split into sections.
        max_docs_per_bundle: Hard cap on bundle size.
        multi_doc_manifest: Optional path to a JSON/YAML manifest
            defining explicit bundles; falls back to automatic bundling
            for any files not listed.
    """

    seed_type: Literal["document-chunker"] = "document-chunker"

    file_extensions: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of allowed file extensions (e.g. ['.txt', '.md']). "
            "Filtered after glob matching against file_pattern."
        ),
    )
    min_text_length: int = Field(default=0, ge=0)
    sentences_per_chunk: int = Field(default=5, ge=1)
    num_sections: int = Field(default=1, ge=1)
    num_files: int | None = Field(default=None, ge=1)

    multi_doc: bool = False
    bundle_size: int = Field(default=2, ge=1)
    bundle_strategy: Literal["sequential", "doc_balanced", "interleaved"] = "sequential"
    max_docs_per_bundle: int = Field(default=3, ge=1)
    multi_doc_manifest: str | None = None
