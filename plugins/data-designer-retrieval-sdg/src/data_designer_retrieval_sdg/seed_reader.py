# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem seed reader that loads, chunks, and sections text files.

Implements the framework's :class:`FileSystemSeedReader` contract: a cheap
``build_manifest`` that lists discovered files (or bundles), and an
expensive ``hydrate_row`` that reads file contents and produces the
chunked output rows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from data_designer.engine.resources.seed_reader import (
    FileSystemSeedReader,
    SeedReaderError,
    SeedReaderFileSystemContext,
)

from data_designer_retrieval_sdg.chunking import (
    build_bundle_id,
    build_bundles,
    chunks_to_sections_structured,
    load_multi_doc_manifest,
    text_to_sentence_chunks,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

logger = logging.getLogger(__name__)


def _path_matches_extensions(relative_path: str, extensions: list[str] | None) -> bool:
    """Return ``True`` when ``relative_path`` passes extension filtering.

    When ``extensions`` is ``None``, no filtering is applied.  A literal
    empty string ``""`` in the list matches files whose basename contains
    no dot (i.e. no extension).
    """
    if not extensions:
        return True
    ext_set = {e.lower() for e in extensions}
    suffix = PurePosixPath(relative_path).suffix.lower()
    if suffix in ext_set:
        return True
    if "" in ext_set and "." not in PurePosixPath(relative_path).name:
        return True
    return False


class DocumentChunkerSeedReader(FileSystemSeedReader[DocumentChunkerSeedSource]):
    """Sentence-chunk text files into a DataDesigner seed dataset.

    Output schema (one record per row):

    - ``file_name``: ``list[str]`` of relative paths (always a list,
      even in single-doc mode, for downstream uniformity).
    - ``text``: combined document text.  In multi-doc mode documents are
      joined with ``"\\n\\n=== Document Boundary ===\\n\\n"`` separators.
    - ``chunks``: ``list[dict]`` of sentence chunks with metadata.
    - ``sections_structured``: ``list[str]`` of formatted section blocks.
    - ``bundle_id``: stable hash of the bundle members (single-doc rows
      have an empty string).
    - ``bundle_members``: ``list[str]`` of relative paths (mirrors
      ``file_name``; preserved for backward compatibility).
    - ``is_multi_doc``: ``True`` when ``DocumentChunkerSeedSource.multi_doc``
      is enabled, ``False`` otherwise.
    """

    output_columns: ClassVar[list[str] | None] = [
        "file_name",
        "text",
        "chunks",
        "sections_structured",
        "bundle_id",
        "bundle_members",
        "is_multi_doc",
    ]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> list[dict[str, Any]]:
        """Discover files (and bundles) under ``context.root_path``.

        In single-doc mode each row references one file.  In multi-doc
        mode each row references a bundle of files; the bundle membership
        is JSON-encoded in ``bundle_members_json`` so the manifest stays
        a flat string-only schema (DuckDB-friendly).
        """
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        matched_paths = [p for p in matched_paths if _path_matches_extensions(p, self.source.file_extensions)]

        if self.source.num_files is not None:
            matched_paths = matched_paths[: self.source.num_files]

        if not matched_paths:
            raise SeedReaderError(
                f"No files matched extensions {self.source.file_extensions!r} under {context.root_path}"
            )

        if self.source.multi_doc:
            return self._build_multi_doc_manifest(matched_paths, context)
        return [{"bundle_members_json": json.dumps([p])} for p in matched_paths]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Read file contents for the manifest row and emit a chunked record.

        Returns an empty list when no file in the row passes
        ``min_text_length`` or no chunks are produced (the row is dropped).
        """
        members: list[str] = json.loads(manifest_row["bundle_members_json"])
        is_multi_doc = self.source.multi_doc

        if not is_multi_doc:
            record = self._hydrate_single(members[0], context)
            return [record] if record else []

        record = self._hydrate_bundle(members, context)
        return [record] if record else []

    def _build_multi_doc_manifest(
        self,
        matched_paths: list[str],
        context: SeedReaderFileSystemContext,
    ) -> list[dict[str, Any]]:
        manifest_path = Path(self.source.multi_doc_manifest) if self.source.multi_doc_manifest else None
        manifest_bundles = load_multi_doc_manifest(manifest_path)

        absolute_paths = [context.root_path / rel for rel in matched_paths]
        bundles = build_bundles(
            absolute_paths,
            bundle_size=self.source.bundle_size,
            max_docs_per_bundle=self.source.max_docs_per_bundle,
            manifest_bundles=manifest_bundles,
            input_dir=context.root_path,
        )
        if not bundles:
            raise SeedReaderError(f"build_bundles produced no bundles from {context.root_path}")

        manifest: list[dict[str, Any]] = []
        for bundle_paths in bundles:
            relative_members = [str(p.relative_to(context.root_path)) for p in bundle_paths]
            manifest.append({"bundle_members_json": json.dumps(relative_members)})
        return manifest

    def _read_file(self, relative_path: str, context: SeedReaderFileSystemContext) -> str | None:
        """Read a single file, returning ``None`` when it is too short or unreadable."""
        absolute_path = context.root_path / relative_path
        try:
            with context.fs.open(relative_path, "r", encoding="utf-8") as handle:
                content = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Skipping %s: %s", absolute_path, exc)
            return None

        if self.source.min_text_length > 0 and len(content) < self.source.min_text_length:
            return None
        return content

    def _hydrate_single(
        self,
        relative_path: str,
        context: SeedReaderFileSystemContext,
    ) -> dict[str, Any] | None:
        content = self._read_file(relative_path, context)
        if content is None:
            return None

        chunks = text_to_sentence_chunks(content, sentences_per_chunk=self.source.sentences_per_chunk)
        if not chunks:
            return None

        sections = chunks_to_sections_structured(
            chunks,
            num_sections=self.source.num_sections,
            strategy=self.source.bundle_strategy,
        )
        return {
            "file_name": [relative_path],
            "text": content,
            "chunks": chunks,
            "sections_structured": sections,
            "bundle_id": "",
            "bundle_members": [relative_path],
            "is_multi_doc": False,
        }

    def _hydrate_bundle(
        self,
        relative_members: list[str],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, Any] | None:
        bundle_texts: list[str] = []
        bundle_chunks: list[dict[str, Any]] = []
        bundle_members: list[str] = []
        chunk_id_offset = 0

        for relative_path in relative_members:
            content = self._read_file(relative_path, context)
            if content is None:
                continue
            bundle_members.append(relative_path)
            bundle_texts.append(content)
            doc_chunks = text_to_sentence_chunks(
                content,
                sentences_per_chunk=self.source.sentences_per_chunk,
                doc_id=relative_path,
                doc_path=str(context.root_path / relative_path),
                chunk_id_offset=chunk_id_offset,
            )
            bundle_chunks.extend(doc_chunks)
            chunk_id_offset += len(doc_chunks)

        if not bundle_chunks:
            return None

        combined_text = "\n\n=== Document Boundary ===\n\n".join(bundle_texts)
        sections = chunks_to_sections_structured(
            bundle_chunks,
            num_sections=self.source.num_sections,
            strategy=self.source.bundle_strategy,
        )
        return {
            "file_name": bundle_members,
            "text": combined_text,
            "chunks": bundle_chunks,
            "sections_structured": sections,
            "bundle_id": build_bundle_id(bundle_members),
            "bundle_members": bundle_members,
            "is_multi_doc": True,
        }
