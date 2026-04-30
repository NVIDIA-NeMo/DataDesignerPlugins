# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text chunking, section-building, and multi-document bundling helpers.

These pure utilities are shared by the document-chunker seed reader and
exposed for direct use in tests.  They contain no DataDesigner-specific
state: file IO is performed by the seed reader, while this module focuses
on shaping text into chunks/sections and grouping files into bundles.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

import nltk
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


def load_multi_doc_manifest(manifest_path: Path | None) -> list[list[str]]:
    """Load a multi-doc manifest file.

    Supports JSON or YAML format::

        [["doc1.txt", "doc2.txt"], ["doc3.txt"]]
        {"bundles": [{"docs": ["doc1.txt", "doc2.txt"]}]}

    Args:
        manifest_path: Path to the manifest file, or ``None``.

    Returns:
        List of bundles, each a list of file-path strings.
    """
    import yaml

    if not manifest_path:
        return []

    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Unable to read multi_doc_manifest at %s: %s", manifest_path, exc)
        return []

    data = None
    try:
        data = json.loads(manifest_text)
    except json.JSONDecodeError:
        try:
            data = yaml.safe_load(manifest_text)
        except yaml.YAMLError as exc:
            logger.warning("Failed to parse multi_doc_manifest: %s", exc)
            return []

    if isinstance(data, dict) and "bundles" in data:
        data = data["bundles"]

    bundles: list[list[str]] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "docs" in entry:
                docs = entry["docs"]
            elif isinstance(entry, list):
                docs = entry
            else:
                docs = []
            clean_docs = [str(doc) for doc in docs if doc]
            if clean_docs:
                bundles.append(clean_docs)
    else:
        logger.warning("multi_doc_manifest must be a list or dict with 'bundles'")

    return bundles


def build_bundle_id(bundle_members: list[str]) -> str:
    """Generate a stable bundle ID from member identifiers.

    Args:
        bundle_members: List of member paths (relative or absolute).

    Returns:
        MD5 hex digest of sorted, normalised members.
    """
    if not bundle_members:
        return ""
    normalized = "||".join(sorted(str(member) for member in bundle_members))
    return hashlib.md5(normalized.encode()).hexdigest()


def build_bundles(
    file_paths: list[Path],
    bundle_size: int = 2,
    max_docs_per_bundle: int = 3,
    manifest_bundles: list[list[str]] | None = None,
    input_dir: Path | None = None,
) -> list[list[Path]]:
    """Group file paths into document bundles.

    Manifest-defined bundles take priority.  Remaining documents are grouped
    sequentially according to ``bundle_size``.

    Args:
        file_paths: All candidate file paths.
        bundle_size: Documents per automatic bundle.
        max_docs_per_bundle: Hard cap on bundle size.
        manifest_bundles: Pre-defined bundles from a manifest file.
        input_dir: Root directory for resolving relative manifest paths.

    Returns:
        List of bundles, each a list of resolved ``Path`` objects.

    Raises:
        ValueError: If any bundle exceeds ``max_docs_per_bundle``.
    """
    if not file_paths:
        return []

    resolved_paths = [path.resolve() for path in file_paths]
    seen: set[Path] = set()
    bundles: list[list[Path]] = []

    if manifest_bundles:
        for entry in manifest_bundles:
            resolved_bundle: list[Path] = []
            for raw_doc in entry:
                candidate = Path(raw_doc)
                if not candidate.is_absolute() and input_dir:
                    candidate = (input_dir / raw_doc).resolve()
                candidate = candidate.resolve()
                if candidate in resolved_paths and candidate not in seen:
                    resolved_bundle.append(candidate)
                    seen.add(candidate)
            if resolved_bundle:
                bundles.append(resolved_bundle)

    remaining = [p for p in resolved_paths if p not in seen]
    for start in range(0, len(remaining), bundle_size):
        bundle = remaining[start : start + bundle_size]
        if bundle:
            bundles.append(bundle)

    for i, bundle in enumerate(bundles):
        if len(bundle) > max_docs_per_bundle:
            raise ValueError(
                f"Bundle {i} has {len(bundle)} documents, which exceeds "
                f"max_docs_per_bundle={max_docs_per_bundle}. "
                f"Either reduce the bundle size in your manifest or increase max_docs_per_bundle."
            )

    return [b for b in bundles if b]


def group_chunks_by_doc(chunks: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    """Group chunks by their ``doc_id`` field."""
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        doc_id = chunk.get("doc_id", "default")
        grouped[doc_id].append((idx, chunk))
    return dict(grouped)


def format_section_chunks(indexed_chunks: list[tuple[int, dict]], section_number: int) -> str:
    """Render a list of indexed chunks into a section string."""
    section_lines: list[str] = []
    for _, chunk in indexed_chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        segment_id = chunk.get("chunk_id", 1)
        doc_id = chunk.get("doc_id", "")
        start_time = "00:00:00"
        end_time = "00:00:00"
        if doc_id:
            segment_info = f"Segment {segment_id} [Doc: {doc_id}] ({start_time} - {end_time}): {text}"
        else:
            segment_info = f"Segment {segment_id} ({start_time} - {end_time}): {text}"
        section_lines.append(segment_info)

    if section_lines:
        return f"=== Section {section_number} ===\n" + "\n".join(section_lines)
    return ""


def chunks_to_sections_sequential(chunks: list[dict], num_sections: int = 1) -> list[str]:
    """Split chunks sequentially into ``num_sections`` formatted sections."""
    total = len(chunks)
    if total == 0:
        return []

    section_size = max(1, total // num_sections)
    formatted_sections: list[str] = []

    for i in range(num_sections):
        start_idx = i * section_size
        end_idx = (i + 1) * section_size if i < num_sections - 1 else total
        indexed_chunks = [(j, chunks[j]) for j in range(start_idx, end_idx)]
        section_text = format_section_chunks(indexed_chunks, i + 1)
        if section_text:
            formatted_sections.append(section_text)

    return formatted_sections


def chunks_to_sections_doc_balanced(chunks: list[dict], num_sections: int = 1) -> list[str]:
    """Split chunks so each section has proportional doc representation."""
    if not chunks:
        return []

    grouped = group_chunks_by_doc(chunks)
    if len(grouped) <= 1:
        return chunks_to_sections_sequential(chunks, num_sections)

    chunk_sizes = {doc_id: max(1, math.ceil(len(entries) / num_sections)) for doc_id, entries in grouped.items()}

    sections: list[list[tuple[int, dict]]] = []
    for part_idx in range(num_sections):
        part_entries: list[tuple[int, dict]] = []
        for doc_id, entries in grouped.items():
            chunk_size = chunk_sizes[doc_id]
            start = part_idx * chunk_size
            end = min(len(entries), start + chunk_size)
            if start < len(entries):
                part_entries.extend(entries[start:end])
        if part_entries:
            sections.append(part_entries)

    formatted_sections: list[str] = []
    for i, indexed_chunks in enumerate(sections):
        section_text = format_section_chunks(indexed_chunks, i + 1)
        if section_text:
            formatted_sections.append(section_text)

    return formatted_sections


def chunks_to_sections_interleaved(chunks: list[dict], num_sections: int = 1) -> list[str]:
    """Split chunks with round-robin interleaving across documents."""
    if not chunks:
        return []

    grouped = group_chunks_by_doc(chunks)
    if len(grouped) <= 1:
        return chunks_to_sections_sequential(chunks, num_sections)

    doc_iterators = {doc_id: deque(entries) for doc_id, entries in grouped.items()}
    doc_order = list(grouped.keys())
    interleaved: list[tuple[int, dict]] = []

    while True:
        added = False
        for doc_id in doc_order:
            doc_queue = doc_iterators[doc_id]
            if doc_queue:
                interleaved.append(doc_queue.popleft())
                added = True
        if not added:
            break

    if not interleaved:
        return []

    total = len(interleaved)
    section_size = max(1, total // num_sections)
    formatted_sections: list[str] = []

    for i in range(num_sections):
        start_idx = i * section_size
        end_idx = (i + 1) * section_size if i < num_sections - 1 else total
        indexed_chunks = interleaved[start_idx:end_idx]
        section_text = format_section_chunks(indexed_chunks, i + 1)
        if section_text:
            formatted_sections.append(section_text)

    return formatted_sections


def chunks_to_sections_structured(
    chunks: list[dict],
    num_sections: int = 1,
    strategy: Literal["sequential", "doc_balanced", "interleaved"] = "sequential",
) -> list[str]:
    """Split chunks into sections using the specified strategy."""
    if strategy == "doc_balanced":
        return chunks_to_sections_doc_balanced(chunks, num_sections)
    if strategy == "interleaved":
        return chunks_to_sections_interleaved(chunks, num_sections)
    return chunks_to_sections_sequential(chunks, num_sections)


def ensure_nltk_punkt() -> None:
    """Download NLTK punkt tokeniser data if not already present."""
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


def text_to_sentence_chunks(
    text: str,
    sentences_per_chunk: int = 5,
    doc_id: str | None = None,
    doc_path: str | None = None,
    chunk_id_offset: int = 0,
) -> list[dict]:
    """Chunk ``text`` into groups of sentences with metadata.

    Args:
        text: Input text to chunk.
        sentences_per_chunk: Sentences per chunk.
        doc_id: Optional document identifier for multi-doc bundles.
        doc_path: Optional document path for multi-doc bundles.
        chunk_id_offset: Offset for global chunk IDs when aggregating.

    Returns:
        List of chunk dicts with keys ``text``, ``start``, ``end``,
        ``sentence_count``, ``word_count``, ``chunk_id``,
        ``doc_chunk_index``, and optionally ``doc_id`` / ``doc_path``.
    """
    ensure_nltk_punkt()

    paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    sentences: list[str] = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))

    chunks: list[dict] = []
    word_position = 0
    doc_chunk_index = 0

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[i : i + sentences_per_chunk]
        chunk_text = ". ".join(chunk_sentences)
        if chunk_text and not chunk_text.endswith("."):
            chunk_text += "."

        chunk_words = chunk_text.split()
        start_word_pos = word_position
        end_word_pos = word_position + len(chunk_words)
        word_position = end_word_pos
        doc_chunk_index += 1

        chunk_data: dict = {
            "text": chunk_text,
            "start": start_word_pos,
            "end": end_word_pos,
            "sentence_count": len(chunk_sentences),
            "word_count": len(chunk_words),
            "chunk_id": chunk_id_offset + len(chunks) + 1,
            "doc_chunk_index": doc_chunk_index,
        }

        if doc_id is not None:
            chunk_data["doc_id"] = doc_id
        if doc_path is not None:
            chunk_data["doc_path"] = doc_path

        chunks.append(chunk_data)

    return chunks
