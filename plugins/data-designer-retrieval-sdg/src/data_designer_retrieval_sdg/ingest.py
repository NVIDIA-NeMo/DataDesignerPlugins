# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text ingestion, chunking, and section-building utilities.

This module handles loading text files from a directory, chunking them by
sentence boundaries, and organising chunks into sections using various
strategies (sequential, doc-balanced, interleaved).  It supports both
single-document and multi-document (bundled) modes.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

# ---------------------------------------------------------------------------
# File-matching helpers
# ---------------------------------------------------------------------------


def is_traditional_extension(suffix: str) -> bool:
    """Check whether *suffix* looks like a real file extension.

    Traditional extensions are short (1-10 chars), start with a period, and
    contain only alphanumeric characters.  For example ``.txt``, ``.md``,
    ``.json`` are traditional, whereas
    ``.com_publication_2001-08_user-programmable`` is not.

    Args:
        suffix: The file suffix (including leading ``'.'``).

    Returns:
        ``True`` when the suffix matches the traditional pattern.
    """
    if not suffix or not suffix.startswith("."):
        return False
    ext_part = suffix[1:]
    return len(ext_part) <= 10 and ext_part.replace("_", "").isalnum()


def file_matches_extensions(file_path: Path, file_extensions: list[str]) -> bool:
    """Decide whether *file_path* has one of the allowed extensions.

    Files whose suffix is not *traditional* (see
    :func:`is_traditional_extension`) are treated as having no extension
    and matched against ``""`` in *file_extensions*.

    Args:
        file_path: Path to the file.
        file_extensions: Allowed extensions, e.g. ``[".txt", ".md", ""]``.

    Returns:
        ``True`` when the file matches.
    """
    suffix = file_path.suffix.lower()
    if is_traditional_extension(suffix):
        return suffix in file_extensions
    return "" in file_extensions


# ---------------------------------------------------------------------------
# Multi-document bundling helpers
# ---------------------------------------------------------------------------


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
    import json

    import yaml

    if not manifest_path:
        return []

    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Warning: Unable to read multi_doc_manifest at {manifest_path}: {exc}")
        return []

    data = None
    try:
        data = json.loads(manifest_text)
    except json.JSONDecodeError:
        try:
            data = yaml.safe_load(manifest_text)
        except Exception as exc:
            print(f"Warning: Failed to parse multi_doc_manifest: {exc}")
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
        print("Warning: multi_doc_manifest must be a list or dict with 'bundles'")

    return bundles


def build_bundle_id(bundle_members: list[str]) -> str:
    """Generate a unique bundle ID from member paths.

    Args:
        bundle_members: List of file paths in the bundle.

    Returns:
        MD5 hex digest of sorted, normalised paths.
    """
    if not bundle_members:
        return ""
    normalized = "||".join(sorted(str(Path(member).resolve()) for member in bundle_members))
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
    sequentially according to *bundle_size*.

    Args:
        file_paths: All candidate file paths.
        bundle_size: Documents per automatic bundle.
        max_docs_per_bundle: Hard cap on bundle size.
        manifest_bundles: Pre-defined bundles from a manifest file.
        input_dir: Root directory for resolving relative manifest paths.

    Returns:
        List of bundles, each a list of resolved ``Path`` objects.

    Raises:
        ValueError: If any bundle exceeds *max_docs_per_bundle*.
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


# ---------------------------------------------------------------------------
# Section-building strategies
# ---------------------------------------------------------------------------


def group_chunks_by_doc(chunks: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    """Group chunks by their ``doc_id`` field.

    Args:
        chunks: Chunk dicts, each optionally containing ``'doc_id'``.

    Returns:
        Mapping from ``doc_id`` to ``(global_index, chunk)`` pairs.
    """
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        doc_id = chunk.get("doc_id", "default")
        grouped[doc_id].append((idx, chunk))
    return dict(grouped)


def format_section_chunks(indexed_chunks: list[tuple[int, dict]], section_number: int) -> str:
    """Render a list of indexed chunks into a section string.

    Args:
        indexed_chunks: ``(global_index, chunk)`` tuples.
        section_number: Section ordinal for the header.

    Returns:
        Formatted section text, or ``""`` if no content.
    """
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
    """Split chunks sequentially into *num_sections* sections.

    Args:
        chunks: Chunk dicts in document order.
        num_sections: How many sections to produce.

    Returns:
        List of formatted section strings.
    """
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
    """Split chunks so each section has proportional doc representation.

    Falls back to sequential when there is only one document.

    Args:
        chunks: Chunk dicts with ``'doc_id'`` fields.
        num_sections: How many sections to produce.

    Returns:
        List of formatted section strings.
    """
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
    """Split chunks with round-robin interleaving across documents.

    Falls back to sequential when there is only one document.

    Args:
        chunks: Chunk dicts with ``'doc_id'`` fields.
        num_sections: How many sections to produce.

    Returns:
        List of formatted section strings.
    """
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
    """Split chunks into sections using the specified strategy.

    Args:
        chunks: Chunk dicts.
        num_sections: How many sections to produce.
        strategy: ``"sequential"``, ``"doc_balanced"``, or ``"interleaved"``.

    Returns:
        List of formatted section strings.
    """
    if strategy == "doc_balanced":
        return chunks_to_sections_doc_balanced(chunks, num_sections)
    if strategy == "interleaved":
        return chunks_to_sections_interleaved(chunks, num_sections)
    return chunks_to_sections_sequential(chunks, num_sections)


# ---------------------------------------------------------------------------
# Sentence chunking
# ---------------------------------------------------------------------------


def _ensure_nltk_punkt() -> None:
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
    """Chunk *text* into groups of sentences with metadata.

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
    _ensure_nltk_punkt()

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


# ---------------------------------------------------------------------------
# Top-level directory loader
# ---------------------------------------------------------------------------


def load_text_files_from_directory(
    input_dir: Path,
    file_extensions: list[str] | None = None,
    min_text_length: int = 0,
    sentences_per_chunk: int = 5,
    num_sections: int = 1,
    num_files: int | None = None,
    multi_doc: bool = False,
    bundle_size: int = 2,
    bundle_strategy: Literal["sequential", "doc_balanced", "interleaved"] = "sequential",
    max_docs_per_bundle: int = 3,
    multi_doc_manifest: Path | None = None,
) -> pd.DataFrame:
    """Load text files from a directory into a seed DataFrame.

    Supports single-document mode (one row per file) and multi-document mode
    (files grouped into bundles, one row per bundle).

    Args:
        input_dir: Root directory containing text files.
        file_extensions: Allowed extensions (default ``[".txt", ".md", ".text", ""]``).
        min_text_length: Minimum character count to include a document.
        sentences_per_chunk: Sentences per chunk.
        num_sections: Sections to split chunks into.
        num_files: Cap on the number of files to process.
        multi_doc: Enable multi-document bundling.
        bundle_size: Documents per automatic bundle.
        bundle_strategy: Section-building strategy.
        max_docs_per_bundle: Hard cap on bundle size.
        multi_doc_manifest: Path to a manifest defining explicit bundles.

    Returns:
        DataFrame with columns ``file_name``, ``text``, ``chunks``,
        ``sections_structured``, and (when multi-doc) ``bundle_id``,
        ``bundle_members``, ``is_multi_doc``.

    Raises:
        ValueError: If no text files or valid documents are found.
    """
    if file_extensions is None:
        file_extensions = [".txt", ".md", ".text", ""]

    all_file_paths: list[Path] = []
    for file_path in input_dir.rglob("*"):
        if num_files is not None and len(all_file_paths) >= num_files:
            break
        if file_path.is_file() and file_matches_extensions(file_path, file_extensions):
            try:
                content = file_path.read_text(encoding="utf-8")
                if min_text_length > 0 and len(content) < min_text_length:
                    continue
                all_file_paths.append(file_path)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue

    if not all_file_paths:
        raise ValueError(f"No text files found in {input_dir} with extensions {file_extensions}")

    resolved_input_dir = input_dir.resolve()
    documents: list[dict] = []

    if multi_doc:
        documents = _load_multi_doc(
            all_file_paths,
            resolved_input_dir,
            sentences_per_chunk,
            num_sections,
            bundle_size,
            bundle_strategy,
            max_docs_per_bundle,
            multi_doc_manifest,
        )
    else:
        documents = _load_single_doc(
            all_file_paths,
            input_dir,
            sentences_per_chunk,
            num_sections,
            bundle_strategy,
        )

    if not documents:
        raise ValueError(f"No valid documents created from {input_dir}")

    df = pd.DataFrame(documents)
    _print_load_stats(df, all_file_paths, multi_doc, min_text_length, bundle_strategy)
    return df


# ---------------------------------------------------------------------------
# Internal loader helpers
# ---------------------------------------------------------------------------


def _load_single_doc(
    file_paths: list[Path],
    input_dir: Path,
    sentences_per_chunk: int,
    num_sections: int,
    bundle_strategy: Literal["sequential", "doc_balanced", "interleaved"],
) -> list[dict]:
    """Build one row per file."""
    documents: list[dict] = []
    for file_path in file_paths:
        relative_path = file_path.relative_to(input_dir)
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read {relative_path}: {e}")
            continue

        chunks = text_to_sentence_chunks(content, sentences_per_chunk=sentences_per_chunk)
        sections_structured = chunks_to_sections_structured(chunks, num_sections=num_sections, strategy=bundle_strategy)
        documents.append(
            {
                "file_name": [str(relative_path)],
                "text": content,
                "chunks": chunks,
                "sections_structured": sections_structured,
                "bundle_id": "",
                "bundle_members": [str(relative_path)],
                "is_multi_doc": False,
            }
        )
    return documents


def _load_multi_doc(
    file_paths: list[Path],
    resolved_input_dir: Path,
    sentences_per_chunk: int,
    num_sections: int,
    bundle_size: int,
    bundle_strategy: Literal["sequential", "doc_balanced", "interleaved"],
    max_docs_per_bundle: int,
    multi_doc_manifest: Path | None,
) -> list[dict]:
    """Build one row per bundle."""
    manifest_bundles = load_multi_doc_manifest(multi_doc_manifest)
    bundles = build_bundles(
        file_paths,
        bundle_size=bundle_size,
        max_docs_per_bundle=max_docs_per_bundle,
        manifest_bundles=manifest_bundles,
        input_dir=resolved_input_dir,
    )

    print(f"Multi-doc mode: Created {len(bundles)} bundles from {len(file_paths)} files")
    documents: list[dict] = []

    for bundle in bundles:
        bundle_texts: list[str] = []
        bundle_chunks: list[dict] = []
        bundle_members: list[str] = []
        chunk_id_offset = 0

        for file_path in bundle:
            relative_path = file_path.relative_to(resolved_input_dir)
            doc_id = str(relative_path)
            bundle_members.append(doc_id)

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue

            bundle_texts.append(content)
            doc_chunks = text_to_sentence_chunks(
                content,
                sentences_per_chunk=sentences_per_chunk,
                doc_id=doc_id,
                doc_path=str(file_path),
                chunk_id_offset=chunk_id_offset,
            )
            bundle_chunks.extend(doc_chunks)
            chunk_id_offset += len(doc_chunks)

        if not bundle_chunks:
            continue

        combined_text = "\n\n=== Document Boundary ===\n\n".join(bundle_texts)
        sections_structured = chunks_to_sections_structured(
            bundle_chunks, num_sections=num_sections, strategy=bundle_strategy
        )
        bid = build_bundle_id(bundle_members)

        documents.append(
            {
                "file_name": bundle_members,
                "text": combined_text,
                "chunks": bundle_chunks,
                "sections_structured": sections_structured,
                "bundle_id": bid,
                "bundle_members": bundle_members,
                "is_multi_doc": True,
            }
        )

    return documents


def _print_load_stats(
    df: pd.DataFrame,
    all_file_paths: list[Path],
    multi_doc: bool,
    min_text_length: int,
    bundle_strategy: str,
) -> None:
    """Print statistics about the loaded data."""
    row_type = "bundle" if multi_doc else "document"
    if multi_doc:
        avg_docs = sum(len(m) for m in df["bundle_members"]) / len(df) if len(df) > 0 else 0
        print(f"Created {len(df)} bundles from {len(all_file_paths)} files")
        print(f"Average documents per bundle: {avg_docs:.1f}")
    else:
        print(f"Loaded {len(df)} text files from directory")

    if min_text_length > 0:
        print(f"Filtered to documents with at least {min_text_length} characters")

    total_chunks = sum(len(c) for c in df["chunks"])
    avg_chunks = total_chunks / len(df) if len(df) > 0 else 0
    print(f"Created {total_chunks} total chunks ({avg_chunks:.1f} chunks per {row_type})")

    total_sections = sum(len(s) for s in df["sections_structured"])
    avg_sections = total_sections / len(df) if len(df) > 0 else 0
    avg_chunks_per_section = total_chunks / total_sections if total_sections > 0 else 0
    print(
        f"Organized into {total_sections} sections "
        f"({avg_sections:.1f} sections per {row_type}, "
        f"{avg_chunks_per_section:.1f} chunks per section)"
    )
    print(f"Bundle strategy: {bundle_strategy}")
