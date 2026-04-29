# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from data_designer_retrieval_sdg.ingest import (
    build_bundle_id,
    build_bundles,
    chunks_to_sections_structured,
    file_matches_extensions,
    is_traditional_extension,
    load_text_files_from_directory,
    text_to_sentence_chunks,
)

# ---------------------------------------------------------------------------
# is_traditional_extension
# ---------------------------------------------------------------------------


def test_traditional_extensions() -> None:
    assert is_traditional_extension(".txt") is True
    assert is_traditional_extension(".md") is True
    assert is_traditional_extension(".json") is True
    assert is_traditional_extension(".mp3") is True


def test_non_traditional_extensions() -> None:
    assert is_traditional_extension("") is False
    assert is_traditional_extension(".com_publication_2001") is False
    assert is_traditional_extension(".averylongextension123") is False


# ---------------------------------------------------------------------------
# file_matches_extensions
# ---------------------------------------------------------------------------


def test_file_matches_extensions_standard() -> None:
    assert file_matches_extensions(Path("doc.txt"), [".txt", ".md"]) is True
    assert file_matches_extensions(Path("doc.py"), [".txt", ".md"]) is False


def test_file_matches_extensions_no_ext() -> None:
    assert file_matches_extensions(Path("README"), [""]) is True


# ---------------------------------------------------------------------------
# text_to_sentence_chunks
# ---------------------------------------------------------------------------


def test_text_to_sentence_chunks_basic() -> None:
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence."
    chunks = text_to_sentence_chunks(text, sentences_per_chunk=3)
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == 1
    assert chunks[1]["chunk_id"] == 2
    assert chunks[0]["sentence_count"] == 3


def test_text_to_sentence_chunks_with_doc_id() -> None:
    chunks = text_to_sentence_chunks("Hello world. Goodbye.", sentences_per_chunk=5, doc_id="doc1")
    assert len(chunks) == 1
    assert chunks[0]["doc_id"] == "doc1"


def test_text_to_sentence_chunks_empty() -> None:
    assert text_to_sentence_chunks("") == []


# ---------------------------------------------------------------------------
# Section strategies
# ---------------------------------------------------------------------------


def test_chunks_to_sections_sequential() -> None:
    chunks = [{"text": f"chunk {i}", "chunk_id": i} for i in range(1, 7)]
    sections = chunks_to_sections_structured(chunks, num_sections=2, strategy="sequential")
    assert len(sections) == 2
    assert "Section 1" in sections[0]
    assert "Section 2" in sections[1]


def test_chunks_to_sections_empty() -> None:
    assert chunks_to_sections_structured([], num_sections=2) == []


# ---------------------------------------------------------------------------
# build_bundles
# ---------------------------------------------------------------------------


def test_build_bundles_sequential(tmp_path: Path) -> None:
    files = [tmp_path / f"f{i}.txt" for i in range(4)]
    for f in files:
        f.write_text("content")
    bundles = build_bundles(files, bundle_size=2, max_docs_per_bundle=3)
    assert len(bundles) == 2
    assert len(bundles[0]) == 2


def test_build_bundles_exceeds_max(tmp_path: Path) -> None:
    files = [tmp_path / f"f{i}.txt" for i in range(4)]
    for f in files:
        f.write_text("content")
    with pytest.raises(ValueError, match="exceeds max_docs_per_bundle"):
        build_bundles(files, bundle_size=4, max_docs_per_bundle=2)


# ---------------------------------------------------------------------------
# build_bundle_id
# ---------------------------------------------------------------------------


def test_build_bundle_id_deterministic() -> None:
    a = build_bundle_id(["a.txt", "b.txt"])
    b = build_bundle_id(["b.txt", "a.txt"])
    assert a == b


def test_build_bundle_id_empty() -> None:
    assert build_bundle_id([]) == ""


# ---------------------------------------------------------------------------
# load_text_files_from_directory
# ---------------------------------------------------------------------------


def test_load_text_files_single_doc(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Hello world. This is a test. Another sentence.")
    (tmp_path / "b.txt").write_text("Foo bar. Baz quux. Something else.")
    df = load_text_files_from_directory(tmp_path, sentences_per_chunk=2)
    assert len(df) == 2
    assert "file_name" in df.columns
    assert "chunks" in df.columns
    assert "sections_structured" in df.columns


def test_load_text_files_no_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No text files found"):
        load_text_files_from_directory(tmp_path)
