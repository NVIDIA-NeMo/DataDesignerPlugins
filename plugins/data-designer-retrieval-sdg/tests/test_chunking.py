# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the chunking, section, and bundling helpers."""

from pathlib import Path

import pytest

from data_designer_retrieval_sdg.chunking import (
    build_bundle_id,
    build_bundles,
    chunks_to_sections_structured,
    text_to_sentence_chunks,
)


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


def test_chunks_to_sections_sequential() -> None:
    chunks = [{"text": f"chunk {i}", "chunk_id": i} for i in range(1, 7)]
    sections = chunks_to_sections_structured(chunks, num_sections=2, strategy="sequential")
    assert len(sections) == 2
    assert "Section 1" in sections[0]
    assert "Section 2" in sections[1]


def test_chunks_to_sections_empty() -> None:
    assert chunks_to_sections_structured([], num_sections=2) == []


def test_chunks_to_sections_doc_balanced_falls_back_to_sequential_for_single_doc() -> None:
    chunks = [{"text": f"chunk {i}", "chunk_id": i, "doc_id": "only"} for i in range(1, 5)]
    sections = chunks_to_sections_structured(chunks, num_sections=2, strategy="doc_balanced")
    assert len(sections) == 2


def test_chunks_to_sections_doc_balanced_multi_doc() -> None:
    chunks = [
        {"text": "a1", "chunk_id": 1, "doc_id": "a"},
        {"text": "a2", "chunk_id": 2, "doc_id": "a"},
        {"text": "b1", "chunk_id": 3, "doc_id": "b"},
        {"text": "b2", "chunk_id": 4, "doc_id": "b"},
    ]
    sections = chunks_to_sections_structured(chunks, num_sections=2, strategy="doc_balanced")
    assert len(sections) == 2
    for section in sections:
        assert "[Doc: a]" in section
        assert "[Doc: b]" in section


def test_chunks_to_sections_interleaved_multi_doc() -> None:
    chunks = [
        {"text": "a1", "chunk_id": 1, "doc_id": "a"},
        {"text": "a2", "chunk_id": 2, "doc_id": "a"},
        {"text": "b1", "chunk_id": 3, "doc_id": "b"},
    ]
    sections = chunks_to_sections_structured(chunks, num_sections=1, strategy="interleaved")
    assert len(sections) == 1
    assert "[Doc: a]" in sections[0]
    assert "[Doc: b]" in sections[0]


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


def test_build_bundles_empty() -> None:
    assert build_bundles([], bundle_size=2, max_docs_per_bundle=3) == []


def test_build_bundle_id_deterministic() -> None:
    a = build_bundle_id(["a.txt", "b.txt"])
    b = build_bundle_id(["b.txt", "a.txt"])
    assert a == b


def test_build_bundle_id_empty() -> None:
    assert build_bundle_id([]) == ""
