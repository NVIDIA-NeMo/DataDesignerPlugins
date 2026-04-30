# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`DocumentChunkerSeedReader`."""

from pathlib import Path

import pytest
from data_designer.engine.resources.seed_reader import SeedReaderError
from data_designer.engine.secret_resolver import PlaintextResolver

from data_designer_retrieval_sdg.seed_reader import DocumentChunkerSeedReader
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


def _attached_reader(source: DocumentChunkerSeedSource) -> DocumentChunkerSeedReader:
    reader = DocumentChunkerSeedReader()
    reader.attach(source, PlaintextResolver())
    return reader


def _write_sample_files(root: Path) -> None:
    (root / "a.txt").write_text("First doc. Has two sentences.")
    (root / "b.txt").write_text("Second doc. Has three sentences. Done.")
    (root / "skip.bin").write_text("ignored")
    nested = root / "nested"
    nested.mkdir()
    (nested / "c.md").write_text("Nested doc content. Another sentence.")


def test_single_doc_manifest_and_hydration(tmp_path: Path) -> None:
    _write_sample_files(tmp_path)
    source = DocumentChunkerSeedSource(
        path=str(tmp_path),
        file_extensions=[".txt", ".md"],
        sentences_per_chunk=1,
    )
    reader = _attached_reader(source)

    assert reader.get_seed_dataset_size() == 3

    output_df = reader._get_output_dataframe()
    assert sorted(output_df.columns) == sorted(DocumentChunkerSeedReader.output_columns)
    assert len(output_df) == 3

    first = output_df.iloc[0].to_dict()
    assert first["is_multi_doc"] is False
    assert isinstance(first["file_name"], list)
    assert len(first["file_name"]) == 1
    assert first["bundle_members"] == first["file_name"]
    assert first["bundle_id"] == ""
    assert first["chunks"], "expected non-empty chunk list"


def test_extension_filtering(tmp_path: Path) -> None:
    _write_sample_files(tmp_path)
    source = DocumentChunkerSeedSource(
        path=str(tmp_path),
        file_extensions=[".md"],
    )
    reader = _attached_reader(source)
    assert reader.get_seed_dataset_size() == 1


def test_min_text_length_drops_short_files(tmp_path: Path) -> None:
    (tmp_path / "tiny.txt").write_text("hi.")
    (tmp_path / "long.txt").write_text("This is a much longer document. It has many sentences. Good.")
    source = DocumentChunkerSeedSource(
        path=str(tmp_path),
        file_extensions=[".txt"],
        min_text_length=20,
    )
    reader = _attached_reader(source)
    output_df = reader._get_output_dataframe()
    assert len(output_df) == 1
    assert output_df.iloc[0]["file_name"] == ["long.txt"]


def test_num_files_caps_manifest(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"d{i}.txt").write_text(f"Content {i}. More text.")
    source = DocumentChunkerSeedSource(
        path=str(tmp_path),
        file_extensions=[".txt"],
        num_files=2,
    )
    reader = _attached_reader(source)
    assert reader.get_seed_dataset_size() == 2


def test_no_matching_files_raises(tmp_path: Path) -> None:
    (tmp_path / "ignored.bin").write_text("x")
    source = DocumentChunkerSeedSource(
        path=str(tmp_path),
        file_extensions=[".txt"],
    )
    reader = _attached_reader(source)
    with pytest.raises(SeedReaderError):
        reader.get_seed_dataset_size()


def test_multi_doc_bundles(tmp_path: Path) -> None:
    for i in range(4):
        (tmp_path / f"d{i}.txt").write_text(f"Doc {i}. Sentence two.")
    source = DocumentChunkerSeedSource(
        path=str(tmp_path),
        file_extensions=[".txt"],
        multi_doc=True,
        bundle_size=2,
    )
    reader = _attached_reader(source)
    output_df = reader._get_output_dataframe()

    assert len(output_df) == 2
    for _, row in output_df.iterrows():
        assert row["is_multi_doc"] is True
        assert len(row["bundle_members"]) == 2
        assert row["bundle_id"], "multi-doc rows must carry a non-empty bundle_id"
        assert "=== Document Boundary ===" in row["text"]
