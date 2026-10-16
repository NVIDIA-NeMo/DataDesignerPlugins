# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-file contracts for multimodal retrieval generation."""

import json
from pathlib import Path

import pytest

from data_designer_retrieval_sdg import load_generation_config
from data_designer_retrieval_sdg.multimodal.source_file import load_retrieval_sources
from data_designer_retrieval_sdg.text_qa.seed_source import DocumentChunkerSeedSource


def test_canonical_sources_preserve_order_language_and_relative_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image fixture")
    records = [
        {"document_id": "doc", "unit_id": "p1", "text": "Texte source", "language": "fr"},
        {"document_id": "doc", "unit_id": "p2", "images": ["page.png"], "page_number": 2},
        {"document_id": "other", "unit_id": "p3", "text": "Text", "images": [str(image)]},
    ]
    source_file = tmp_path / "sources.jsonl"
    source_file.write_text("\n" + "\n\n".join(map(json.dumps, records)), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    sources = load_retrieval_sources(source_file)
    assert [source.unit_id for source in sources] == ["p1", "p2", "p3"]
    assert sources[0].language == "fr"
    assert sources[0].images == []
    assert sources[1].text == ""
    assert sources[1].page_number == 2
    assert sources[1].images == sources[2].images == [str(image)]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("\n", "contains no records"),
        ("not JSON", "line 1"),
        ('{"unit_id":"a","document_id":"d"}', "line 1"),
        ('{"unit_id":"a","document_id":"d","text":"x","unexpected":1}', "line 1"),
        ('{"unit_id":"a","document_id":"d","text":"x"}\n' * 2, "Duplicate retrieval unit_id"),
    ],
)
def test_invalid_sources_fail(tmp_path: Path, content: str, message: str) -> None:
    source_file = tmp_path / "sources.jsonl"
    source_file.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_retrieval_sources(source_file)


def test_missing_image_fails(tmp_path: Path) -> None:
    source_file = tmp_path / "sources.jsonl"
    source_file.write_text('{"unit_id":"a","document_id":"d","images":["missing.png"]}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Missing image"):
        load_retrieval_sources(source_file)


def test_legacy_generation_config_keeps_chunker_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "legacy.json"
    config_file.write_text(json.dumps({"seed_source": {"path": "documents"}}))
    loaded = load_generation_config(config_file)
    assert isinstance(loaded.config.seed_source, DocumentChunkerSeedSource)
    assert loaded.config.seed_source.min_text_length == 50
