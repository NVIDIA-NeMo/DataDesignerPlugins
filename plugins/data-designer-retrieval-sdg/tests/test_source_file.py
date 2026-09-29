# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical source-file contracts for the public generation runner."""

import json
from pathlib import Path

import data_designer.config as dd
import pytest
from data_designer.interface import DataDesigner
from data_designer.interface.errors import DataDesignerGenerationError

from data_designer_retrieval_sdg import (
    GenerationRunConfig,
    RetrievalSourcesFile,
    load_generation_config,
    load_retrieval_sources,
)
from data_designer_retrieval_sdg.pipeline import build_qa_generation_pipeline
from data_designer_retrieval_sdg.retrieval.source_file import (
    snapshot_retrieval_sources,
    validate_generated_source_coverage,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


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


def test_generation_config_loads_canonical_source_without_chunker_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"seed_source": {"seed_type": "retrieval-sources", "path": "sources.jsonl"}}))
    loaded = load_generation_config(config_file)
    assert loaded.config.seed_source == RetrievalSourcesFile(path=Path("sources.jsonl"))
    roundtrip = GenerationRunConfig.model_validate_json(loaded.config.model_dump_json())
    assert roundtrip.seed_source == loaded.config.seed_source
    overridden = load_generation_config(config_file, cli_overrides={"seed_source": {"path": "other.jsonl"}})
    assert overridden.config.seed_source == RetrievalSourcesFile(path=Path("other.jsonl"))


def test_legacy_generation_config_keeps_chunker_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "legacy.json"
    config_file.write_text(json.dumps({"seed_source": {"path": "documents"}}))
    loaded = load_generation_config(config_file)
    assert isinstance(loaded.config.seed_source, DocumentChunkerSeedSource)
    assert loaded.config.seed_source.min_text_length == 50


def test_snapshot_binds_native_fingerprint_to_text_and_image_bytes(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"first image")
    source = tmp_path / "sources.jsonl"
    row = {"unit_id": "p1", "document_id": "doc", "text": "first text", "images": ["page.png"]}
    source.write_text(json.dumps(row))
    artifacts = tmp_path / "artifacts"
    original = snapshot_retrieval_sources(source, artifacts)
    assert snapshot_retrieval_sources(source, artifacts) == original
    original_rows = [json.loads(line) for line in original.read_text().splitlines()]
    copied_image = Path(original_rows[0]["images"][0])
    first_config = build_qa_generation_pipeline(dd.LocalFileSeedSource(path=str(original))).build()
    image.write_bytes(b"second image")
    changed_image = snapshot_retrieval_sources(source, artifacts)
    second_config = build_qa_generation_pipeline(dd.LocalFileSeedSource(path=str(changed_image))).build()
    assert first_config.fingerprint() != second_config.fingerprint()
    assert copied_image.read_bytes() == b"first image"
    row["text"] = "second text"
    source.write_text(json.dumps(row))
    changed_text = snapshot_retrieval_sources(source, artifacts)
    third_config = build_qa_generation_pipeline(dd.LocalFileSeedSource(path=str(changed_text))).build()
    assert second_config.fingerprint() != third_config.fingerprint()
    assert json.loads(original.read_text())["text"] == "first text"


def test_changed_cached_snapshot_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "sources.jsonl"
    source.write_text('{"unit_id":"p1","document_id":"doc","text":"source"}')
    snapshot = snapshot_retrieval_sources(source, tmp_path / "artifacts")
    snapshot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="snapshot has changed"):
        snapshot_retrieval_sources(source, tmp_path / "artifacts")
    assert snapshot.read_bytes() == b"changed"


@pytest.mark.parametrize("failure", ["missing", "duplicate", "reordered", "text", "language", "extra"])
def test_generated_source_coverage_rejects_source_drift(tmp_path: Path, failure: str) -> None:
    source = tmp_path / "sources.jsonl"
    rows = [{"unit_id": f"p{i}", "document_id": "doc", "text": f"source {i}"} for i in range(2)]
    source.write_text("\n".join(map(json.dumps, rows)))
    snapshot = snapshot_retrieval_sources(source, tmp_path / "artifacts")
    generated = tmp_path / "generated.jsonl"
    records = [json.loads(line) for line in snapshot.read_text().splitlines()]
    generated.write_text("\n".join(map(json.dumps, records)) + "\n")
    validate_generated_source_coverage(snapshot, generated, 2)
    if failure == "missing":
        records.pop()
    elif failure == "duplicate":
        records[1] = records[0]
    elif failure == "reordered":
        records.reverse()
    elif failure == "text":
        records[0]["retrieval_units"][0]["text"] = "altered"
    elif failure == "language":
        records[0]["language"] = "altered"
    else:
        records.append(records[0])
    generated.write_text("\n".join(map(json.dumps, records)) + "\n")
    with pytest.raises(ValueError, match="Generated source"):
        validate_generated_source_coverage(snapshot, generated, 2)


def native_identity_builder(snapshot: Path) -> dd.DataDesignerConfigBuilder:
    """Build a local-only workflow exercising native seed reading and resume."""
    builder = dd.DataDesignerConfigBuilder()
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(snapshot)))
    builder.add_column(dd.ExpressionColumnConfig(name="copied_id", expr="{{ source_id }}"))
    return builder


@pytest.mark.parametrize("changed_field", ["text", "image"])
def test_actual_native_resume_rejects_changed_input(tmp_path: Path, changed_field: str) -> None:
    source = tmp_path / "sources.jsonl"
    image = tmp_path / "image.png"
    image.write_bytes(b"first image")
    row = {"unit_id": "001", "document_id": "007", "text": "first text", "images": ["image.png"]}
    source.write_text(json.dumps(row))
    root = tmp_path / "artifacts"
    snapshot = snapshot_retrieval_sources(source, root)
    designer = DataDesigner(artifact_path=root)
    designer.set_run_config(dd.RunConfig(otel_metrics_port=None))
    first = designer.create(native_identity_builder(snapshot), num_records=1, dataset_name="identity")
    assert first.load_dataset().iloc[0]["source_id"] == "001"
    first.export(tmp_path / "generated.jsonl", format="jsonl")
    validate_generated_source_coverage(snapshot, tmp_path / "generated.jsonl", 1)
    resumed = designer.create(
        native_identity_builder(snapshot), num_records=1, dataset_name="identity", resume="always"
    )
    assert resumed.count_records() == 1
    if changed_field == "text":
        row["text"] = "changed text"
        source.write_text(json.dumps(row))
    else:
        image.write_bytes(b"changed image")
    changed = snapshot_retrieval_sources(source, root)
    with pytest.raises(DataDesignerGenerationError, match="Cannot resume"):
        designer.create(native_identity_builder(changed), num_records=1, dataset_name="identity", resume="always")
    fresh = designer.create(
        native_identity_builder(changed), num_records=1, dataset_name="identity", resume="if_possible"
    )
    assert fresh.artifact_storage.resolved_dataset_name != first.artifact_storage.resolved_dataset_name
    assert first.load_dataset().iloc[0]["text"] == "first text"


def test_changed_cached_image_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "sources.jsonl"
    image = tmp_path / "image.png"
    image.write_bytes(b"original")
    source.write_text('{"unit_id":"a","document_id":"d","images":["image.png"]}')
    snapshot = snapshot_retrieval_sources(source, tmp_path / "artifacts")
    copied = Path(json.loads(snapshot.read_text())["images"][0])
    copied.write_bytes(b"modified cached image")
    with pytest.raises(ValueError, match="snapshot has changed"):
        snapshot_retrieval_sources(source, tmp_path / "artifacts")
    assert copied.read_bytes() == b"modified cached image"
