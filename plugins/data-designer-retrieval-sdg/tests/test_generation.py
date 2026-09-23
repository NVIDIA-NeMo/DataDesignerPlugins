# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stable generation Python API."""

import json
from pathlib import Path

import pytest
from data_designer.engine.storage.artifact_storage import ResumeMode

from data_designer_retrieval_sdg import generation
from data_designer_retrieval_sdg.retrieval.source_file import RetrievalSourcesFile
from data_designer_retrieval_sdg.run_config import GenerationPipelineConfig, GenerationRunConfig
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


class FakeArtifactStorage:
    """Minimal artifact storage surface returned by Data Designer."""

    def __init__(self, artifact_path: Path) -> None:
        self.base_dataset_path = artifact_path / "retrieval_resolved"
        self.resolved_dataset_name = "retrieval_resolved"


class FakeCreateResult:
    """Capture exported paths from a generation result."""

    def __init__(self, artifact_path: Path) -> None:
        self.artifact_storage = FakeArtifactStorage(artifact_path)
        self.export_calls: list[tuple[Path, str | None]] = []
        self.rows: list[dict] | None = None

    def count_records(self) -> int:
        """Return a count that differs from the requested record count."""
        return len(self.rows) if self.rows is not None else 2

    def export(self, path: Path, *, format: str | None = None) -> Path:
        self.export_calls.append((path, format))
        rows = self.rows if self.rows is not None else [{}]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def display_sample_record(self) -> None:
        """Provide the preview display surface used by the public API."""


class FakeDataDesigner:
    """Capture the Data Designer create contract used by the public runner."""

    instances: list["FakeDataDesigner"] = []

    def __init__(self, artifact_path: Path, model_providers: object) -> None:
        self.artifact_path = artifact_path
        self.model_providers = model_providers
        self.run_config = None
        self.create_calls: list[dict[str, object]] = []
        self.result = FakeCreateResult(artifact_path)
        self.instances.append(self)

    def set_run_config(self, run_config: object) -> None:
        self.run_config = run_config

    def create(
        self,
        config_builder: object,
        *,
        num_records: int,
        dataset_name: str,
        resume: ResumeMode,
    ) -> FakeCreateResult:
        if hasattr(config_builder, "build"):
            source = config_builder.build().seed_config.source
            if isinstance(source, generation.dd.LocalFileSeedSource):
                self.result.rows = [json.loads(line) for line in Path(source.path).read_text().splitlines()][
                    :num_records
                ]
        self.create_calls.append(
            {
                "config_builder": config_builder,
                "num_records": num_records,
                "dataset_name": dataset_name,
                "resume": resume,
            }
        )
        return self.result

    def preview(self, config_builder: object, *, num_records: int) -> FakeCreateResult:
        """Return a preview result using the same captured fake object."""
        self.create_calls.append({"config_builder": config_builder, "num_records": num_records})
        return self.result


class ChangedSourceDataDesigner(FakeDataDesigner):
    """Simulate complete-count output whose trusted source content was changed."""

    def create(self, *args: object, **kwargs: object) -> FakeCreateResult:
        result = super().create(*args, **kwargs)
        result.rows[0]["retrieval_units"][0]["text"] = "Changed source text"
        return result


def test_canonical_source_drift_cannot_publish_success_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sources.jsonl"
    source.write_text('{"unit_id":"p1","document_id":"doc","text":"Original source"}')
    config = GenerationRunConfig(
        seed_source=RetrievalSourcesFile(path=source),
        artifact_path=tmp_path / "artifacts",
        output_dir=tmp_path / "output",
        model_providers=[],
    )
    monkeypatch.setattr(generation, "DataDesigner", ChangedSourceDataDesigner)
    with pytest.raises(ValueError, match="identity or content"):
        generation.run_generation(config)
    assert (config.output_dir / "retrieval_resolved.jsonl").is_file()
    assert not (config.artifact_path / ".retrieval_sdg_runs").exists()


@pytest.mark.parametrize("preview", [False, True])
def test_canonical_inputs_use_shared_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, preview: bool) -> None:
    FakeDataDesigner.instances.clear()
    image = tmp_path / "page.png"
    image.write_bytes(b"image fixture")
    sources = tmp_path / "sources.jsonl"
    records = [
        {"unit_id": "p1", "document_id": "doc", "text": "Texte", "language": "fr"},
        {"unit_id": "p2", "document_id": "doc", "images": ["page.png"], "language": "fr"},
    ]
    sources.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    monkeypatch.setattr(generation, "DataDesigner", FakeDataDesigner)
    config = GenerationRunConfig(
        seed_source=RetrievalSourcesFile(path=sources),
        output_dir=tmp_path / "output",
        artifact_path=tmp_path / "artifacts",
        model_providers=[],
    )
    if preview:
        result = generation.preview_generation(config)
        assert result.num_seed_records == 2
    else:
        result = generation.run_generation(config)
        assert result.requested_num_records == 2
        provenance = json.loads(result.provenance_path.read_text())
        assert provenance["config_sources"][0] == generation.config_source_from_path(sources).to_dict()
        assert len(provenance["config_sources"]) == 2
    builder = FakeDataDesigner.instances[0].create_calls[0]["config_builder"]
    native = builder.build()
    snapshot = Path(native.seed_config.source.path)
    rows = [json.loads(line) for line in snapshot.read_text().splitlines()]
    assert [row["source_id"] for row in rows] == ["p1", "p2"]
    assert [row["language"] for row in rows] == ["fr", "fr"]
    assert Path(rows[1]["images"][0]).read_bytes() == image.read_bytes()
    assert Path(rows[1]["images"][0]) != image
    assert rows[1]["retrieval_units"][0]["document_id"] == "doc"
    assert {"source_blind_evaluations", "qa_evaluations", "source_assessments"} <= {
        column.name for column in native.columns
    }


@pytest.mark.parametrize("preview", [False, True])
def test_canonical_overrequest_fails_before_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, preview: bool
) -> None:
    FakeDataDesigner.instances.clear()
    sources = tmp_path / "sources.jsonl"
    sources.write_text('{"unit_id":"a","document_id":"d","text":"source"}', encoding="utf-8")
    monkeypatch.setattr(generation, "DataDesigner", FakeDataDesigner)
    config = GenerationRunConfig(
        seed_source=RetrievalSourcesFile(path=sources), num_records=2, artifact_path=tmp_path / "artifacts"
    )
    with pytest.raises(ValueError, match="exceeds"):
        if preview:
            generation.preview_generation(config)
        else:
            generation.run_generation(config)
    assert FakeDataDesigner.instances == []


def test_run_generation_returns_stable_artifact_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeDataDesigner.instances.clear()
    build_calls: list[dict[str, object]] = []
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "source.txt").write_text("A source document.", encoding="utf-8")

    monkeypatch.setattr(generation, "DataDesigner", FakeDataDesigner)
    monkeypatch.setattr(generation, "_count_seed_records", lambda _: 3)
    monkeypatch.setattr(
        generation,
        "build_qa_generation_pipeline",
        lambda **kwargs: build_calls.append(kwargs) or {"builder": "qa"},
    )
    monkeypatch.setattr(generation, "_producer_version", lambda: "0.1.0")

    result = generation.run_generation(
        GenerationRunConfig(
            seed_source=DocumentChunkerSeedSource(path=str(docs)),
            output_dir=tmp_path / "output",
            artifact_path=tmp_path / "artifacts",
            dataset_name="retrieval",
            buffer_size=37,
            resume=ResumeMode.ALWAYS,
            model_providers=[],
            pipeline=GenerationPipelineConfig(num_pairs=7),
        )
    )

    instance = FakeDataDesigner.instances[0]
    assert instance.run_config.buffer_size == 37
    assert instance.run_config.disable_early_shutdown is True
    assert instance.create_calls == [
        {
            "config_builder": {"builder": "qa"},
            "num_records": 3,
            "dataset_name": "retrieval",
            "resume": ResumeMode.ALWAYS,
        }
    ]
    assert len(build_calls) == 1
    assert build_calls[0]["seed_source"] == generation.DocumentChunkerSeedSource(path=str(docs))
    assert build_calls[0]["start_index"] == 0
    assert build_calls[0]["end_index"] == 2
    assert build_calls[0]["num_pairs"] == 7
    assert build_calls[0]["artifact_extraction_model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert build_calls[0]["embed_model"] == "nvidia/nemotron-3-embed-1b"
    assert instance.result.export_calls == [(tmp_path / "output" / "retrieval_resolved.jsonl", "jsonl")]
    assert result.output_path == tmp_path / "output" / "retrieval_resolved.jsonl"
    assert result.dataset_path == tmp_path / "artifacts" / "retrieval_resolved"
    assert result.dataset_name == "retrieval_resolved"
    assert result.num_records == 2
    assert result.requested_num_records == 3
    assert result.producer_version == "0.1.0"
    assert (
        result.resolved_config_path
        == tmp_path / "artifacts" / ".retrieval_sdg_runs" / "retrieval_resolved" / "resolved_config.yaml"
    )
    assert result.provenance_path == (
        tmp_path / "artifacts" / ".retrieval_sdg_runs" / "retrieval_resolved" / "config_provenance.json"
    )


def test_run_generation_does_not_write_plugin_metadata_when_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingDataDesigner(FakeDataDesigner):
        def create(self, *args: object, **kwargs: object) -> FakeCreateResult:
            raise RuntimeError("generation failed")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "source.txt").write_text("A source document.", encoding="utf-8")
    monkeypatch.setattr(generation, "DataDesigner", FailingDataDesigner)
    monkeypatch.setattr(generation, "_count_seed_records", lambda _: 1)
    monkeypatch.setattr(generation, "build_qa_generation_pipeline", lambda **_: {"builder": "qa"})

    with pytest.raises(RuntimeError, match="generation failed"):
        generation.run_generation(
            GenerationRunConfig(
                seed_source=DocumentChunkerSeedSource(path=str(docs)),
                output_dir=tmp_path / "output",
                artifact_path=tmp_path / "artifacts",
                dataset_name="retrieval",
                model_providers=[],
            )
        )

    assert not (tmp_path / "artifacts" / ".retrieval_sdg_runs").exists()


def test_run_generation_creates_output_directory_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(generation, "_count_seed_records", lambda _: 1)
    monkeypatch.setattr(
        generation,
        "DataDesigner",
        lambda *args, **kwargs: pytest.fail("DataDesigner must not be constructed before output directory validation"),
    )

    with pytest.raises(FileExistsError):
        generation.run_generation(
            GenerationRunConfig(
                seed_source=DocumentChunkerSeedSource(path=str(tmp_path)),
                output_dir=output_dir,
            )
        )


@pytest.mark.parametrize("buffer_size", [0, -1])
def test_run_generation_rejects_nonpositive_buffer_size(tmp_path: Path, buffer_size: int) -> None:
    with pytest.raises(ValueError, match="buffer_size"):
        generation.run_generation(
            GenerationRunConfig(
                seed_source=DocumentChunkerSeedSource(path=str(tmp_path)),
                output_dir=tmp_path / "output",
                buffer_size=buffer_size,
            )
        )


def test_preview_generation_uses_bounded_seed_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeDataDesigner.instances.clear()
    build_calls: list[dict[str, object]] = []
    monkeypatch.setattr(generation, "DataDesigner", FakeDataDesigner)
    monkeypatch.setattr(generation, "_count_seed_records", lambda _: 100)
    monkeypatch.setattr(
        generation,
        "build_qa_generation_pipeline",
        lambda **kwargs: build_calls.append(kwargs) or {"builder": "preview"},
    )

    result = generation.preview_generation(
        GenerationRunConfig(
            seed_source=DocumentChunkerSeedSource(path=str(tmp_path)),
            output_dir=tmp_path / "output",
            buffer_size=20,
        )
    )

    assert build_calls[0]["start_index"] == 0
    assert build_calls[0]["end_index"] == 19
    assert FakeDataDesigner.instances[0].create_calls == [{"config_builder": {"builder": "preview"}, "num_records": 1}]
    assert result.num_seed_records == 100
    assert result.num_preview_records == 1
