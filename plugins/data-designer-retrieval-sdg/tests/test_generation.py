# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stable generation Python API."""

from pathlib import Path

import pytest
from data_designer.engine.storage.artifact_storage import ResumeMode

from data_designer_retrieval_sdg import generation
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

    def export(self, path: Path, *, format: str | None = None) -> Path:
        self.export_calls.append((path, format))
        path.write_text("{}\n", encoding="utf-8")
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


def test_run_generation_returns_stable_artifact_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeDataDesigner.instances.clear()
    build_calls: list[dict[str, object]] = []
    docs = tmp_path / "docs"
    docs.mkdir()

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
            pipeline=GenerationPipelineConfig(num_pairs=10),
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
    assert build_calls[0]["num_pairs"] == 10
    assert build_calls[0]["artifact_extraction_model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert build_calls[0]["embed_model"] == "nvidia/nemotron-3-embed-1b"
    assert instance.result.export_calls == [(tmp_path / "output" / "retrieval_resolved.jsonl", "jsonl")]
    assert result.output_path == tmp_path / "output" / "retrieval_resolved.jsonl"
    assert result.dataset_path == tmp_path / "artifacts" / "retrieval_resolved"
    assert result.dataset_name == "retrieval_resolved"
    assert result.num_records == 3
    assert result.producer_version == "0.1.0"


def test_run_generation_rejects_nonpositive_buffer_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="buffer_size"):
        generation.run_generation(
            GenerationRunConfig(
                seed_source=DocumentChunkerSeedSource(path=str(tmp_path)),
                output_dir=tmp_path / "output",
                buffer_size=0,
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
