# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
from data_designer.engine.storage.artifact_storage import ResumeMode

from data_designer_retrieval_sdg import cli

BUILD_CALLS: list[dict[str, object]] = []


def fake_count_seed_records(seed_source: object) -> int:
    """Return a deterministic seed count for CLI generation tests."""
    return 3


def fake_build_model_providers(**kwargs: object) -> tuple[list[str], list[object]]:
    """Return a deterministic provider tuple for CLI generation tests."""
    return ["providers"], []


def fake_build_qa_generation_pipeline(**kwargs: object) -> object:
    """Capture pipeline-builder kwargs and return a sentinel builder."""
    BUILD_CALLS.append(kwargs)
    return {"builder": "qa"}


class FakeArtifactStorage:
    """Minimal artifact storage surface used by the generate command."""

    def __init__(self, base_dataset_path: Path, resolved_dataset_name: str) -> None:
        self.base_dataset_path = base_dataset_path
        self.resolved_dataset_name = resolved_dataset_name


class FakeCreateResult:
    """Minimal DataDesigner result surface used by the generate command."""

    def __init__(self, artifact_storage: FakeArtifactStorage) -> None:
        self.artifact_storage = artifact_storage
        self.export_calls: list[tuple[Path, str | None]] = []

    def export(self, path: Path, *, format: str | None = None) -> Path:
        self.export_calls.append((path, format))
        path.write_text("", encoding="utf-8")
        return path


class FakeDataDesigner:
    """Capture DataDesigner calls made by the generate command."""

    instances: list[FakeDataDesigner] = []

    def __init__(self, artifact_path: Path, model_providers: object) -> None:
        self.artifact_path = artifact_path
        self.model_providers = model_providers
        self.run_config = None
        self.create_calls: list[dict[str, object]] = []
        self.result = FakeCreateResult(FakeArtifactStorage(artifact_path / "my_run", "my_run"))
        FakeDataDesigner.instances.append(self)

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


def _generate_args(tmp_path: Path) -> argparse.Namespace:
    """Build generate args with defaults that match the CLI parser."""
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    return argparse.Namespace(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        file_pattern="*",
        recursive=True,
        file_extensions=[".txt", ".md", ".text"],
        min_text_length=50,
        sentences_per_chunk=5,
        num_sections=1,
        num_files=None,
        max_artifacts_per_type=2,
        num_pairs=7,
        min_hops=2,
        max_hops=4,
        min_complexity=4,
        similarity_threshold=0.9,
        preview=False,
        artifact_path=tmp_path / "artifacts",
        dataset_name="my_run",
        buffer_size=37,
        resume=ResumeMode.ALWAYS.value,
        multi_doc=False,
        bundle_size=2,
        bundle_strategy="sequential",
        max_docs_per_bundle=3,
        multi_doc_manifest=None,
        log_level="INFO",
        artifact_extraction_model="artifact-model",
        artifact_extraction_provider="nvidia",
        qa_generation_model="qa-model",
        qa_generation_provider="nvidia",
        quality_judge_model="judge-model",
        quality_judge_provider="nvidia",
        embed_model="embed-model",
        embed_provider="nvidia",
        max_parallel_requests_for_gen=None,
        custom_provider_endpoint=None,
        custom_provider_name="custom",
        custom_provider_type="openai",
        custom_provider_api_key=None,
        model_providers_file=None,
    )


def test_generate_uses_native_resume_and_exports_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    BUILD_CALLS.clear()
    FakeDataDesigner.instances.clear()
    monkeypatch.setattr(cli, "DataDesigner", FakeDataDesigner)
    monkeypatch.setattr(cli, "_count_seed_records", fake_count_seed_records)
    monkeypatch.setattr(cli, "build_model_providers", fake_build_model_providers)
    monkeypatch.setattr(cli, "build_qa_generation_pipeline", fake_build_qa_generation_pipeline)

    cli._run_generate(_generate_args(tmp_path))

    instance = FakeDataDesigner.instances[0]
    assert instance.run_config.buffer_size == 37
    assert instance.run_config.disable_early_shutdown is True
    assert instance.create_calls == [
        {
            "config_builder": {"builder": "qa"},
            "num_records": 3,
            "dataset_name": "my_run",
            "resume": ResumeMode.ALWAYS,
        }
    ]
    assert BUILD_CALLS[0]["start_index"] == 0
    assert BUILD_CALLS[0]["end_index"] == 2
    assert instance.result.export_calls == [(tmp_path / "out" / "my_run.jsonl", "jsonl")]


@pytest.mark.parametrize("removed_flag", ["--batch-size", "--start-batch-index", "--end-batch-index"])
def test_generate_rejects_removed_batch_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    removed_flag: str,
) -> None:
    argv = [
        "data-designer-retrieval-sdg",
        "generate",
        "--input-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        removed_flag,
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
