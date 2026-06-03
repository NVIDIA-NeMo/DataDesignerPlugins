# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        self.result = FakeCreateResult(FakeArtifactStorage(artifact_path / "my_run_resolved", "my_run_resolved"))
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


def generate_argv(
    tmp_path: Path,
    *,
    dataset_name: str = "my_run",
    artifact_path: Path | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build generate CLI arguments for parser-level tests."""
    input_dir = tmp_path / "docs"
    input_dir.mkdir(exist_ok=True)
    argv = [
        "data-designer-retrieval-sdg",
        "generate",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(tmp_path / "out"),
        "--artifact-path",
        str(artifact_path or tmp_path / "artifacts"),
        "--dataset-name",
        dataset_name,
        "--buffer-size",
        "37",
        "--resume",
        "always",
    ]
    if extra_args:
        argv.extend(extra_args)
    return argv


def test_generate_uses_native_resume_and_exports_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    BUILD_CALLS.clear()
    FakeDataDesigner.instances.clear()
    monkeypatch.setattr(cli, "DataDesigner", FakeDataDesigner)
    monkeypatch.setattr(cli, "_count_seed_records", fake_count_seed_records)
    monkeypatch.setattr(cli, "build_model_providers", fake_build_model_providers)
    monkeypatch.setattr(cli, "build_qa_generation_pipeline", fake_build_qa_generation_pipeline)
    monkeypatch.setattr(sys, "argv", generate_argv(tmp_path))

    cli.main()

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
    assert instance.result.export_calls == [(tmp_path / "out" / "my_run_resolved.jsonl", "jsonl")]


@pytest.mark.parametrize("dataset_name", ["", ".", "..", "nested/name", "nested\\name", "bad\nname"])
def test_generate_rejects_unsafe_dataset_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
) -> None:
    FakeDataDesigner.instances.clear()
    monkeypatch.setattr(cli, "DataDesigner", FakeDataDesigner)
    monkeypatch.setattr(cli, "_count_seed_records", fake_count_seed_records)
    monkeypatch.setattr(sys, "argv", generate_argv(tmp_path, dataset_name=dataset_name))

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert FakeDataDesigner.instances == []


def test_generate_rejects_dataset_name_that_resolves_outside_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    outside_path = tmp_path / "outside"
    outside_path.mkdir()
    (artifact_path / "linked").symlink_to(outside_path, target_is_directory=True)
    FakeDataDesigner.instances.clear()
    monkeypatch.setattr(cli, "DataDesigner", FakeDataDesigner)
    monkeypatch.setattr(cli, "_count_seed_records", fake_count_seed_records)
    monkeypatch.setattr(sys, "argv", generate_argv(tmp_path, dataset_name="linked", artifact_path=artifact_path))

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert FakeDataDesigner.instances == []


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
