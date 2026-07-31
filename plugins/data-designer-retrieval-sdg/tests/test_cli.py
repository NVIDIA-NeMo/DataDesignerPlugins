# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import data_designer.config as dd
import pytest
from data_designer.engine.storage.artifact_storage import ResumeMode

from data_designer_retrieval_sdg import cli
from data_designer_retrieval_sdg.generation import GenerationResult

RUN_CONFIGS: list[object] = []
PREVIEW_CONFIGS: list[object] = []


def fake_count_seed_records(seed_source: object) -> int:
    """Return a deterministic seed count for CLI generation tests."""
    return 3


def fake_build_model_providers(**kwargs: object) -> tuple[list[dd.ModelProvider], list[dd.ModelProvider]]:
    """Return a deterministic provider tuple for CLI generation tests."""
    return [], []


def fake_run_generation(config: object) -> GenerationResult:
    """Capture a public generation request and return deterministic metadata."""
    RUN_CONFIGS.append(config)
    return GenerationResult(
        output_path=Path("/output/my_run_resolved.jsonl"),
        dataset_path=Path("/artifacts/my_run_resolved"),
        dataset_name="my_run_resolved",
        num_records=3,
        requested_num_records=3,
        producer_version="0.1.0",
    )


def fake_preview_generation(config: object) -> None:
    """Capture a public preview request."""
    PREVIEW_CONFIGS.append(config)


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
    RUN_CONFIGS.clear()
    monkeypatch.setattr(cli, "_count_seed_records", fake_count_seed_records)
    monkeypatch.setattr(cli, "build_model_providers", fake_build_model_providers)
    monkeypatch.setattr(cli, "run_generation", fake_run_generation)
    monkeypatch.setattr(sys, "argv", generate_argv(tmp_path))

    cli.main()

    config = RUN_CONFIGS[0]
    assert config.buffer_size == 37
    assert config.resume == ResumeMode.ALWAYS.value
    assert config.num_records == 3
    assert config.dataset_name == "my_run"
    assert config.output_dir == tmp_path / "out"
    assert config.pipeline.num_pairs == 7
    assert config.pipeline.embed_model == "nvidia/nemotron-3-embed-1b"


def test_preview_delegates_to_public_generation_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    PREVIEW_CONFIGS.clear()
    monkeypatch.setattr(cli, "_count_seed_records", fake_count_seed_records)
    monkeypatch.setattr(cli, "build_model_providers", fake_build_model_providers)
    monkeypatch.setattr(cli, "preview_generation", fake_preview_generation)
    monkeypatch.setattr(sys, "argv", generate_argv(tmp_path, extra_args=["--preview"]))

    cli.main()

    config = PREVIEW_CONFIGS[0]
    assert config.num_records == 3
    assert config.buffer_size == 37
    assert config.pipeline.num_pairs == 7


def test_generate_accepts_matching_custom_question_distributions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RUN_CONFIGS.clear()
    monkeypatch.setattr(cli, "_count_seed_records", fake_count_seed_records)
    monkeypatch.setattr(cli, "build_model_providers", fake_build_model_providers)
    monkeypatch.setattr(cli, "run_generation", fake_run_generation)
    monkeypatch.setattr(
        sys,
        "argv",
        generate_argv(
            tmp_path,
            extra_args=[
                "--num-pairs",
                "3",
                "--query-counts",
                "multi_hop=1",
                "structural=1",
                "contextual=1",
                "--reasoning-counts",
                "factual=1",
                "relational=1",
                "inferential=1",
                "temporal=0",
                "procedural=0",
                "causal=0",
                "visual=0",
            ],
        ),
    )

    cli.main()

    config = RUN_CONFIGS[0]
    assert config.pipeline.num_pairs == 3
    assert config.pipeline.query_counts == {"multi_hop": 1, "structural": 1, "contextual": 1}
    assert config.pipeline.reasoning_counts == {
        "factual": 1,
        "relational": 1,
        "inferential": 1,
        "temporal": 0,
        "procedural": 0,
        "causal": 0,
        "visual": 0,
    }


def test_print_model_config_does_not_expose_provider_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    provider = dd.ModelProvider(
        name="custom",
        endpoint="https://example.invalid/v1",
        provider_type="openai",
        api_key="do-not-print-this",
    )

    cli._print_model_config(cli.GenerationPipelineConfig(), [provider])

    output = capsys.readouterr().out
    assert "do-not-print-this" not in output
    assert "credential=configured" in output


@pytest.mark.parametrize("dataset_name", ["", ".", "..", "nested/name", "nested\\name", "bad\nname"])
def test_generate_rejects_unsafe_dataset_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
) -> None:
    FakeDataDesigner.instances.clear()
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
