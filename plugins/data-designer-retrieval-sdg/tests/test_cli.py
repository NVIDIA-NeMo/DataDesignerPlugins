# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import data_designer.config as dd
import pytest
import yaml
from data_designer.engine.storage.artifact_storage import ResumeMode

from data_designer_retrieval_sdg import cli
from data_designer_retrieval_sdg.generation import GenerationResult
from data_designer_retrieval_sdg.run_config import GenerationPipelineConfig

RUN_CONFIGS: list[object] = []
RUN_KWARGS: list[dict[str, object]] = []
PREVIEW_CONFIGS: list[object] = []


def fake_count_seed_records(seed_source: object) -> int:
    """Return a deterministic seed count for CLI generation tests."""
    return 3


def fake_build_model_providers(**kwargs: object) -> tuple[list[dd.ModelProvider], list[dd.ModelProvider]]:
    """Return a deterministic provider tuple for CLI generation tests."""
    return [], []


def fake_run_generation(config: object, **kwargs: object) -> GenerationResult:
    """Capture a public generation request and return deterministic metadata."""
    RUN_CONFIGS.append(config)
    RUN_KWARGS.append(kwargs)
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
    RUN_KWARGS.clear()
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
    assert RUN_KWARGS[0]["override_paths"] == (
        "seed_source.path",
        "output_dir",
        "artifact_path",
        "dataset_name",
        "buffer_size",
        "resume",
    )
    assert RUN_KWARGS[0]["config_sources"] == ()


def test_generate_respects_num_records_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    RUN_CONFIGS.clear()
    monkeypatch.setattr(cli, "_count_seed_records", fake_count_seed_records)
    monkeypatch.setattr(cli, "build_model_providers", fake_build_model_providers)
    monkeypatch.setattr(cli, "run_generation", fake_run_generation)
    monkeypatch.setattr(sys, "argv", generate_argv(tmp_path, extra_args=["--num-records", "2"]))

    cli.main()

    assert RUN_CONFIGS[0].num_records == 2
    output = capsys.readouterr().out
    assert "Discovered 3 text files" in output
    assert "Selected the first 2 seed records" in output
    assert "Records to process: 2" in output


def test_generate_rejects_num_records_above_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_count_seed_records", fake_count_seed_records)
    monkeypatch.setattr(cli, "build_model_providers", fake_build_model_providers)
    monkeypatch.setattr(cli, "run_generation", lambda *_args, **_kwargs: pytest.fail("generation must not start"))
    monkeypatch.setattr(sys, "argv", generate_argv(tmp_path, extra_args=["--num-records", "4"]))

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "num_records=4 exceeds the 3 available seed records" in capsys.readouterr().err


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

    cli._print_model_config(GenerationPipelineConfig(), [provider])

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


def test_generate_print_resolved_config_uses_documented_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "generation.yaml"
    config_path.write_text(
        "seed_source:\n  multi_doc: true\nnum_records: 2\npipeline:\n  min_complexity: 1\n",
        encoding="utf-8",
    )
    argv = generate_argv(
        tmp_path,
        extra_args=[
            "--config",
            str(config_path),
            "--min-complexity",
            "2",
            "--num-records",
            "3",
            "--no-multi-doc",
            "--print-resolved-config",
        ],
    )
    monkeypatch.setattr(cli, "_count_seed_records", lambda _: pytest.fail("print should not scan the corpus"))
    monkeypatch.setattr(cli, "build_model_providers", fake_build_model_providers)
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["pipeline"]["min_complexity"] == 2
    assert resolved["dataset_name"] == "my_run"
    assert resolved["buffer_size"] == 37
    assert resolved["seed_source"]["multi_doc"] is False
    assert resolved["num_records"] == 3


def test_generate_print_resolved_config_redacts_provider_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "generation.yaml"
    config_path.write_text(
        """
model_providers:
  - name: custom
    endpoint: https://example.invalid/v1
    provider_type: openai
    api_key: do-not-print
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-designer-retrieval-sdg",
            "generate",
            "--config",
            str(config_path),
            "--print-resolved-config",
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "do-not-print" not in output
    assert "api_key: <redacted>" in output


def test_convert_print_resolved_config_uses_file_and_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "conversion.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input_path": str(tmp_path / "records.jsonl"),
                "corpus_id": "configured",
                "seed": 5,
                "eval_only": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-designer-retrieval-sdg",
            "convert",
            "--config",
            str(config_path),
            "--seed",
            "7",
            "--no-eval-only",
            "--print-resolved-config",
        ],
    )

    cli.main()

    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["input_path"] == str(tmp_path / "records.jsonl")
    assert resolved["corpus_id"] == "configured"
    assert resolved["seed"] == 7
    assert resolved["eval_only"] is False


def test_convert_preserves_positional_and_multi_value_legacy_forms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    positional_input = tmp_path / "positional.jsonl"
    explicit_input = tmp_path / "explicit.jsonl"
    group_paths = [tmp_path / "groups-a.json", tmp_path / "groups-b.json"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-designer-retrieval-sdg",
            "convert",
            str(positional_input),
            "--input-path",
            str(explicit_input),
            "--corpus-id",
            "configured",
            "--groups-json",
            *(str(path) for path in group_paths),
            "--print-resolved-config",
        ],
    )

    cli.main()

    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["input_path"] == str(explicit_input)
    assert resolved["groups_json"] == [str(path) for path in group_paths]


def test_generate_cli_provider_overrides_matching_config_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "generation.yaml"
    config_path.write_text(
        """
model_providers:
  - name: custom
    endpoint: https://old.example.invalid/v1
    provider_type: openai
    api_key: CUSTOM_API_KEY
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data-designer-retrieval-sdg",
            "generate",
            "--config",
            str(config_path),
            "--custom-provider-name",
            "custom",
            "--custom-provider-endpoint",
            "https://new.example.invalid/v1",
            "--print-resolved-config",
        ],
    )

    cli.main()

    resolved = yaml.safe_load(capsys.readouterr().out)
    provider = next(item for item in resolved["model_providers"] if item["name"] == "custom")
    assert provider["endpoint"] == "https://new.example.invalid/v1"
    assert provider["provider_type"] == "openai"
    assert provider["api_key"] == "<redacted>"


@pytest.mark.parametrize("command", ["generate", "convert"])
def test_commands_reject_removed_set_override(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    monkeypatch.setattr(sys, "argv", ["data-designer-retrieval-sdg", command, "--set", "seed=1"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("command", "expected_defaults"),
    [
        (
            "generate",
            (
                "(default: generated)",
                "(default: ['.txt', '.md', '.text'])",
                "(default: 50)",
                "(default: nvidia/nemotron-3-ultra-550b-a55b)",
            ),
        ),
        (
            "convert",
            (
                "(default: generated/retrieval_sdg.jsonl)",
                "(default: retrieval_sdg)",
                "(default: 0.8)",
            ),
        ),
    ],
)
def test_command_help_reflects_pydantic_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    expected_defaults: tuple[str, ...],
) -> None:
    monkeypatch.setattr(sys, "argv", ["data-designer-retrieval-sdg", command, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for expected in expected_defaults:
        assert expected in output


@pytest.mark.parametrize("command", ["generate", "convert"])
def test_commands_require_explicit_input_for_execution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["data-designer-retrieval-sdg", command])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "provide" in capsys.readouterr().err
