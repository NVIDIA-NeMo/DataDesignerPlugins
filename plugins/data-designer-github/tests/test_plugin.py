# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner

from data_designer_github.config import GitHubSeedSource
from data_designer_github.impl import GitHubSeedReader, normalize_github_repository
from data_designer_github.plugin import plugin


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)


def test_normalize_github_repository() -> None:
    assert normalize_github_repository("NVIDIA-NeMo/DataDesigner")[0] == "NVIDIA-NeMo/DataDesigner"
    assert normalize_github_repository("https://github.com/NVIDIA-NeMo/DataDesigner.git")[0] == (
        "NVIDIA-NeMo/DataDesigner"
    )


def test_source_requires_at_least_one_repository_source() -> None:
    with pytest.raises(ValueError, match="At least one"):
        GitHubSeedSource()


def test_reader_hydrates_local_repository_files(tmp_path: Path) -> None:
    repo = _create_git_repo(tmp_path / "sample-repo")
    source = GitHubSeedSource(repository_paths=[str(repo)], file_pattern="*.py")
    reader = GitHubSeedReader()
    reader.attach(source, PlaintextResolver())

    assert reader.get_seed_dataset_size() == 1
    batch = reader.create_batch_reader(batch_size=10, index_range=None, shuffle=False).read_next_batch()
    rows = batch.to_pandas().to_dict(orient="records")

    assert len(rows) == 1
    row = rows[0]
    assert row["repo_id"] == "sample-repo"
    assert row["source_kind"] == "git_repository"
    assert row["relative_path"] == "src/example.py"
    assert row["file_name"] == "example.py"
    assert row["file_extension"] == ".py"
    assert row["code_lang"] == "python"
    assert row["size_bytes"] > 0
    assert len(row["commit_sha"]) == 40
    assert len(row["content_sha256"]) == 64
    assert "def greet" in row["content"]


def test_parent_path_discovers_child_git_repositories(tmp_path: Path) -> None:
    repo = _create_git_repo(tmp_path / "repos" / "child-repo")
    source = GitHubSeedSource(path=str(repo.parent), file_pattern="*.py")
    reader = GitHubSeedReader()
    reader.attach(source, PlaintextResolver())

    batch = reader.create_batch_reader(batch_size=10, index_range=None, shuffle=False).read_next_batch()
    rows = batch.to_pandas().to_dict(orient="records")

    assert [row["repo_id"] for row in rows] == ["child-repo"]


def test_preview_uses_github_seed_reader(tmp_path: Path) -> None:
    repo = _create_git_repo(tmp_path / "preview-repo")
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(GitHubSeedSource(repository_paths=[str(repo)], file_pattern="*.py"))
    builder.add_column(name="_row_id", column_type="sampler", sampler_type="uuid", params={})

    result = DataDesigner(artifact_path=tmp_path / "artifacts").preview(builder, num_records=1)

    assert result.dataset is not None
    assert list(result.dataset["repo_id"]) == ["preview-repo"]
    assert list(result.dataset["relative_path"]) == ["src/example.py"]
    assert "def greet" in result.dataset["content"].iloc[0]


def _create_git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    src = path / "src"
    src.mkdir()
    (src / "example.py").write_text(
        "import os\n\n\ndef greet(name: str) -> str:\n    return f'hello {name} from {os.getcwd()}'\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text("# Sample\n", encoding="utf-8")
    _git(path, "init", "--quiet")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")
    _git(path, "add", ".")
    _git(path, "commit", "--quiet", "-m", "initial")
    return path


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)
