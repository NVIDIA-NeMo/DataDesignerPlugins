# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``ddp new`` scaffold generator."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ddp import scaffold


def write_external_tap_repo(root: Path) -> None:
    """Write a minimal external-style tap repository skeleton.

    Args:
        root: Temporary repository root.
    """
    (root / "plugins").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "external-tap-workspace"

            [tool.ddp.tap]
            catalog-url = "https://catalog.example.test/plugins.json"
            repository-url = "https://git.example.test/acme/dd-plugins"
            repository-git-url = "https://git.example.test/acme/dd-plugins.git"
            docs-base-url = "https://docs.example.test/ddp/"
            package-prefix = "acme-dd-"
            default-source = "pypi"
            release-ref-template = "release/{package}/{version}"
            default-data-designer-requirement = "data-designer>=9.9"
            author-name = "ACME Labs"
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_scaffold_uses_external_tap_config_in_generated_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_external_tap_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scaffold, "_discover_owner", lambda: "@acme/platform")

    scaffold.main(["sample-plugin"])

    plugin_dir = tmp_path / "plugins" / "acme-dd-sample-plugin"
    assert plugin_dir.is_dir()
    assert (plugin_dir / "src" / "acme_dd_sample_plugin").is_dir()

    pyproject = (plugin_dir / "pyproject.toml").read_text(encoding="utf-8")
    readme = (plugin_dir / "README.md").read_text(encoding="utf-8")
    docs_index = (plugin_dir / "docs" / "index.md").read_text(encoding="utf-8")
    test_file = (plugin_dir / "tests" / "test_plugin.py").read_text(encoding="utf-8")

    assert 'name = "acme-dd-sample-plugin"' in pyproject
    assert '"data-designer>=9.9"' in pyproject
    assert '{name = "ACME Labs"}' in pyproject
    assert 'Repository = "https://git.example.test/acme/dd-plugins"' in pyproject
    assert 'sample-plugin = "acme_dd_sample_plugin.plugin:plugin"' in pyproject
    assert "github.com/NVIDIA-NeMo/DataDesignerPlugins" not in pyproject
    assert "NVIDIA Corporation" not in pyproject

    assert "# acme-dd-sample-plugin" in readme
    assert "uv add data-designer acme-dd-sample-plugin" in readme
    assert "https://docs.example.test/ddp/authoring/" in readme
    assert "github.com/NVIDIA-NeMo/DataDesignerPlugins" not in readme

    assert "# acme-dd-sample-plugin" in docs_index
    assert "uv add data-designer acme-dd-sample-plugin" in docs_index
    assert "https://docs.example.test/ddp/authoring/" in docs_index
    assert "github.com/NVIDIA-NeMo/DataDesignerPlugins" not in docs_index

    assert "from acme_dd_sample_plugin.plugin import plugin" in test_file


def test_scaffold_missing_tap_config_exits_with_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "plugins").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "missing-tap"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        scaffold.main(["sample-plugin"])

    assert exc_info.value.code == 1
    assert "[tool.ddp.tap]" in capsys.readouterr().err
