# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ddp.plugin_docs."""

from __future__ import annotations

from pathlib import Path

from ddp.plugin_docs import check_plugin_docs, discover_plugin_docs, sync_plugin_docs

ZENSICAL_TEMPLATE = """\
[project]
site_name = "Test"
nav = [
  {"Home" = "index.md"},
  {"Plugins" = [
    {"Overview" = "plugins/index.md"},
    # BEGIN GENERATED PLUGIN DOCS NAV
    # END GENERATED PLUGIN DOCS NAV
  ]},
]
"""


def write_file(path: Path, content: str) -> None:
    """Write a UTF-8 fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_plugin_pyproject(repo_root: Path, package_name: str, column_type: str) -> None:
    """Write minimal plugin pyproject metadata."""
    plugin_dir = repo_root / "plugins" / package_name
    write_file(
        plugin_dir / "pyproject.toml",
        f"""\
[project]
name = "{package_name}"
version = "0.1.0"
description = "{package_name} description"

[project.entry-points."data_designer.plugins"]
{column_type} = "{package_name.replace("-", "_")}.plugin:plugin"
""",
    )


def write_repo_skeleton(repo_root: Path) -> None:
    """Write minimal repository files needed by the docs generator."""
    write_file(repo_root / "pyproject.toml", '[project]\nname = "workspace"\n')
    write_file(repo_root / "docs" / "index.md", "# Home\n")
    write_file(repo_root / "zensical.toml", ZENSICAL_TEMPLATE)


def test_discover_plugin_docs_reads_metadata_and_source_docs(tmp_path: Path) -> None:
    write_repo_skeleton(tmp_path)
    write_plugin_pyproject(tmp_path, "data-designer-alpha", "alpha")
    write_file(tmp_path / "plugins" / "data-designer-alpha" / "docs" / "index.md", "# Alpha\n")

    plugins = discover_plugin_docs(tmp_path)

    assert len(plugins) == 1
    assert plugins[0].package_name == "data-designer-alpha"
    assert plugins[0].column_types == ("alpha",)
    assert plugins[0].source_docs_dir == tmp_path / "plugins" / "data-designer-alpha" / "docs"


def test_sync_plugin_docs_copies_plugin_docs_and_updates_nav(tmp_path: Path) -> None:
    write_repo_skeleton(tmp_path)
    write_plugin_pyproject(tmp_path, "data-designer-alpha", "alpha")
    write_file(tmp_path / "plugins" / "data-designer-alpha" / "docs" / "index.md", "# Alpha docs\n")
    write_file(tmp_path / "plugins" / "data-designer-alpha" / "docs" / "usage.md", "# Usage\n")
    write_file(tmp_path / "plugins" / "data-designer-alpha" / "docs" / "assets" / "sample.txt", "asset\n")

    sync_plugin_docs(tmp_path)

    generated_root = tmp_path / "docs" / "plugins"
    index_page = (generated_root / "index.md").read_text(encoding="utf-8")
    assert "data-designer-alpha" in index_page
    assert "Browse available Data Designer plugins" in index_page
    assert "make plugin-docs" not in index_page
    assert (generated_root / "data-designer-alpha" / "index.md").read_text(encoding="utf-8") == "# Alpha docs\n"
    assert (generated_root / "data-designer-alpha" / "usage.md").read_text(encoding="utf-8") == "# Usage\n"
    assert (generated_root / "data-designer-alpha" / "assets" / "sample.txt").read_text(encoding="utf-8") == "asset\n"

    zensical = (tmp_path / "zensical.toml").read_text(encoding="utf-8")
    assert '{"data-designer-alpha" = [' in zensical
    assert '{"Overview" = "plugins/data-designer-alpha/index.md"}' in zensical
    assert '{"Usage" = "plugins/data-designer-alpha/usage.md"}' in zensical


def test_sync_plugin_docs_generates_fallback_page_without_plugin_docs(tmp_path: Path) -> None:
    write_repo_skeleton(tmp_path)
    write_plugin_pyproject(tmp_path, "data-designer-alpha", "alpha")

    sync_plugin_docs(tmp_path)

    page = (tmp_path / "docs" / "plugins" / "data-designer-alpha" / "index.md").read_text(encoding="utf-8")
    assert "# data-designer-alpha" in page
    assert "custom documentation" in page
    assert "Column types: `alpha`" in page


def test_sync_plugin_docs_maps_root_readme_to_index(tmp_path: Path) -> None:
    write_repo_skeleton(tmp_path)
    write_plugin_pyproject(tmp_path, "data-designer-alpha", "alpha")
    write_file(tmp_path / "plugins" / "data-designer-alpha" / "docs" / "README.md", "# Alpha README\n")

    sync_plugin_docs(tmp_path)

    page = (tmp_path / "docs" / "plugins" / "data-designer-alpha" / "index.md").read_text(encoding="utf-8")
    assert page == "# Alpha README\n"


def test_check_plugin_docs_detects_stale_generated_content(tmp_path: Path) -> None:
    write_repo_skeleton(tmp_path)
    write_plugin_pyproject(tmp_path, "data-designer-alpha", "alpha")
    write_file(tmp_path / "plugins" / "data-designer-alpha" / "docs" / "index.md", "# Alpha docs\n")
    sync_plugin_docs(tmp_path)
    write_file(tmp_path / "docs" / "plugins" / "data-designer-alpha" / "index.md", "# Stale\n")

    drift = check_plugin_docs(tmp_path)

    assert any("outdated generated docs file" in message for message in drift)
