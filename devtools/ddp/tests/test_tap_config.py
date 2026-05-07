# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for repository-level tap metadata loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ddp._repo import find_repo_root
from ddp.tap_config import TapConfigError, load_tap_config


def write_tap_pyproject(root: Path, overrides: dict[str, str] | None = None) -> None:
    """Write a root pyproject.toml containing a complete tap config.

    Args:
        root: Temporary repository root.
        overrides: Optional field values to override in ``[tool.ddp.tap]``.
    """
    values = {
        "catalog-url": "https://catalog.example.test/plugins.json",
        "repository-url": "https://git.example.test/acme/dd-plugins",
        "repository-git-url": "https://git.example.test/acme/dd-plugins.git",
        "docs-base-url": "https://docs.example.test/ddp/",
        "package-prefix": "acme-dd-",
        "default-source": "pypi",
        "release-ref-template": "release/{package}/{version}",
        "default-data-designer-requirement": "data-designer>=9.9",
        "author-name": "ACME Labs",
    }
    values.update(overrides or {})

    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "[project]",
        'name = "test-workspace"',
        "",
        "[tool.ddp.tap]",
    ]
    lines.extend(f'{key} = "{value}"' for key, value in values.items())
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_tap_config_reads_root_metadata_and_accessors(tmp_path: Path) -> None:
    write_tap_pyproject(tmp_path)

    config = load_tap_config(tmp_path)

    assert config.catalog_url == "https://catalog.example.test/plugins.json"
    assert config.repository_url == "https://git.example.test/acme/dd-plugins"
    assert config.repository_git_url == "https://git.example.test/acme/dd-plugins.git"
    assert config.docs_base_url == "https://docs.example.test/ddp/"
    assert config.package_prefix == "acme-dd-"
    assert config.default_source == "pypi"
    assert config.release_ref_for_package("acme-dd-widget", "1.2.3") == "release/acme-dd-widget/1.2.3"
    assert config.default_data_designer_requirement == "data-designer>=9.9"
    assert config.author_name == "ACME Labs"
    assert config.package_name_for_slug("widget") == "acme-dd-widget"
    assert config.docs_url_for_package("acme-dd-widget") == "https://docs.example.test/ddp/plugins/acme-dd-widget/"
    assert config.source_metadata_for_package("acme-dd-widget", "1.2.3", "plugins/acme-dd-widget") == {
        "type": "pypi",
        "package": "acme-dd-widget",
    }


def test_repository_tap_config_uses_expected_nvidia_defaults() -> None:
    config = load_tap_config(find_repo_root())

    assert (
        config.catalog_url
        == "https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json"
    )
    assert config.repository_url == "https://github.com/NVIDIA-NeMo/DataDesignerPlugins"
    assert config.repository_git_url == "https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git"
    assert config.docs_base_url == "https://nvidia-nemo.github.io/DataDesignerPlugins/"
    assert config.package_prefix == "data-designer-"
    assert config.default_source == "pypi"
    assert config.release_ref_template == "{package}/v{version}"
    assert config.default_data_designer_requirement == "data-designer>=0.5.7"
    assert config.author_name == "NVIDIA Corporation"


def test_source_metadata_for_git_source_uses_configured_repository_and_release_ref(tmp_path: Path) -> None:
    write_tap_pyproject(tmp_path, {"default-source": "git"})

    config = load_tap_config(tmp_path)

    assert config.source_metadata_for_package("acme-dd-widget", "1.2.3", "plugins/acme-dd-widget") == {
        "type": "git",
        "url": "https://git.example.test/acme/dd-plugins.git",
        "ref": "release/acme-dd-widget/1.2.3",
        "subdirectory": "plugins/acme-dd-widget",
    }


def test_source_metadata_for_path_source_uses_editable_local_path(tmp_path: Path) -> None:
    write_tap_pyproject(tmp_path, {"default-source": "path"})

    config = load_tap_config(tmp_path)

    assert config.source_metadata_for_package("acme-dd-widget", "1.2.3", "plugins/acme-dd-widget") == {
        "type": "path",
        "path": "plugins/acme-dd-widget",
        "editable": True,
    }


def test_missing_tap_config_errors_deterministically(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "test-workspace"
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(TapConfigError) as exc_info:
        load_tap_config(tmp_path)

    assert "[tool.ddp.tap]" in str(exc_info.value)
    assert "missing required" in str(exc_info.value)


def test_missing_required_tap_field_errors_deterministically(tmp_path: Path) -> None:
    write_tap_pyproject(tmp_path, {"author-name": ""})

    with pytest.raises(TapConfigError) as exc_info:
        load_tap_config(tmp_path)

    message = str(exc_info.value)
    assert "[tool.ddp.tap].author-name" in message
    assert "non-empty string" in message


def test_malformed_default_source_errors_deterministically(tmp_path: Path) -> None:
    write_tap_pyproject(tmp_path, {"default-source": "registry"})

    with pytest.raises(TapConfigError) as exc_info:
        load_tap_config(tmp_path)

    message = str(exc_info.value)
    assert "[tool.ddp.tap].default-source" in message
    assert "'pypi'" in message


def test_malformed_url_errors_deterministically(tmp_path: Path) -> None:
    write_tap_pyproject(tmp_path, {"catalog-url": "not-a-url"})

    with pytest.raises(TapConfigError) as exc_info:
        load_tap_config(tmp_path)

    message = str(exc_info.value)
    assert "[tool.ddp.tap].catalog-url" in message
    assert "absolute URL" in message


def test_malformed_data_designer_requirement_errors_deterministically(tmp_path: Path) -> None:
    write_tap_pyproject(tmp_path, {"default-data-designer-requirement": "requests>=2"})

    with pytest.raises(TapConfigError) as exc_info:
        load_tap_config(tmp_path)

    message = str(exc_info.value)
    assert "[tool.ddp.tap].default-data-designer-requirement" in message
    assert "data-designer" in message
