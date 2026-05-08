# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for repository-level catalog metadata loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ddp._repo import find_repo_root
from ddp.catalog_config import CatalogConfigError, load_catalog_config


def write_catalog_pyproject(root: Path, overrides: dict[str, str | None] | None = None) -> None:
    """Write a root pyproject.toml containing a complete catalog config.

    Args:
        root: Temporary repository root.
        overrides: Optional field values to override in ``[tool.ddp.catalog]``.
    """
    values = {
        "catalog-url": "https://docs.example.test/ddp/catalog/plugins.json",
        "repository-url": "https://git.example.test/acme/dd-plugins",
        "repository-git-url": "https://git.example.test/acme/dd-plugins.git",
        "docs-base-url": "https://docs.example.test/ddp/",
        "package-prefix": "acme-dd-",
        "package-index-url": "https://docs.example.test/ddp/simple/",
        "package-assets-url": "https://git.example.test/acme/dd-plugins/releases/download/ddp-package-assets/",
        "package-assets-release-tag": "ddp-package-assets",
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
        "[tool.ddp.catalog]",
    ]
    lines.extend(f'{key} = "{value}"' for key, value in values.items() if value is not None)
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_catalog_config_reads_root_metadata_and_accessors(tmp_path: Path) -> None:
    write_catalog_pyproject(tmp_path)

    config = load_catalog_config(tmp_path)

    assert config.catalog_url == "https://docs.example.test/ddp/catalog/plugins.json"
    assert config.repository_url == "https://git.example.test/acme/dd-plugins"
    assert config.repository_git_url == "https://git.example.test/acme/dd-plugins.git"
    assert config.docs_base_url == "https://docs.example.test/ddp/"
    assert config.package_prefix == "acme-dd-"
    assert config.package_index_url == "https://docs.example.test/ddp/simple/"
    assert config.package_assets_url == "https://git.example.test/acme/dd-plugins/releases/download/ddp-package-assets/"
    assert config.package_assets_release_tag == "ddp-package-assets"
    assert config.release_ref_for_package("acme-dd-widget", "1.2.3") == "release/acme-dd-widget/1.2.3"
    assert config.default_data_designer_requirement == "data-designer>=9.9"
    assert config.author_name == "ACME Labs"
    assert config.package_name_for_slug("widget") == "acme-dd-widget"
    assert config.docs_url_for_package("acme-dd-widget") == "https://docs.example.test/ddp/plugins/acme-dd-widget/"
    assert config.install_metadata_for_package("acme-dd-widget") == {
        "requirement": "acme-dd-widget",
        "index_url": "https://docs.example.test/ddp/simple/",
    }


def test_repository_catalog_config_uses_expected_nvidia_defaults() -> None:
    config = load_catalog_config(find_repo_root())

    assert config.catalog_url == "https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    assert config.repository_url == "https://github.com/NVIDIA-NeMo/DataDesignerPlugins"
    assert config.repository_git_url == "https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git"
    assert config.docs_base_url == "https://nvidia-nemo.github.io/DataDesignerPlugins/"
    assert config.package_prefix == "data-designer-"
    assert config.package_index_url == "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
    assert (
        config.package_assets_url
        == "https://github.com/NVIDIA-NeMo/DataDesignerPlugins/releases/download/ddp-package-assets/"
    )
    assert config.package_assets_release_tag == "ddp-package-assets"
    assert config.release_ref_template == "{package}/v{version}"
    assert config.default_data_designer_requirement == "data-designer>=0.5.7"
    assert config.author_name == "NVIDIA Corporation"


def test_git_only_fields_remain_optional_for_package_index_installs(tmp_path: Path) -> None:
    write_catalog_pyproject(
        tmp_path,
        {
            "repository-git-url": None,
            "release-ref-template": None,
        },
    )

    config = load_catalog_config(tmp_path)

    assert config.repository_git_url is None
    assert config.release_ref_template is None
    assert config.install_metadata_for_package("acme-dd-widget") == {
        "requirement": "acme-dd-widget",
        "index_url": "https://docs.example.test/ddp/simple/",
    }


def test_release_ref_for_package_requires_configured_template(tmp_path: Path) -> None:
    write_catalog_pyproject(tmp_path, {"release-ref-template": None})
    config = load_catalog_config(tmp_path)

    with pytest.raises(CatalogConfigError) as exc_info:
        config.release_ref_for_package("acme-dd-widget", "1.2.3")

    assert "release-ref-template" in str(exc_info.value)


def test_missing_catalog_config_errors_deterministically(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "test-workspace"
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(CatalogConfigError) as exc_info:
        load_catalog_config(tmp_path)

    assert "[tool.ddp.catalog]" in str(exc_info.value)
    assert "missing required" in str(exc_info.value)


def test_missing_required_catalog_field_errors_deterministically(tmp_path: Path) -> None:
    write_catalog_pyproject(tmp_path, {"author-name": ""})

    with pytest.raises(CatalogConfigError) as exc_info:
        load_catalog_config(tmp_path)

    message = str(exc_info.value)
    assert "[tool.ddp.catalog].author-name" in message
    assert "non-empty string" in message


def test_malformed_url_errors_deterministically(tmp_path: Path) -> None:
    write_catalog_pyproject(tmp_path, {"package-index-url": "not-a-url"})

    with pytest.raises(CatalogConfigError) as exc_info:
        load_catalog_config(tmp_path)

    message = str(exc_info.value)
    assert "[tool.ddp.catalog].package-index-url" in message
    assert "absolute URL" in message


def test_malformed_data_designer_requirement_errors_deterministically(tmp_path: Path) -> None:
    write_catalog_pyproject(tmp_path, {"default-data-designer-requirement": "requests>=2"})

    with pytest.raises(CatalogConfigError) as exc_info:
        load_catalog_config(tmp_path)

    message = str(exc_info.value)
    assert "[tool.ddp.catalog].default-data-designer-requirement" in message
    assert "data-designer" in message
