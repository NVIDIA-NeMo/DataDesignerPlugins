# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ddp.validate_release."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from ddp._repo import find_repo_root, load_toml
from ddp.validate_release import REQUIRED_FIELDS, main, validate_release

PACKAGE_NAME = "data-designer-example"
PACKAGE_VERSION = "0.1.0"
PACKAGE_DESCRIPTION = "Package description"
DOCS_BASE_URL = "https://docs.example.test/ddp/"


def write_tap_pyproject(
    repo_root: Path,
    release_ref_template: str = "{package}/v{version}",
    docs_base_url: str = DOCS_BASE_URL,
) -> None:
    """Write root tap metadata for release validation tests.

    Args:
        repo_root: Temporary repository root.
        release_ref_template: Release ref template value.
        docs_base_url: Documentation base URL.
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "test-workspace"

            [tool.ddp.tap]
            catalog-url = "https://catalog.example.test/plugins.json"
            repository-url = "https://git.example.test/acme/dd-plugins"
            repository-git-url = "https://git.example.test/acme/dd-plugins.git"
            docs-base-url = "{docs_base_url}"
            package-prefix = "data-designer-"
            package-index-url = "https://docs.example.test/ddp/simple/"
            package-assets-url = "https://git.example.test/acme/dd-plugins/releases/download/ddp-package-assets/"
            package-assets-release-tag = "ddp-package-assets"
            release-ref-template = "{release_ref_template}"
            default-data-designer-requirement = "data-designer>=0.5.7"
            author-name = "ACME Labs"
            """
        ).lstrip(),
        encoding="utf-8",
    )


def write_plugin_pyproject(
    repo_root: Path,
    package_name: str = PACKAGE_NAME,
    version: str = PACKAGE_VERSION,
    entry_points: dict[str, str] | None = None,
    dependencies: list[str] | None = None,
) -> None:
    """Write plugin package metadata for release validation tests.

    Args:
        repo_root: Temporary repository root.
        package_name: Plugin package name.
        version: Plugin package version.
        entry_points: Data Designer entry points.
        dependencies: Package dependencies.
    """
    entry_points = entry_points or {"runtime-entry": "example.plugin:plugin"}
    dependencies = dependencies or ["data-designer>=0.5.7"]
    dependency_list = "[" + ", ".join(f'"{dependency}"' for dependency in dependencies) + "]"
    entry_point_lines = "\n".join(f'{name} = "{value}"' for name, value in entry_points.items())
    plugin_dir = repo_root / "plugins" / package_name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "{package_name}"
            version = "{version}"
            description = "{PACKAGE_DESCRIPTION}"
            requires-python = ">=3.10"
            dependencies = {dependency_list}
            license = "Apache-2.0"
            readme = "README.md"
            authors = [
                {{name = "ACME Labs"}},
            ]

            [project.entry-points."data_designer.plugins"]
            {entry_point_lines}
            """
        ).lstrip(),
        encoding="utf-8",
    )


def write_codeowners(repo_root: Path, package_name: str = PACKAGE_NAME, owners: str = "@acme/platform") -> None:
    """Write a per-plugin CODEOWNERS file.

    Args:
        repo_root: Temporary repository root.
        package_name: Plugin package name.
        owners: Owner token string.
    """
    plugin_dir = repo_root / "plugins" / package_name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "CODEOWNERS").write_text(f"* {owners}\n", encoding="utf-8")


def catalog_entry(
    package_name: str = PACKAGE_NAME,
    version: str = PACKAGE_VERSION,
    entry_point_name: str = "runtime-entry",
    entry_point_value: str = "example.plugin:plugin",
    runtime_name: str | None = None,
    install: dict[str, object] | None = None,
    docs_url: str | None = None,
) -> dict[str, object]:
    """Return a schema v2 catalog package for release validation tests.

    Args:
        package_name: Plugin package name.
        version: Plugin package version.
        entry_point_name: Data Designer entry point name.
        entry_point_value: Data Designer entry point target.
        runtime_name: Runtime plugin name.
        install: Catalog install metadata.
        docs_url: Catalog docs URL.

    Returns:
        Catalog package object.
    """
    requirement = Requirement("data-designer>=0.5.7")
    return {
        "name": package_name,
        "version": version,
        "description": PACKAGE_DESCRIPTION,
        "install": install
        or {
            "requirement": f"{package_name}=={version}",
            "index_url": "https://docs.example.test/ddp/simple/",
        },
        "compatibility": {
            "python": {
                "specifier": ">=3.10",
            },
            "data_designer": {
                "requirement": "data-designer>=0.5.7",
                "specifier": str(requirement.specifier),
                "marker": None,
            },
        },
        "docs": {
            "url": docs_url or f"{DOCS_BASE_URL.rstrip('/')}/plugins/{package_name}/",
        },
        "plugins": [
            {
                "name": runtime_name or entry_point_name,
                "plugin_type": "column-generator",
                "entry_point": {
                    "group": "data_designer.plugins",
                    "name": entry_point_name,
                    "value": entry_point_value,
                },
            }
        ],
    }


def write_catalog(repo_root: Path, entries: list[dict[str, object]]) -> None:
    """Write checked-in catalog JSON for release validation tests.

    Args:
        repo_root: Temporary repository root.
        entries: Catalog package entries.
    """
    catalog_dir = repo_root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "plugins.json").write_text(
        f"{json.dumps({'schema_version': 2, 'packages': entries}, indent=2)}\n",
        encoding="utf-8",
    )


def write_release_repo(
    repo_root: Path,
    entry_points: dict[str, str] | None = None,
    entries: list[dict[str, object]] | None = None,
    codeowners: str = "@acme/platform",
) -> None:
    """Write a complete temporary release validation repository.

    Args:
        repo_root: Temporary repository root.
        entry_points: Data Designer entry points.
        entries: Catalog entries.
        codeowners: Per-plugin owner tokens.
    """
    write_tap_pyproject(repo_root)
    write_plugin_pyproject(repo_root, entry_points=entry_points)
    write_codeowners(repo_root, owners=codeowners)
    write_catalog(repo_root, entries or [catalog_entry()])


def test_valid_single_entry_package_passes(tmp_path: Path) -> None:
    write_release_repo(tmp_path)

    assert validate_release(tmp_path, PACKAGE_NAME, PACKAGE_VERSION) == []


def test_valid_multi_entry_package_passes(tmp_path: Path) -> None:
    entry_points = {
        "first-entry": "example.plugin:first_plugin",
        "second-entry": "example.plugin:second_plugin",
    }
    package = catalog_entry(
        entry_point_name="first-entry",
        entry_point_value="example.plugin:first_plugin",
        runtime_name="first-runtime",
    )
    plugins = package["plugins"]
    assert isinstance(plugins, list)
    plugins.append(
        {
            "name": "second-runtime",
            "plugin_type": "column-generator",
            "entry_point": {
                "group": "data_designer.plugins",
                "name": "second-entry",
                "value": "example.plugin:second_plugin",
            },
        }
    )
    write_release_repo(
        tmp_path,
        entry_points=entry_points,
        entries=[package],
    )

    assert validate_release(tmp_path, PACKAGE_NAME, PACKAGE_VERSION) == []


def test_stale_catalog_entry_fails(tmp_path: Path) -> None:
    write_release_repo(tmp_path, entries=[catalog_entry(version="0.0.9")])

    errors = validate_release(tmp_path, PACKAGE_NAME, PACKAGE_VERSION)

    assert any(".version" in error and "0.0.9" in error for error in errors)


@pytest.mark.parametrize("missing_field", ["docs", "install"])
def test_missing_docs_or_install_fails(tmp_path: Path, missing_field: str) -> None:
    entry = catalog_entry()
    entry.pop(missing_field)
    write_release_repo(tmp_path, entries=[entry])

    errors = validate_release(tmp_path, PACKAGE_NAME, PACKAGE_VERSION)

    assert any(f"missing {missing_field}" in error for error in errors)


def test_mismatched_install_requirement_fails(tmp_path: Path) -> None:
    write_release_repo(
        tmp_path,
        entries=[
            catalog_entry(
                install={
                    "requirement": "data-designer-example==0.0.9",
                    "index_url": "https://docs.example.test/ddp/simple/",
                },
            )
        ],
    )

    errors = validate_release(tmp_path, PACKAGE_NAME, PACKAGE_VERSION)

    assert any("install.requirement" in error and "==0.1.0" in error for error in errors)


def test_direct_reference_install_fails_release_validation(tmp_path: Path) -> None:
    write_release_repo(
        tmp_path,
        entries=[
            catalog_entry(
                install={
                    "requirement": (
                        "data-designer-example @ "
                        "https://packages.example.test/data_designer_example-0.1.0-py3-none-any.whl"
                    ),
                },
            )
        ],
    )

    errors = validate_release(tmp_path, PACKAGE_NAME, PACKAGE_VERSION)

    assert any("install.requirement" in error for error in errors)


def test_email_only_codeowners_fails(tmp_path: Path) -> None:
    write_release_repo(tmp_path, codeowners="release-owner@example.test")

    errors = validate_release(tmp_path, PACKAGE_NAME, PACKAGE_VERSION)

    assert any("GitHub @user or @org/team" in error and "release-owner@example.test" in error for error in errors)


def test_release_ref_template_mismatch_fails(tmp_path: Path) -> None:
    write_release_repo(tmp_path)
    write_tap_pyproject(tmp_path, release_ref_template="release/{package}/{version}")

    errors = validate_release(tmp_path, PACKAGE_NAME, PACKAGE_VERSION)

    assert any("release-ref-template" in error and "release/data-designer-example/0.1.0" in error for error in errors)


def test_valid_template_plugin_passes() -> None:
    """validate_release should pass for the template plugin with its actual version."""
    root = find_repo_root()
    data = load_toml(root / "plugins" / "data-designer-template" / "pyproject.toml")
    version = data["project"]["version"]

    result = main(["data-designer-template", version])
    assert result == 0


def test_wrong_version_fails() -> None:
    result = main(["data-designer-template", "99.99.99"])
    assert result == 1


def test_required_fields_are_present() -> None:
    assert "description" in REQUIRED_FIELDS
    assert "license" in REQUIRED_FIELDS
    assert "readme" in REQUIRED_FIELDS
    assert "authors" in REQUIRED_FIELDS
