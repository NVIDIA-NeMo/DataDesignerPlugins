# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for checked-in schema v2 catalog fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddp import catalog

CATALOG_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "catalogs"


def load_catalog_fixture(name: str) -> dict[str, object]:
    """Load a catalog fixture by file name.

    Args:
        name: Fixture file name under ``fixtures/catalogs``.

    Returns:
        Decoded JSON catalog document.
    """
    return json.loads((CATALOG_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_schema_v2_valid_fixture_exercises_consumer_contract() -> None:
    document = load_catalog_fixture("schema-v2-valid.json")

    catalog.validate_catalog_document(document)

    packages = document["packages"]
    assert isinstance(packages, list)
    assert len(packages) >= 5
    packages_by_name = {package["name"]: package for package in packages}
    plugins_by_name = {plugin["name"]: (package, plugin) for package in packages for plugin in package["plugins"]}

    compatible_package, compatible_column = plugins_by_name["compatible-column"]
    assert compatible_column["plugin_type"] == "column-generator"
    assert compatible_package["install"] == {
        "requirement": "data-designer-compatible-column==0.1.0",
        "index_url": "https://docs.example.test/simple/",
    }
    assert catalog.install_target_for_install_metadata(
        package_name=compatible_package["name"],
        version=compatible_package["version"],
        install=compatible_package["install"],
    ) == catalog.InstallTarget(
        target="data-designer-compatible-column==0.1.0",
        index_url="https://docs.example.test/simple/",
    )

    git_package, git_seed_reader = plugins_by_name["compatible-git-seed-reader"]
    assert git_seed_reader["plugin_type"] == "seed-reader"
    assert git_package["install"] == {
        "requirement": (
            "data-designer-git-seed-reader @ "
            "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git"
            "@data-designer-git-seed-reader/v0.2.0#subdirectory=plugins/data-designer-git-seed-reader"
        )
    }
    assert catalog.install_target_for_install_metadata(
        package_name=git_package["name"],
        version=git_package["version"],
        install=git_package["install"],
    ) == catalog.InstallTarget(
        target=(
            "data-designer-git-seed-reader @ "
            "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git"
            "@data-designer-git-seed-reader/v0.2.0#subdirectory=plugins/data-designer-git-seed-reader"
        )
    )

    url_package, url_processor = plugins_by_name["compatible-url-processor"]
    assert url_processor["plugin_type"] == "processor"
    assert catalog.install_target_for_install_metadata(
        package_name=url_package["name"],
        version=url_package["version"],
        install=url_package["install"],
    ) == catalog.InstallTarget(
        target=(
            "data-designer-url-processor @ "
            "https://packages.example.test/data_designer_url_processor-0.2.1-py3-none-any.whl"
        )
    )

    assert packages_by_name["data-designer-python312-column"]["compatibility"]["python"]["specifier"] == ">=3.12"
    assert (
        packages_by_name["data-designer-future-dd-processor"]["compatibility"]["data_designer"]["specifier"]
        == ">=999.0"
    )
    assert all(package["docs"]["url"].startswith("https://docs.example.test/plugins/") for package in packages)

    multi_package = packages_by_name["data-designer-multi-plugin-package"]
    multi_plugins = multi_package["plugins"]
    assert {plugin["name"] for plugin in multi_plugins} == {"multi-seed-reader", "multi-processor"}
    assert {plugin["plugin_type"] for plugin in multi_plugins} == {"seed-reader", "processor"}
    assert multi_package["install"] == {
        "requirement": "data-designer-multi-plugin-package==1.4.0",
        "index_url": "https://docs.example.test/simple/",
    }


@pytest.mark.parametrize(
    ("fixture_name", "message_parts"),
    [
        (
            "schema-v2-invalid-install.json",
            ("install.requirement", "expected a requirement"),
        ),
        (
            "schema-v2-unsupported-version.json",
            ("unsupported catalog schema_version", "999", "expected 2"),
        ),
    ],
)
def test_schema_v2_invalid_fixtures_fail_for_expected_reason(
    fixture_name: str,
    message_parts: tuple[str, ...],
) -> None:
    document = load_catalog_fixture(fixture_name)

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.validate_catalog_document(document)

    message = str(exc_info.value)
    for message_part in message_parts:
        assert message_part in message


def test_catalog_document_rejects_invalid_package_names() -> None:
    document = load_catalog_fixture("schema-v2-valid.json")
    packages = document["packages"]
    assert isinstance(packages, list)
    package = packages[0]
    assert isinstance(package, dict)
    package["name"] = "not a valid package name"

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.validate_catalog_document(document)

    message = str(exc_info.value)
    assert ".name" in message
    assert "valid package name" in message
