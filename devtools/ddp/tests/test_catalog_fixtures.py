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

    plugins = document["plugins"]
    assert isinstance(plugins, list)
    assert len(plugins) >= 5
    plugins_by_name = {plugin["name"]: plugin for plugin in plugins}

    compatible_column = plugins_by_name["compatible-column"]
    assert compatible_column["plugin_type"] == "column-generator"
    assert compatible_column["source"] == {
        "type": "pypi",
        "package": "data-designer-compatible-column",
    }
    assert catalog.install_target_for_source_metadata(
        package_name=compatible_column["package"]["name"],
        version=compatible_column["package"]["version"],
        source=compatible_column["source"],
    ) == catalog.InstallTarget(target="data-designer-compatible-column==0.1.0")

    git_seed_reader = plugins_by_name["compatible-git-seed-reader"]
    assert git_seed_reader["plugin_type"] == "seed-reader"
    assert git_seed_reader["source"] == {
        "type": "git",
        "url": "https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git",
        "ref": "data-designer-git-seed-reader/v0.2.0",
        "subdirectory": "plugins/data-designer-git-seed-reader",
    }
    assert catalog.install_target_for_source_metadata(
        package_name=git_seed_reader["package"]["name"],
        version=git_seed_reader["package"]["version"],
        source=git_seed_reader["source"],
    ) == catalog.InstallTarget(
        target=(
            "data-designer-git-seed-reader @ "
            "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git"
            "@data-designer-git-seed-reader/v0.2.0#subdirectory=plugins/data-designer-git-seed-reader"
        )
    )

    assert plugins_by_name["python312-column"]["compatibility"]["python"]["specifier"] == ">=3.12"
    assert plugins_by_name["future-dd-processor"]["compatibility"]["data_designer"]["specifier"] == ">=999.0"
    assert all(plugin["docs"]["url"].startswith("https://docs.example.test/plugins/") for plugin in plugins)

    multi_plugins = [plugin for plugin in plugins if plugin["package"]["name"] == "data-designer-multi-plugin-package"]
    assert len(multi_plugins) == 2
    assert {plugin["name"] for plugin in multi_plugins} == {"multi-seed-reader", "multi-processor"}
    assert {plugin["plugin_type"] for plugin in multi_plugins} == {"seed-reader", "processor"}
    assert multi_plugins[0]["package"] == multi_plugins[1]["package"]
    assert multi_plugins[0]["source"] == multi_plugins[1]["source"]
    assert multi_plugins[0]["docs"] == multi_plugins[1]["docs"]


@pytest.mark.parametrize(
    ("fixture_name", "message_parts"),
    [
        (
            "schema-v2-invalid-source.json",
            ("invalid 'git' source fields", "ref", "subdirectory"),
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
    plugins = document["plugins"]
    assert isinstance(plugins, list)
    plugin = plugins[0]
    assert isinstance(plugin, dict)
    package = plugin["package"]
    assert isinstance(package, dict)
    package["name"] = "not a valid package name"

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.validate_catalog_document(document)

    message = str(exc_info.value)
    assert "package.name" in message
    assert "valid package name" in message
