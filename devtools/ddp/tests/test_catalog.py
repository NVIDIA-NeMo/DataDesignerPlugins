# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ddp.catalog."""

from __future__ import annotations

import io
import json
import textwrap
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from ddp import catalog


class FakePluginType:
    """Plugin type stand-in with the DataDesigner enum ``value`` interface."""

    def __init__(self, value: str) -> None:
        """Initialize the fake plugin type.

        Args:
            value: Runtime plugin type value.
        """
        self.value = value


class FakePlugin:
    """Plugin stand-in with the runtime catalog metadata interface."""

    def __init__(self, name: str, plugin_type: str) -> None:
        """Initialize the fake plugin.

        Args:
            name: Runtime plugin name.
            plugin_type: Runtime plugin type value.
        """
        self.name = name
        self.plugin_type = FakePluginType(plugin_type)


class FakePluginLoader:
    """Callable fake entry point loader for catalog row tests."""

    def __init__(self, plugins: dict[str, FakePlugin]) -> None:
        """Initialize the fake loader.

        Args:
            plugins: Fake plugins keyed by entry point name.
        """
        self.plugins = plugins
        self.calls: list[tuple[str, str]] = []

    def __call__(self, package_name: str, entry_point_name: str) -> FakePlugin:
        """Load a fake plugin by entry point name.

        Args:
            package_name: Local plugin package name.
            entry_point_name: Entry point name in the ``data_designer.plugins`` group.

        Returns:
            Fake plugin object.
        """
        self.calls.append((package_name, entry_point_name))
        return self.plugins[entry_point_name]


def write_plugin_pyproject(
    plugins_dir: Path,
    package_name: str,
    version: str,
    description: str,
    entry_points: dict[str, str],
    dependencies: list[str] | None = None,
) -> None:
    """Write a minimal plugin pyproject for catalog tests.

    Args:
        plugins_dir: Temporary ``plugins/`` directory.
        package_name: Package name for ``[project].name``.
        version: Package version for ``[project].version``.
        description: Package description for ``[project].description``.
        entry_points: Entry points for ``data_designer.plugins``.
        dependencies: Requirement strings for ``[project].dependencies``.
    """
    plugin_dir = plugins_dir / package_name
    plugin_dir.mkdir(parents=True)
    dependencies = dependencies or ["data-designer>=0.5.7"]
    dependencies_toml = "[" + ", ".join(f'"{dependency}"' for dependency in dependencies) + "]"
    entry_point_lines = "\n".join(f'{name} = "{value}"' for name, value in entry_points.items())
    pyproject = textwrap.dedent(
        f"""
        [project]
        name = "{package_name}"
        version = "{version}"
        description = "{description}"
        dependencies = {dependencies_toml}

        [project.entry-points."data_designer.plugins"]
        {entry_point_lines}
        """
    ).lstrip()
    (plugin_dir / "pyproject.toml").write_text(pyproject, encoding="utf-8")


def test_main_produces_json_catalog() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        catalog.main()
    output = json.loads(buf.getvalue())
    assert output["schema_version"] == 1
    assert isinstance(output["plugins"], list)


def test_discover_catalog_entries_uses_entry_point_runtime_metadata(monkeypatch, tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    write_plugin_pyproject(
        plugins_dir=plugins_dir,
        package_name="data-designer-multi",
        version="1.2.3",
        description="Package-level description",
        entry_points={
            "z-entry": "example.plugin:z_plugin",
            "a-entry": "example.plugin:a_plugin",
        },
    )
    loader = FakePluginLoader(
        {
            "z-entry": FakePlugin(name="z-runtime-name", plugin_type="processor"),
            "a-entry": FakePlugin(name="a-runtime-name", plugin_type="seed-reader"),
        }
    )
    monkeypatch.setattr(catalog, "load_plugin_from_entry_point", loader)

    entries = catalog.discover_catalog_entries(plugins_dir)

    assert loader.calls == [
        ("data-designer-multi", "a-entry"),
        ("data-designer-multi", "z-entry"),
    ]
    assert entries == [
        catalog.CatalogEntry(
            plugin_package="data-designer-multi",
            version="1.2.3",
            name="a-runtime-name",
            plugin_type="seed-reader",
            description="Package-level description",
            entry_point_name="a-entry",
            entry_point_value="example.plugin:a_plugin",
            repository_path="plugins/data-designer-multi",
            data_designer_requirement="data-designer>=0.5.7",
            data_designer_version_specifier=">=0.5.7",
        ),
        catalog.CatalogEntry(
            plugin_package="data-designer-multi",
            version="1.2.3",
            name="z-runtime-name",
            plugin_type="processor",
            description="Package-level description",
            entry_point_name="z-entry",
            entry_point_value="example.plugin:z_plugin",
            repository_path="plugins/data-designer-multi",
            data_designer_requirement="data-designer>=0.5.7",
            data_designer_version_specifier=">=0.5.7",
        ),
    ]


def test_render_catalog_json_outputs_plugin_entry_point_entries() -> None:
    output = catalog.render_catalog_json(
        [
            catalog.CatalogEntry(
                plugin_package="data-designer-example",
                version="0.2.0",
                name="runtime-name",
                plugin_type="column-generator",
                description="Package description",
                entry_point_name="runtime-entry",
                entry_point_value="example.plugin:plugin",
                repository_path="plugins/data-designer-example",
                data_designer_requirement="data-designer>=0.5.7,<0.6",
                data_designer_version_specifier=">=0.5.7,<0.6",
            )
        ]
    )
    data = json.loads(output)

    assert data == {
        "schema_version": 1,
        "plugins": [
            {
                "name": "runtime-name",
                "plugin_type": "column-generator",
                "description": "Package description",
                "package": {
                    "name": "data-designer-example",
                    "version": "0.2.0",
                    "path": "plugins/data-designer-example",
                },
                "entry_point": {
                    "group": "data_designer.plugins",
                    "name": "runtime-entry",
                    "value": "example.plugin:plugin",
                },
                "compatibility": {
                    "data_designer": {
                        "requirement": "data-designer>=0.5.7,<0.6",
                        "specifier": ">=0.5.7,<0.6",
                    },
                },
            }
        ],
    }


def test_sync_and_check_catalog_use_default_repo_path(monkeypatch, tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    catalog_path = tmp_path / "catalog" / "plugins.json"
    write_plugin_pyproject(
        plugins_dir=plugins_dir,
        package_name="data-designer-example",
        version="0.2.0",
        description="Package description",
        entry_points={"runtime-entry": "example.plugin:plugin"},
    )
    loader = FakePluginLoader({"runtime-entry": FakePlugin(name="runtime-name", plugin_type="column-generator")})
    monkeypatch.setattr(catalog, "PLUGINS_DIR", plugins_dir)
    monkeypatch.setattr(catalog, "CATALOG_BASE_PATH", tmp_path / "catalog")
    monkeypatch.setattr(catalog, "PLUGINS_CATALOG_PATH", catalog_path)
    monkeypatch.setattr(catalog, "load_plugin_from_entry_point", loader)

    output_path = catalog.sync_catalog()

    assert output_path == catalog_path
    assert catalog.check_catalog()
    assert json.loads(output_path.read_text(encoding="utf-8"))["plugins"][0]["name"] == "runtime-name"

    output_path.write_text("{}\n", encoding="utf-8")
    assert not catalog.check_catalog()
    assert (tmp_path / "catalog" / "plugins.json.new").is_file()


def test_missing_installed_entry_point_error_names_package_and_entry_point() -> None:
    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.find_installed_entry_point("data-designer-ddp-test-missing", "missing-entry")

    message = str(exc_info.value)
    assert "data-designer-ddp-test-missing" in message
    assert "missing-entry" in message


def test_missing_data_designer_dependency_errors(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    write_plugin_pyproject(
        plugins_dir=plugins_dir,
        package_name="data-designer-missing-dependency",
        version="0.2.0",
        description="Package description",
        entry_points={"runtime-entry": "example.plugin:plugin"},
        dependencies=["requests>=2"],
    )

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.discover_catalog_entries(plugins_dir)

    message = str(exc_info.value)
    assert "data-designer-missing-dependency" in message
    assert "data-designer" in message


def test_malformed_dependencies_errors() -> None:
    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.data_designer_requirement_for_dependencies(
            package_name="data-designer-malformed-dependency",
            dependencies={"data-designer": ">=0.5.7"},
        )

    message = str(exc_info.value)
    assert "data-designer-malformed-dependency" in message
    assert "list of strings" in message


def test_unversioned_data_designer_dependency_errors(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    write_plugin_pyproject(
        plugins_dir=plugins_dir,
        package_name="data-designer-unversioned-dependency",
        version="0.2.0",
        description="Package description",
        entry_points={"runtime-entry": "example.plugin:plugin"},
        dependencies=["data-designer"],
    )

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.discover_catalog_entries(plugins_dir)

    message = str(exc_info.value)
    assert "data-designer-unversioned-dependency" in message
    assert "version specifier" in message


def test_main_includes_template_plugin() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        catalog.main()
    output = json.loads(buf.getvalue())
    assert {
        "name": "text-transform",
        "plugin_type": "column-generator",
        "description": "Template Data Designer plugin — text transform column generator",
        "package": {
            "name": "data-designer-template",
            "version": "0.1.0",
            "path": "plugins/data-designer-template",
        },
        "entry_point": {
            "group": "data_designer.plugins",
            "name": "text-transform",
            "value": "data_designer_template.plugin:plugin",
        },
        "compatibility": {
            "data_designer": {
                "requirement": "data-designer>=0.5.7",
                "specifier": ">=0.5.7",
            },
        },
    } in output["plugins"]
