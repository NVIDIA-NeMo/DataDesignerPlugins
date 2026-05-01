# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ddp.catalog."""

from __future__ import annotations

import io
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
) -> None:
    """Write a minimal plugin pyproject for catalog tests.

    Args:
        plugins_dir: Temporary ``plugins/`` directory.
        package_name: Package name for ``[project].name``.
        version: Package version for ``[project].version``.
        description: Package description for ``[project].description``.
        entry_points: Entry points for ``data_designer.plugins``.
    """
    plugin_dir = plugins_dir / package_name
    plugin_dir.mkdir(parents=True)
    entry_point_lines = "\n".join(f'{name} = "{value}"' for name, value in entry_points.items())
    pyproject = textwrap.dedent(
        f"""
        [project]
        name = "{package_name}"
        version = "{version}"
        description = "{description}"

        [project.entry-points."data_designer.plugins"]
        {entry_point_lines}
        """
    ).lstrip()
    (plugin_dir / "pyproject.toml").write_text(pyproject, encoding="utf-8")


def test_main_produces_markdown_table() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        catalog.main()
    output = buf.getvalue()
    assert "# Plugin Catalog" in output
    assert "| Plugin Package | Version | Name | Type | Description |" in output


def test_discover_catalog_rows_uses_entry_point_runtime_metadata(monkeypatch, tmp_path: Path) -> None:
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

    rows = catalog.discover_catalog_rows(plugins_dir)

    assert loader.calls == [
        ("data-designer-multi", "a-entry"),
        ("data-designer-multi", "z-entry"),
    ]
    assert rows == [
        catalog.CatalogRow(
            plugin_package="data-designer-multi",
            version="1.2.3",
            name="a-runtime-name",
            plugin_type="seed-reader",
            description="Package-level description",
        ),
        catalog.CatalogRow(
            plugin_package="data-designer-multi",
            version="1.2.3",
            name="z-runtime-name",
            plugin_type="processor",
            description="Package-level description",
        ),
    ]


def test_render_catalog_outputs_plugin_entry_point_rows() -> None:
    output = catalog.render_catalog(
        [
            catalog.CatalogRow(
                plugin_package="data-designer-example",
                version="0.2.0",
                name="runtime-name",
                plugin_type="column-generator",
                description="Package description",
            )
        ]
    )

    assert "| data-designer-example | 0.2.0 | `runtime-name` | `column-generator` | Package description |" in output


def test_missing_installed_entry_point_error_names_package_and_entry_point() -> None:
    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.find_installed_entry_point("data-designer-ddp-test-missing", "missing-entry")

    message = str(exc_info.value)
    assert "data-designer-ddp-test-missing" in message
    assert "missing-entry" in message


def test_main_includes_template_plugin() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        catalog.main()
    output = buf.getvalue()
    assert "data-designer-template" in output
    assert "`text-transform`" in output
    assert "`column-generator`" in output
