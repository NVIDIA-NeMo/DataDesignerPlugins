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
        self.calls: list[tuple[str, str, str, Path]] = []

    def __call__(
        self,
        package_name: str,
        entry_point_name: str,
        entry_point_value: str,
        package_dir: Path,
    ) -> FakePlugin:
        """Load a fake plugin by entry point name.

        Args:
            package_name: Local plugin package name.
            entry_point_name: Entry point name in the ``data_designer.plugins`` group.
            entry_point_value: Entry point import target.
            package_dir: Local plugin package directory.

        Returns:
            Fake plugin object.
        """
        self.calls.append((package_name, entry_point_name, entry_point_value, package_dir))
        return self.plugins[entry_point_name]


class FakeEntryPoint:
    """Entry point stand-in for installed metadata validation tests."""

    def __init__(self, name: str, value: str) -> None:
        """Initialize the fake entry point.

        Args:
            name: Entry point name.
            value: Entry point import target.
        """
        self.name = name
        self.value = value


def write_catalog_pyproject(root: Path, overrides: dict[str, str | None] | None = None) -> None:
    """Write root catalog metadata for catalog tests.

    Args:
        root: Temporary repository root.
        overrides: Optional field values to override in ``[tool.ddp.catalog]``.
    """
    values = {
        "catalog-url": "https://docs.example.test/ddp/catalog/plugins.json",
        "repository-url": "https://git.example.test/acme/dd-plugins",
        "repository-git-url": "https://git.example.test/acme/dd-plugins.git",
        "docs-base-url": "https://docs.example.test/ddp/",
        "package-prefix": "data-designer-",
        "package-index-url": "https://docs.example.test/ddp/simple/",
        "package-assets-url": "https://git.example.test/acme/dd-plugins/releases/download/ddp-package-assets/",
        "package-assets-release-tag": "ddp-package-assets",
        "release-ref-template": "{package}/v{version}",
        "default-data-designer-requirement": "data-designer>=0.5.7",
        "author-name": "ACME Labs",
    }
    values.update(overrides or {})
    lines = [
        "[project]",
        'name = "test-workspace"',
        "",
        "[tool.ddp.catalog]",
    ]
    lines.extend(f'{key} = "{value}"' for key, value in values.items() if value is not None)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plugin_pyproject(
    plugins_dir: Path,
    package_name: str,
    version: str,
    description: str,
    entry_points: dict[str, str] | None,
    dependencies: list[str] | None = None,
    requires_python: str = ">=3.10",
    catalog_overrides: dict[str, str | None] | None = None,
) -> None:
    """Write a minimal plugin pyproject for catalog tests.

    Args:
        plugins_dir: Temporary ``plugins/`` directory.
        package_name: Package name for ``[project].name``.
        version: Package version for ``[project].version``.
        description: Package description for ``[project].description``.
        entry_points: Entry points for ``data_designer.plugins``.
        dependencies: Requirement strings for ``[project].dependencies``.
        requires_python: Python compatibility specifier.
        catalog_overrides: Optional root catalog metadata overrides.
    """
    write_catalog_pyproject(plugins_dir.parent, catalog_overrides)
    plugin_dir = plugins_dir / package_name
    plugin_dir.mkdir(parents=True)
    dependencies = dependencies or ["data-designer>=0.5.7"]
    dependencies_toml = "[" + ", ".join(f'"{dependency}"' for dependency in dependencies) + "]"
    entry_point_section = ""
    if entry_points is not None:
        entry_point_lines = "\n".join(f'{name} = "{value}"' for name, value in entry_points.items())
        entry_point_section = textwrap.dedent(
            f"""

            [project.entry-points."data_designer.plugins"]
            {entry_point_lines}
            """
        )
    pyproject = textwrap.dedent(
        f"""
        [project]
        name = "{package_name}"
        version = "{version}"
        description = "{description}"
        requires-python = "{requires_python}"
        dependencies = {dependencies_toml}
        {entry_point_section}
        """
    ).lstrip()
    (plugin_dir / "pyproject.toml").write_text(pyproject, encoding="utf-8")


def fake_catalog_entry(
    plugin_name: str,
    package_name: str = "data-designer-example",
    entry_point_name: str = "runtime-entry",
    install: dict[str, object] | None = None,
) -> catalog.CatalogEntry:
    """Create a catalog entry for render-time validation tests.

    Args:
        plugin_name: Runtime plugin name.
        package_name: Plugin package distribution name.
        entry_point_name: Entry point name.
        install: Optional install metadata override.

    Returns:
        Catalog entry with valid default metadata.
    """
    return catalog.CatalogEntry(
        plugin_package=package_name,
        version="0.2.0",
        name=plugin_name,
        plugin_type="column-generator",
        description="Package description",
        entry_point_name=entry_point_name,
        entry_point_value="example.plugin:plugin",
        repository_path=f"plugins/{package_name}",
        python_requires=">=3.10",
        data_designer_requirement="data-designer>=0.5.7",
        data_designer_version_specifier=">=0.5.7",
        data_designer_marker=None,
        install=install
        or {
            "requirement": package_name,
            "index_url": "https://docs.example.test/ddp/simple/",
        },
        docs_url=f"https://docs.example.test/ddp/plugins/{package_name}/",
    )


def test_main_produces_json_catalog() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        catalog.main()
    output = json.loads(buf.getvalue())
    assert output["schema_version"] == 2
    assert isinstance(output["packages"], list)


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
        ("data-designer-multi", "a-entry", "example.plugin:a_plugin", plugins_dir / "data-designer-multi"),
        ("data-designer-multi", "z-entry", "example.plugin:z_plugin", plugins_dir / "data-designer-multi"),
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
            python_requires=">=3.10",
            data_designer_requirement="data-designer>=0.5.7",
            data_designer_version_specifier=">=0.5.7",
            data_designer_marker=None,
            install={
                "requirement": "data-designer-multi",
                "index_url": "https://docs.example.test/ddp/simple/",
            },
            docs_url="https://docs.example.test/ddp/plugins/data-designer-multi/",
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
            python_requires=">=3.10",
            data_designer_requirement="data-designer>=0.5.7",
            data_designer_version_specifier=">=0.5.7",
            data_designer_marker=None,
            install={
                "requirement": "data-designer-multi",
                "index_url": "https://docs.example.test/ddp/simple/",
            },
            docs_url="https://docs.example.test/ddp/plugins/data-designer-multi/",
        ),
    ]


def test_render_catalog_json_outputs_plugin_compatibility_contract() -> None:
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
                python_requires=">=3.10",
                data_designer_requirement='data-designer>=0.5.7,<0.6; python_version >= "3.10"',
                data_designer_version_specifier=">=0.5.7,<0.6",
                data_designer_marker='python_version >= "3.10"',
                install={
                    "requirement": "data-designer-example",
                    "index_url": "https://docs.example.test/ddp/simple/",
                },
                docs_url="https://docs.example.test/ddp/plugins/data-designer-example/",
            )
        ]
    )
    data = json.loads(output)

    assert data == {
        "schema_version": 2,
        "packages": [
            {
                "description": "Package description",
                "name": "data-designer-example",
                "install": {
                    "requirement": "data-designer-example",
                    "index_url": "https://docs.example.test/ddp/simple/",
                },
                "compatibility": {
                    "python": {
                        "specifier": ">=3.10",
                    },
                    "data_designer": {
                        "requirement": 'data-designer>=0.5.7,<0.6; python_version >= "3.10"',
                        "specifier": ">=0.5.7,<0.6",
                        "marker": 'python_version >= "3.10"',
                    },
                },
                "docs": {
                    "url": "https://docs.example.test/ddp/plugins/data-designer-example/",
                },
                "plugins": [
                    {
                        "name": "runtime-name",
                        "plugin_type": "column-generator",
                        "entry_point": {
                            "group": "data_designer.plugins",
                            "name": "runtime-entry",
                            "value": "example.plugin:plugin",
                        },
                    }
                ],
            }
        ],
    }


def test_render_catalog_json_keeps_multi_entry_package_install_and_docs_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugins_dir = tmp_path / "plugins"
    write_plugin_pyproject(
        plugins_dir=plugins_dir,
        package_name="data-designer-multi",
        version="1.2.3",
        description="Package-level description",
        entry_points={
            "first-entry": "example.plugin:first_plugin",
            "second-entry": "example.plugin:second_plugin",
        },
    )
    loader = FakePluginLoader(
        {
            "first-entry": FakePlugin(name="first-runtime-name", plugin_type="seed-reader"),
            "second-entry": FakePlugin(name="second-runtime-name", plugin_type="processor"),
        }
    )
    monkeypatch.setattr(catalog, "load_plugin_from_entry_point", loader)

    output = json.loads(catalog.render_catalog_json(catalog.discover_catalog_entries(plugins_dir)))

    [package] = output["packages"]
    assert package["name"] == "data-designer-multi"
    assert package["install"] == {
        "requirement": "data-designer-multi",
        "index_url": "https://docs.example.test/ddp/simple/",
    }
    assert package["docs"] == {
        "url": "https://docs.example.test/ddp/plugins/data-designer-multi/",
    }
    assert [plugin["name"] for plugin in package["plugins"]] == ["first-runtime-name", "second-runtime-name"]


def test_discover_catalog_entries_reuses_one_install_object_for_multi_plugin_package(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugins_dir = tmp_path / "plugins"
    write_plugin_pyproject(
        plugins_dir=plugins_dir,
        package_name="data-designer-multi",
        version="1.2.3",
        description="Package-level description",
        entry_points={
            "first-entry": "example.plugin:first_plugin",
            "second-entry": "example.plugin:second_plugin",
        },
    )
    loader = FakePluginLoader(
        {
            "first-entry": FakePlugin(name="first-runtime-name", plugin_type="seed-reader"),
            "second-entry": FakePlugin(name="second-runtime-name", plugin_type="processor"),
        }
    )
    monkeypatch.setattr(catalog, "load_plugin_from_entry_point", loader)

    entries = catalog.discover_catalog_entries(plugins_dir)

    assert entries[0].install is entries[1].install


def test_install_target_for_install_metadata_uses_requirement_and_index_url() -> None:
    assert catalog.install_target_for_install_metadata(
        package_name="data-designer-example",
        install={
            "requirement": "data-designer-example",
            "index_url": "https://docs.example.test/ddp/simple/",
        },
    ) == catalog.InstallTarget(
        target="data-designer-example",
        index_url="https://docs.example.test/ddp/simple/",
    )


def test_install_target_allows_direct_reference_without_index_url() -> None:
    assert catalog.install_target_for_install_metadata(
        package_name="data-designer-example",
        install={
            "requirement": (
                "data-designer-example @ https://packages.example.test/data_designer_example-0.2.0-py3-none-any.whl"
            ),
        },
    ) == catalog.InstallTarget(
        target="data-designer-example @ https://packages.example.test/data_designer_example-0.2.0-py3-none-any.whl",
    )


def test_install_metadata_rejects_mismatched_requirement_name() -> None:
    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.validate_install_metadata(
            package_name="data-designer-example",
            install={"requirement": "data-designer-other==0.2.0"},
        )

    assert "expected a requirement for 'data-designer-example'" in str(exc_info.value)


def test_install_metadata_allows_resolver_managed_index_requirement() -> None:
    catalog.validate_install_metadata(
        package_name="data-designer-example",
        install={
            "requirement": "data-designer-example>=0.2.0",
            "index_url": "https://docs.example.test/ddp/simple/",
        },
    )


def test_render_catalog_json_errors_on_duplicate_runtime_plugin_names() -> None:
    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.render_catalog_json(
            [
                fake_catalog_entry(plugin_name="duplicate-runtime", entry_point_name="first-entry"),
                fake_catalog_entry(
                    plugin_name="duplicate-runtime",
                    package_name="data-designer-other",
                    entry_point_name="second-entry",
                ),
            ]
        )

    message = str(exc_info.value)
    assert "duplicate runtime plugin name" in message
    assert "duplicate-runtime" in message
    assert "first-entry" in message
    assert "second-entry" in message


def test_render_catalog_json_rejects_inconsistent_package_metadata() -> None:
    entry = fake_catalog_entry(plugin_name="runtime-name")
    inconsistent_entry = catalog.CatalogEntry(
        plugin_package=entry.plugin_package,
        version=entry.version,
        name="second-runtime-name",
        plugin_type=entry.plugin_type,
        description="Different package description",
        entry_point_name="second-entry",
        entry_point_value=entry.entry_point_value,
        repository_path=entry.repository_path,
        python_requires=entry.python_requires,
        data_designer_requirement=entry.data_designer_requirement,
        data_designer_version_specifier=entry.data_designer_version_specifier,
        data_designer_marker=entry.data_designer_marker,
        install=entry.install,
        docs_url=entry.docs_url,
    )

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.render_catalog_json([entry, inconsistent_entry])

    message = str(exc_info.value)
    assert "inconsistent catalog description" in message


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
    assert json.loads(output_path.read_text(encoding="utf-8"))["packages"][0]["plugins"][0]["name"] == "runtime-name"

    output_path.write_text("{}\n", encoding="utf-8")
    assert not catalog.check_catalog()
    assert (tmp_path / "catalog" / "plugins.json.new").is_file()


def test_missing_installed_entry_point_error_names_package_and_entry_point() -> None:
    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.find_installed_entry_point(
            package_name="data-designer-ddp-test-missing",
            entry_point_name="missing-entry",
            entry_point_value="missing.module:plugin",
            package_dir=Path("plugins/data-designer-ddp-test-missing"),
        )

    message = str(exc_info.value)
    assert "data-designer-ddp-test-missing" in message
    assert "missing-entry" in message


def test_missing_entry_point_group_errors(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    write_plugin_pyproject(
        plugins_dir=plugins_dir,
        package_name="data-designer-missing-entry-points",
        version="0.2.0",
        description="Package description",
        entry_points=None,
    )

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.discover_catalog_entries(plugins_dir)

    message = str(exc_info.value)
    assert "data-designer-missing-entry-points" in message
    assert catalog.PLUGIN_ENTRY_POINT_GROUP in message


def test_malformed_entry_point_group_errors() -> None:
    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.data_designer_entry_points(
            package_name="data-designer-malformed-entry-points",
            project={"entry-points": {catalog.PLUGIN_ENTRY_POINT_GROUP: {"runtime-entry": 42}}},
        )

    message = str(exc_info.value)
    assert "data-designer-malformed-entry-points" in message
    assert "runtime-entry" in message
    assert "expected a non-empty string" in message


def test_invalid_project_version_errors(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    write_plugin_pyproject(
        plugins_dir=plugins_dir,
        package_name="data-designer-invalid-version",
        version="unknown",
        description="Package description",
        entry_points={"runtime-entry": "example.plugin:plugin"},
    )

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.discover_catalog_entries(plugins_dir)

    message = str(exc_info.value)
    assert "data-designer-invalid-version" in message
    assert "[project].version" in message


def test_invalid_python_requires_errors(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    write_plugin_pyproject(
        plugins_dir=plugins_dir,
        package_name="data-designer-invalid-python",
        version="0.2.0",
        description="Package description",
        entry_points={"runtime-entry": "example.plugin:plugin"},
        requires_python="not a specifier",
    )

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.discover_catalog_entries(plugins_dir)

    message = str(exc_info.value)
    assert "data-designer-invalid-python" in message
    assert "requires-python" in message


def test_stale_installed_entry_point_target_errors(monkeypatch, tmp_path: Path) -> None:
    package_dir = tmp_path / "plugins" / "data-designer-example"
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(catalog, "entry_point_distribution_source_path", lambda _entry_point: package_dir)

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.validate_installed_entry_point(
            package_name="data-designer-example",
            entry_point=FakeEntryPoint("runtime-entry", "stale.module:plugin"),
            entry_point_value="example.plugin:plugin",
            package_dir=package_dir,
        )

    message = str(exc_info.value)
    assert "data-designer-example" in message
    assert "stale" in message
    assert "example.plugin:plugin" in message


def test_stale_installed_entry_point_source_errors(monkeypatch, tmp_path: Path) -> None:
    package_dir = tmp_path / "plugins" / "data-designer-example"
    package_dir.mkdir(parents=True)
    stale_dir = tmp_path / "stale" / "data-designer-example"
    stale_dir.mkdir(parents=True)
    monkeypatch.setattr(catalog, "entry_point_distribution_source_path", lambda _entry_point: stale_dir)

    with pytest.raises(catalog.CatalogError) as exc_info:
        catalog.validate_installed_entry_point(
            package_name="data-designer-example",
            entry_point=FakeEntryPoint("runtime-entry", "example.plugin:plugin"),
            entry_point_value="example.plugin:plugin",
            package_dir=package_dir,
        )

    message = str(exc_info.value)
    assert "data-designer-example" in message
    assert stale_dir.as_posix() in message
    assert package_dir.as_posix() in message


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
        "description": "Template Data Designer plugin — text transform column generator",
        "name": "data-designer-template",
        "install": {
            "requirement": "data-designer-template",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
        "compatibility": {
            "python": {
                "specifier": ">=3.10",
            },
            "data_designer": {
                "requirement": "data-designer>=0.5.7",
                "specifier": ">=0.5.7",
                "marker": None,
            },
        },
        "docs": {
            "url": "https://nvidia-nemo.github.io/DataDesignerPlugins/plugins/data-designer-template/",
        },
        "plugins": [
            {
                "name": "text-transform",
                "plugin_type": "column-generator",
                "entry_point": {
                    "group": "data_designer.plugins",
                    "name": "text-transform",
                    "value": "data_designer_template.plugin:plugin",
                },
            }
        ],
    } in output["packages"]


def test_checked_in_nvidia_catalog_uses_static_package_index() -> None:
    output = json.loads(catalog.PLUGINS_CATALOG_PATH.read_text(encoding="utf-8"))

    assert {
        package["install"]["index_url"] for package in output["packages"] if isinstance(package.get("install"), dict)
    } == {"https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"}
