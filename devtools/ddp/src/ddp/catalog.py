# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a markdown plugin catalog from package metadata and plugin objects."""

from __future__ import annotations

import importlib.metadata
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ddp._repo import find_repo_root, load_toml

PLUGIN_ENTRY_POINT_GROUP = "data_designer.plugins"


class CatalogError(RuntimeError):
    """Raised when a catalog entry cannot be generated."""


@dataclass(frozen=True)
class CatalogRow:
    """One rendered row in the plugin catalog.

    Attributes:
        plugin_package: Python package name from ``[project].name``.
        version: Package version from ``[project].version``.
        name: Runtime DataDesigner plugin name.
        plugin_type: Runtime DataDesigner plugin type value.
        description: Package description from ``[project].description``.
    """

    plugin_package: str
    version: str
    name: str
    plugin_type: str
    description: str


def main() -> None:
    """Generate a markdown table of all plugin entry points and print to stdout."""
    try:
        rows = discover_catalog_rows(find_repo_root() / "plugins")
    except CatalogError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(render_catalog(rows))


def discover_catalog_rows(plugins_dir: Path) -> list[CatalogRow]:
    """Discover catalog rows for local plugin packages.

    Args:
        plugins_dir: Repository ``plugins/`` directory.

    Returns:
        Rows sorted by package name, then runtime plugin name.

    Raises:
        CatalogError: If a local entry point is not installed, cannot be loaded,
            or does not load to a DataDesigner ``Plugin`` object.
    """
    rows: list[CatalogRow] = []
    for toml_path in sorted(plugins_dir.glob("*/pyproject.toml")):
        data = load_toml(toml_path)

        project = data.get("project", {})
        name = project.get("name", toml_path.parent.name)
        version = project.get("version", "unknown")
        description = project.get("description", "")

        entry_points = project.get("entry-points", {}).get(PLUGIN_ENTRY_POINT_GROUP, {})
        for entry_point_name in sorted(entry_points):
            rows.append(catalog_row_for_entry_point(name, version, description, entry_point_name))

    return sorted(rows, key=lambda row: (row.plugin_package, row.name))


def catalog_row_for_entry_point(
    package_name: str,
    version: str,
    description: str,
    entry_point_name: str,
) -> CatalogRow:
    """Build a catalog row from an installed DataDesigner plugin entry point.

    Args:
        package_name: Local plugin package name.
        version: Local plugin package version.
        description: Local plugin package description.
        entry_point_name: Entry point name in the ``data_designer.plugins`` group.

    Returns:
        Catalog row with runtime plugin metadata.

    Raises:
        CatalogError: If plugin metadata cannot be loaded or read.
    """
    plugin = load_plugin_from_entry_point(package_name, entry_point_name)
    try:
        plugin_name = plugin.name
        plugin_type = plugin.plugin_type.value
    except Exception as exc:
        raise CatalogError(
            f"could not read runtime metadata for package {package_name!r} entry point {entry_point_name!r}: {exc}"
        ) from exc

    if not isinstance(plugin_name, str) or not plugin_name:
        raise CatalogError(
            f"package {package_name!r} entry point {entry_point_name!r} has invalid plugin.name {plugin_name!r}"
        )
    if not isinstance(plugin_type, str) or not plugin_type:
        raise CatalogError(
            f"package {package_name!r} entry point {entry_point_name!r} has invalid plugin.plugin_type.value "
            f"{plugin_type!r}"
        )

    return CatalogRow(
        plugin_package=package_name,
        version=version,
        name=plugin_name,
        plugin_type=plugin_type,
        description=description,
    )


def load_plugin_from_entry_point(package_name: str, entry_point_name: str) -> Any:
    """Load and validate an installed DataDesigner plugin entry point.

    Args:
        package_name: Local plugin package name.
        entry_point_name: Entry point name in the ``data_designer.plugins`` group.

    Returns:
        Loaded DataDesigner ``Plugin`` object.

    Raises:
        CatalogError: If the entry point is missing, fails to load, or returns
            a non-``Plugin`` object.
    """
    try:
        from data_designer.plugins.plugin import Plugin
    except Exception as exc:
        raise CatalogError(
            f"could not import DataDesigner Plugin while loading package {package_name!r} "
            f"entry point {entry_point_name!r}: {exc}"
        ) from exc

    entry_point = find_installed_entry_point(package_name, entry_point_name)
    try:
        plugin = entry_point.load()
    except Exception as exc:
        raise CatalogError(f"could not load package {package_name!r} entry point {entry_point_name!r}: {exc}") from exc

    if not isinstance(plugin, Plugin):
        raise CatalogError(
            f"package {package_name!r} entry point {entry_point_name!r} loaded {type(plugin).__name__}, "
            "expected data_designer.plugins.plugin.Plugin"
        )
    return plugin


def find_installed_entry_point(package_name: str, entry_point_name: str) -> importlib.metadata.EntryPoint:
    """Find an installed entry point owned by a local package.

    Args:
        package_name: Local plugin package name.
        entry_point_name: Entry point name in the ``data_designer.plugins`` group.

    Returns:
        Matching installed entry point.

    Raises:
        CatalogError: If no installed entry point matches the package and name.
    """
    normalized_package_name = normalize_distribution_name(package_name)
    for entry_point in importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP):
        distribution_name = entry_point_distribution_name(entry_point)
        if distribution_name is None:
            continue
        if (
            normalize_distribution_name(distribution_name) == normalized_package_name
            and entry_point.name == entry_point_name
        ):
            return entry_point

    raise CatalogError(
        f"package {package_name!r} entry point {entry_point_name!r} is not installed; "
        "run `make sync` before regenerating the catalog"
    )


def entry_point_distribution_name(entry_point: importlib.metadata.EntryPoint) -> str | None:
    """Return the distribution name that owns an entry point.

    Args:
        entry_point: Installed entry point.

    Returns:
        Owning distribution name, or ``None`` if it cannot be determined.
    """
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return None
    return distribution.metadata.get("Name")


def normalize_distribution_name(name: str) -> str:
    """Normalize a Python distribution name for comparison.

    Args:
        name: Distribution name.

    Returns:
        PEP 503-style normalized distribution name.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def render_catalog(rows: list[CatalogRow]) -> str:
    """Render catalog rows as a markdown table.

    Args:
        rows: Catalog rows to render.

    Returns:
        Markdown catalog content.
    """
    lines = [
        "# Plugin Catalog",
        "",
        "Auto-generated from installed local DataDesigner plugins and package metadata. Do not edit manually.",
        "",
        "| Plugin Package | Version | Name | Type | Description |",
        "|----------------|---------|------|------|-------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row.plugin_package} | {row.version} | `{row.name}` | `{row.plugin_type}` | {row.description} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
