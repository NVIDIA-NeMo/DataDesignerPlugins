# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync a JSON plugin catalog from package metadata and plugin objects."""

from __future__ import annotations

import importlib.metadata
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from ddp._repo import find_repo_root, load_toml

CATALOG_SCHEMA_VERSION = 1
REPO_ROOT = find_repo_root()
PLUGINS_DIR = REPO_ROOT / "plugins"
CATALOG_BASE_PATH = REPO_ROOT / "catalog"
PLUGINS_CATALOG_FILENAME = "plugins.json"
PLUGINS_CATALOG_PATH = CATALOG_BASE_PATH / PLUGINS_CATALOG_FILENAME
DATA_DESIGNER_DISTRIBUTION_NAME = "data-designer"
PLUGIN_ENTRY_POINT_GROUP = "data_designer.plugins"


class CatalogError(RuntimeError):
    """Raised when a catalog entry cannot be generated."""


@dataclass(frozen=True)
class CatalogEntry:
    """One plugin entry in the JSON catalog.

    Attributes:
        plugin_package: Python package name from ``[project].name``.
        version: Package version from ``[project].version``.
        name: Runtime DataDesigner plugin name.
        plugin_type: Runtime DataDesigner plugin type value.
        description: Package description from ``[project].description``.
        entry_point_name: Entry point name in the ``data_designer.plugins`` group.
        entry_point_value: Import target registered for the entry point.
        repository_path: Path to the plugin package from the repository root.
        data_designer_requirement: Direct ``data-designer`` dependency
            requirement string.
        data_designer_version_specifier: Version specifier from the package's
            direct ``data-designer`` dependency.
    """

    plugin_package: str
    version: str
    name: str
    plugin_type: str
    description: str
    entry_point_name: str
    entry_point_value: str
    repository_path: str
    data_designer_requirement: str
    data_designer_version_specifier: str


def main() -> None:
    """Generate a JSON catalog of all plugin entry points and print to stdout."""
    try:
        entries = discover_catalog_entries(PLUGINS_DIR)
    except CatalogError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(render_catalog_json(entries), end="")


def sync_catalog() -> Path:
    """Write the repository plugin catalog JSON file.

    Returns:
        Absolute path to the synced catalog file.

    Raises:
        CatalogError: If a catalog entry cannot be generated.
    """
    entries = discover_catalog_entries(PLUGINS_DIR)
    PLUGINS_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLUGINS_CATALOG_PATH.write_text(render_catalog_json(entries), encoding="utf-8")
    return PLUGINS_CATALOG_PATH


def check_catalog() -> bool:
    """Check whether the repository plugin catalog JSON file is current.

    When the catalog is stale, a sibling ``.new`` file is written with the
    expected content so CI can upload it as a drift artifact.

    Returns:
        ``True`` when the catalog is current, otherwise ``False``.

    Raises:
        CatalogError: If a catalog entry cannot be generated.
    """
    entries = discover_catalog_entries(PLUGINS_DIR)
    expected = render_catalog_json(entries)
    new_path = PLUGINS_CATALOG_PATH.with_name(f"{PLUGINS_CATALOG_PATH.name}.new")
    if PLUGINS_CATALOG_PATH.exists() and PLUGINS_CATALOG_PATH.read_text(encoding="utf-8") == expected:
        new_path.unlink(missing_ok=True)
        return True

    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(expected, encoding="utf-8")
    return False


def discover_catalog_entries(plugins_dir: Path) -> list[CatalogEntry]:
    """Discover catalog entries for local plugin packages.

    Args:
        plugins_dir: Repository ``plugins/`` directory.

    Returns:
        Entries sorted by package name, then runtime plugin name.

    Raises:
        CatalogError: If a local entry point is not installed, cannot be loaded,
            or does not load to a DataDesigner ``Plugin`` object.
    """
    entries: list[CatalogEntry] = []
    for toml_path in sorted(plugins_dir.glob("*/pyproject.toml")):
        data = load_toml(toml_path)

        project = data.get("project", {})
        name = project.get("name", toml_path.parent.name)
        version = project.get("version", "unknown")
        description = project.get("description", "")
        data_designer_requirement = data_designer_requirement_for_dependencies(
            package_name=name,
            dependencies=project.get("dependencies", []),
        )
        data_designer_version_specifier = str(Requirement(data_designer_requirement).specifier)

        entry_points = project.get("entry-points", {}).get(PLUGIN_ENTRY_POINT_GROUP, {})
        repository_path = toml_path.parent.relative_to(plugins_dir.parent).as_posix()
        for entry_point_name, entry_point_value in sorted(entry_points.items()):
            entries.append(
                catalog_entry_for_entry_point(
                    package_name=name,
                    version=version,
                    description=description,
                    entry_point_name=entry_point_name,
                    entry_point_value=entry_point_value,
                    repository_path=repository_path,
                    data_designer_requirement=data_designer_requirement,
                    data_designer_version_specifier=data_designer_version_specifier,
                )
            )

    return sorted(entries, key=lambda entry: (entry.plugin_package, entry.name))


def catalog_entry_for_entry_point(
    package_name: str,
    version: str,
    description: str,
    entry_point_name: str,
    entry_point_value: str,
    repository_path: str,
    data_designer_requirement: str,
    data_designer_version_specifier: str,
) -> CatalogEntry:
    """Build a catalog entry from an installed DataDesigner plugin entry point.

    Args:
        package_name: Local plugin package name.
        version: Local plugin package version.
        description: Local plugin package description.
        entry_point_name: Entry point name in the ``data_designer.plugins`` group.
        entry_point_value: Import target registered for the entry point.
        repository_path: Path to the plugin package from the repository root.
        data_designer_requirement: Direct ``data-designer`` dependency
            requirement string.
        data_designer_version_specifier: Version specifier from the package's
            direct ``data-designer`` dependency.

    Returns:
        Catalog entry with runtime plugin metadata.

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

    return CatalogEntry(
        plugin_package=package_name,
        version=version,
        name=plugin_name,
        plugin_type=plugin_type,
        description=description,
        entry_point_name=entry_point_name,
        entry_point_value=entry_point_value,
        repository_path=repository_path,
        data_designer_requirement=data_designer_requirement,
        data_designer_version_specifier=data_designer_version_specifier,
    )


def data_designer_requirement_for_dependencies(package_name: str, dependencies: object) -> str:
    """Return the direct DataDesigner dependency requirement for a package.

    Args:
        package_name: Local plugin package name.
        dependencies: Package dependency requirement strings from
            ``[project].dependencies``.

    Returns:
        Requirement string for the package's direct ``data-designer``
        dependency. The returned requirement must include a version specifier.

    Raises:
        CatalogError: If dependencies are malformed, missing, or ambiguous.
    """
    if not isinstance(dependencies, list) or not all(isinstance(dependency, str) for dependency in dependencies):
        raise CatalogError(f"package {package_name!r} has invalid [project].dependencies; expected a list of strings")

    matching_requirements: list[str] = []
    for dependency in dependencies:
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement as exc:
            raise CatalogError(f"package {package_name!r} has invalid dependency {dependency!r}: {exc}") from exc

        if canonicalize_name(requirement.name) == DATA_DESIGNER_DISTRIBUTION_NAME:
            if not requirement.specifier:
                raise CatalogError(
                    f"package {package_name!r} direct {DATA_DESIGNER_DISTRIBUTION_NAME!r} dependency "
                    "must include a version specifier"
                )
            matching_requirements.append(dependency)

    if not matching_requirements:
        raise CatalogError(
            f"package {package_name!r} must declare a direct {DATA_DESIGNER_DISTRIBUTION_NAME!r} dependency "
            "to publish catalog compatibility metadata"
        )
    if len(matching_requirements) > 1:
        raise CatalogError(
            f"package {package_name!r} declares multiple direct {DATA_DESIGNER_DISTRIBUTION_NAME!r} dependencies"
        )
    return matching_requirements[0]


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
        "run `make sync` before syncing the catalog"
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


def render_catalog_json(entries: list[CatalogEntry]) -> str:
    """Render catalog entries as deterministic JSON.

    Args:
        entries: Catalog entries to render.

    Returns:
        JSON catalog content.
    """
    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "plugins": [
            {
                "name": entry.name,
                "plugin_type": entry.plugin_type,
                "description": entry.description,
                "package": {
                    "name": entry.plugin_package,
                    "version": entry.version,
                    "path": entry.repository_path,
                },
                "entry_point": {
                    "group": PLUGIN_ENTRY_POINT_GROUP,
                    "name": entry.entry_point_name,
                    "value": entry.entry_point_value,
                },
                "compatibility": {
                    "data_designer": {
                        "requirement": entry.data_designer_requirement,
                        "specifier": entry.data_designer_version_specifier,
                    },
                },
            }
            for entry in entries
        ],
    }
    return f"{json.dumps(catalog, indent=2)}\n"


if __name__ == "__main__":
    main()
