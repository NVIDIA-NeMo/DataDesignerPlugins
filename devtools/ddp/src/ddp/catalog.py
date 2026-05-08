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
from urllib.parse import unquote, urlparse

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version

from ddp._repo import find_repo_root, load_toml
from ddp.catalog_config import CatalogConfig, CatalogConfigError, load_catalog_config

CATALOG_SCHEMA_VERSION = 2
REPO_ROOT = find_repo_root()
PLUGINS_DIR = REPO_ROOT / "plugins"
CATALOG_BASE_PATH = REPO_ROOT / "catalog"
PLUGINS_CATALOG_FILENAME = "plugins.json"
PLUGINS_CATALOG_PATH = CATALOG_BASE_PATH / PLUGINS_CATALOG_FILENAME
DATA_DESIGNER_DISTRIBUTION_NAME = "data-designer"
PLUGIN_ENTRY_POINT_GROUP = "data_designer.plugins"
SUPPORTED_PLUGIN_TYPES = {"column-generator", "processor", "seed-reader"}
CATALOG_DOCUMENT_KEYS = {"packages", "schema_version"}
CATALOG_PACKAGE_KEYS = {
    "compatibility",
    "description",
    "docs",
    "install",
    "name",
    "plugins",
}
CATALOG_PLUGIN_KEYS = {
    "entry_point",
    "name",
    "plugin_type",
}
CATALOG_ENTRY_POINT_KEYS = {"group", "name", "value"}
CATALOG_COMPATIBILITY_KEYS = {"data_designer", "python"}
CATALOG_PYTHON_COMPATIBILITY_KEYS = {"specifier"}
CATALOG_DATA_DESIGNER_COMPATIBILITY_KEYS = {"marker", "requirement", "specifier"}
CATALOG_DOCS_KEYS = {"url"}
CATALOG_INSTALL_REQUIRED_KEYS = {"requirement"}
CATALOG_INSTALL_OPTIONAL_KEYS = {"index_url"}


class CatalogError(RuntimeError):
    """Raised when a catalog entry cannot be generated."""


@dataclass(frozen=True)
class CatalogEntry:
    """One runtime plugin entry used to render the JSON catalog.

    Attributes:
        plugin_package: Python package name from ``[project].name``.
        version: Package version from ``[project].version``. This is retained
            for release validation but is not rendered into the public catalog.
        name: Runtime DataDesigner plugin name.
        plugin_type: Runtime DataDesigner plugin type value.
        description: Package description from ``[project].description``.
        entry_point_name: Entry point name in the ``data_designer.plugins`` group.
        entry_point_value: Import target registered for the entry point.
        repository_path: Local repository path used by this repository's
            generator. This is not part of the public catalog contract.
        python_requires: Python version specifier from ``[project].requires-python``.
        data_designer_requirement: Direct ``data-designer`` dependency
            requirement string.
        data_designer_version_specifier: Version specifier from the package's
            direct ``data-designer`` dependency.
        data_designer_marker: Environment marker from the package's direct
            ``data-designer`` dependency, or ``None`` when the requirement is
            unconditionally active.
        install: Install requirement metadata for the package.
        docs_url: Absolute documentation URL for the package.
    """

    plugin_package: str
    version: str
    name: str
    plugin_type: str
    description: str
    entry_point_name: str
    entry_point_value: str
    repository_path: str
    python_requires: str
    data_designer_requirement: str
    data_designer_version_specifier: str
    data_designer_marker: str | None
    install: dict[str, object]
    docs_url: str


@dataclass(frozen=True)
class InstallTarget:
    """Concrete package install target derived from catalog install metadata.

    Attributes:
        target: Requirement string or local path to pass to the installer.
        index_url: Optional Python package index URL required for resolution.
    """

    target: str
    index_url: str | None = None


def main() -> None:
    """Generate a JSON catalog of all plugin entry points and print to stdout."""
    try:
        entries = discover_catalog_entries(PLUGINS_DIR)
        output = render_catalog_json(entries)
    except CatalogError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(output, end="")


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


def validate_catalog_document(document: object) -> None:
    """Validate one catalog JSON document without importing plugins.

    Args:
        document: Decoded JSON value to validate.

    Raises:
        CatalogError: If the document does not match the catalog contract.
    """
    catalog_document = required_catalog_object("catalog document", document, CATALOG_DOCUMENT_KEYS)
    schema_version = catalog_document["schema_version"]
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise CatalogError(f"unsupported catalog schema_version {schema_version!r}; expected {CATALOG_SCHEMA_VERSION}")

    packages = catalog_document["packages"]
    if not isinstance(packages, list):
        raise CatalogError("catalog document has invalid packages; expected a list")

    entries = [
        entry
        for index, raw_package in enumerate(packages)
        for entry in catalog_entries_for_catalog_package(raw_package, index)
    ]
    validate_catalog_entries(entries)


def catalog_entries_for_catalog_package(raw_package: object, index: int) -> list[CatalogEntry]:
    """Return validated catalog entries from one decoded JSON package object.

    Args:
        raw_package: Decoded JSON package value.
        index: Position of the package value in the document's ``packages`` list.

    Returns:
        Catalog entries matching the catalog JSON object.

    Raises:
        CatalogError: If the package object is malformed.
    """
    context = f"catalog packages[{index}]"
    package = required_catalog_object(context, raw_package, CATALOG_PACKAGE_KEYS)
    compatibility = required_catalog_object(
        f"{context}.compatibility",
        package["compatibility"],
        CATALOG_COMPATIBILITY_KEYS,
    )
    python_compatibility = required_catalog_object(
        f"{context}.compatibility.python",
        compatibility["python"],
        CATALOG_PYTHON_COMPATIBILITY_KEYS,
    )
    data_designer_compatibility = required_catalog_object(
        f"{context}.compatibility.data_designer",
        compatibility["data_designer"],
        CATALOG_DATA_DESIGNER_COMPATIBILITY_KEYS,
    )
    install = required_catalog_object(f"{context}.install", package["install"])
    docs = required_catalog_object(f"{context}.docs", package["docs"], CATALOG_DOCS_KEYS)

    package_name = catalog_package_name(f"{context}.name", package["name"])
    description = required_catalog_string(f"{context}.description", package["description"])
    data_designer_requirement, data_designer_specifier, data_designer_marker = catalog_data_designer_compatibility(
        package_name=package_name,
        context=f"{context}.compatibility.data_designer",
        compatibility=data_designer_compatibility,
    )
    python_requires = catalog_version_specifier(
        package_name=package_name,
        context=f"{context}.compatibility.python.specifier",
        value=python_compatibility["specifier"],
    )
    docs_url = catalog_http_url(f"{context}.docs.url", docs["url"])
    validate_install_metadata(package_name, install)

    plugins = package["plugins"]
    if not isinstance(plugins, list) or not plugins:
        raise CatalogError(f"{context}.plugins is invalid; expected a non-empty list")

    return [
        catalog_entry_for_catalog_plugin(
            raw_plugin=raw_plugin,
            context=f"{context}.plugins[{plugin_index}]",
            package_name=package_name,
            description=description,
            python_requires=python_requires,
            data_designer_requirement=data_designer_requirement,
            data_designer_version_specifier=data_designer_specifier,
            data_designer_marker=data_designer_marker,
            install=install,
            docs_url=docs_url,
        )
        for plugin_index, raw_plugin in enumerate(plugins)
    ]


def catalog_entry_for_catalog_plugin(
    raw_plugin: object,
    context: str,
    package_name: str,
    description: str,
    python_requires: str,
    data_designer_requirement: str,
    data_designer_version_specifier: str,
    data_designer_marker: str | None,
    install: dict[str, object],
    docs_url: str,
) -> CatalogEntry:
    """Return a validated catalog entry from one decoded JSON plugin object.

    Args:
        raw_plugin: Decoded JSON plugin value.
        context: Human-readable object path used in error messages.
        package_name: Parent package distribution name.
        description: Parent package description.
        python_requires: Parent package Python compatibility specifier.
        data_designer_requirement: Parent package Data Designer dependency.
        data_designer_version_specifier: Parent package Data Designer version
            specifier.
        data_designer_marker: Parent package Data Designer environment marker.
        install: Parent package install requirement metadata.
        docs_url: Parent package documentation URL.

    Returns:
        Catalog entry matching the catalog JSON object.
    """
    plugin = required_catalog_object(context, raw_plugin, CATALOG_PLUGIN_KEYS)
    entry_point = required_catalog_object(f"{context}.entry_point", plugin["entry_point"], CATALOG_ENTRY_POINT_KEYS)
    plugin_type = required_catalog_plugin_type(context, plugin["plugin_type"])
    entry_point_group = required_catalog_string(f"{context}.entry_point.group", entry_point["group"])
    if entry_point_group != PLUGIN_ENTRY_POINT_GROUP:
        raise CatalogError(
            f"{context}.entry_point.group {entry_point_group!r} is invalid; expected {PLUGIN_ENTRY_POINT_GROUP!r}"
        )

    return CatalogEntry(
        plugin_package=package_name,
        version="",
        name=required_catalog_string(f"{context}.name", plugin["name"]),
        plugin_type=plugin_type,
        description=description,
        entry_point_name=required_catalog_string(f"{context}.entry_point.name", entry_point["name"]),
        entry_point_value=required_catalog_string(f"{context}.entry_point.value", entry_point["value"]),
        repository_path="",
        python_requires=python_requires,
        data_designer_requirement=data_designer_requirement,
        data_designer_version_specifier=data_designer_version_specifier,
        data_designer_marker=data_designer_marker,
        install=install,
        docs_url=docs_url,
    )


def required_catalog_object(
    context: str,
    value: object,
    expected_keys: set[str] | None = None,
) -> dict[str, object]:
    """Return a validated JSON object.

    Args:
        context: Human-readable object path used in error messages.
        value: Decoded JSON value.
        expected_keys: Optional exact key set for the object.

    Returns:
        JSON object as a dictionary.

    Raises:
        CatalogError: If the value is not an object or has unexpected keys.
    """
    if not isinstance(value, dict):
        raise CatalogError(f"{context} is invalid; expected an object")
    if expected_keys is not None:
        validate_catalog_object_keys(context, value, expected_keys)
    return value


def validate_catalog_object_keys(context: str, value: dict[str, object], expected_keys: set[str]) -> None:
    """Validate that a catalog JSON object has exactly the expected fields.

    Args:
        context: Human-readable object path used in error messages.
        value: JSON object to validate.
        expected_keys: Exact field names allowed for the object.

    Raises:
        CatalogError: If the object has missing or extra fields.
    """
    keys = set(value)
    if keys != expected_keys:
        raise CatalogError(
            f"{context} has invalid fields; expected {{{format_catalog_keys(expected_keys)}}}, "
            f"got {{{format_catalog_keys(keys)}}}"
        )


def required_catalog_string(context: str, value: object) -> str:
    """Return a required non-empty string from a decoded catalog value.

    Args:
        context: Human-readable value path used in error messages.
        value: Decoded JSON value.

    Returns:
        Non-empty string value.

    Raises:
        CatalogError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{context} is invalid; expected a non-empty string")
    return value


def required_catalog_nullable_string(context: str, value: object) -> str | None:
    """Return a required string-or-null field from a decoded catalog value.

    Args:
        context: Human-readable value path used in error messages.
        value: Decoded JSON value.

    Returns:
        String value or ``None``.

    Raises:
        CatalogError: If the value is neither a string nor ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise CatalogError(f"{context} is invalid; expected a string or null")


def catalog_package_name(context: str, value: object) -> str:
    """Return a validated Python distribution package name.

    Args:
        context: Human-readable value path used in error messages.
        value: Decoded JSON value.

    Returns:
        Package name string.

    Raises:
        CatalogError: If the value is not a valid package name.
    """
    package_name = required_catalog_string(context, value)
    try:
        canonicalize_name(package_name, validate=True)
    except InvalidName as exc:
        raise CatalogError(f"{context} {package_name!r} is invalid; expected a valid package name") from exc
    return package_name


def required_catalog_plugin_type(context: str, value: object) -> str:
    """Return a validated catalog plugin type.

    Args:
        context: Human-readable plugin object path used in error messages.
        value: Decoded plugin type value.

    Returns:
        Plugin type string.

    Raises:
        CatalogError: If the plugin type is missing or unsupported.
    """
    plugin_type = required_catalog_string(f"{context}.plugin_type", value)
    if plugin_type not in SUPPORTED_PLUGIN_TYPES:
        raise CatalogError(
            f"{context}.plugin_type {plugin_type!r} is invalid; expected one of "
            f"{format_catalog_choices(SUPPORTED_PLUGIN_TYPES)}"
        )
    return plugin_type


def catalog_version_specifier(package_name: str, context: str, value: object) -> str:
    """Return a validated catalog version specifier string.

    Args:
        package_name: Plugin package distribution name for error context.
        context: Human-readable value path used in error messages.
        value: Decoded JSON value.

    Returns:
        Validated version specifier.

    Raises:
        CatalogError: If the specifier is missing, malformed, or empty.
    """
    raw_specifier = required_catalog_string(context, value)
    try:
        specifier = SpecifierSet(raw_specifier)
    except InvalidSpecifier as exc:
        raise CatalogError(f"package {package_name!r} has invalid {context} {raw_specifier!r}: {exc}") from exc
    if not str(specifier):
        raise CatalogError(f"package {package_name!r} has invalid {context}; expected at least one version specifier")
    return str(specifier)


def catalog_data_designer_compatibility(
    package_name: str,
    context: str,
    compatibility: dict[str, object],
) -> tuple[str, str, str | None]:
    """Return validated Data Designer compatibility metadata.

    Args:
        package_name: Plugin package distribution name for error context.
        context: Human-readable compatibility object path used in errors.
        compatibility: Decoded ``compatibility.data_designer`` object.

    Returns:
        Requirement string, version specifier, and marker.

    Raises:
        CatalogError: If compatibility metadata is malformed or inconsistent.
    """
    requirement_text = required_catalog_string(f"{context}.requirement", compatibility["requirement"])
    try:
        requirement = Requirement(requirement_text)
    except InvalidRequirement as exc:
        raise CatalogError(
            f"package {package_name!r} has invalid {context}.requirement {requirement_text!r}: {exc}"
        ) from exc
    if canonicalize_name(requirement.name) != DATA_DESIGNER_DISTRIBUTION_NAME:
        raise CatalogError(
            f"package {package_name!r} has invalid {context}.requirement {requirement_text!r}; "
            f"expected a {DATA_DESIGNER_DISTRIBUTION_NAME!r} requirement"
        )
    if not requirement.specifier:
        raise CatalogError(f"package {package_name!r} has invalid {context}.requirement; expected a version specifier")

    specifier = catalog_version_specifier(
        package_name=package_name,
        context=f"{context}.specifier",
        value=compatibility["specifier"],
    )
    if specifier != str(requirement.specifier):
        raise CatalogError(
            f"package {package_name!r} has invalid {context}.specifier {specifier!r}; "
            f"expected {str(requirement.specifier)!r} from requirement"
        )

    marker = required_catalog_nullable_string(f"{context}.marker", compatibility["marker"])
    expected_marker = str(requirement.marker) if requirement.marker is not None else None
    if marker != expected_marker:
        raise CatalogError(
            f"package {package_name!r} has invalid {context}.marker {marker!r}; expected {expected_marker!r}"
        )
    return requirement_text, specifier, marker


def catalog_http_url(context: str, value: object) -> str:
    """Return a validated absolute HTTP(S) catalog URL.

    Args:
        context: Human-readable value path used in error messages.
        value: Decoded JSON value.

    Returns:
        Absolute HTTP(S) URL string.

    Raises:
        CatalogError: If the value is not an absolute HTTP(S) URL.
    """
    url = required_catalog_string(context, value)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CatalogError(f"{context} {url!r} is invalid; expected an absolute HTTP(S) URL")
    return url


def format_catalog_keys(keys: set[str]) -> str:
    """Format a set of JSON object keys for an error message.

    Args:
        keys: Key names to format.

    Returns:
        Comma-separated sorted key names.
    """
    return ", ".join(sorted(keys))


def format_catalog_choices(choices: set[str]) -> str:
    """Format a set of choices for an error message.

    Args:
        choices: Choice values to format.

    Returns:
        Comma-separated sorted choice values.
    """
    return ", ".join(repr(choice) for choice in sorted(choices))


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
    catalog_config = catalog_config_for_plugins_dir(plugins_dir)
    entries: list[CatalogEntry] = []
    for toml_path in sorted(plugins_dir.glob("*/pyproject.toml")):
        data = load_toml(toml_path)
        project = project_table_for_pyproject(data, toml_path)

        name = required_project_string(toml_path.parent.name, project, "name")
        version = project_version(toml_path.parent.name, project.get("version"))
        description = optional_project_string(name, project, "description")
        python_requires = python_requires_specifier(name, project.get("requires-python"))
        data_designer_requirement = data_designer_requirement_for_dependencies(
            package_name=name,
            dependencies=project.get("dependencies", []),
        )
        data_designer = Requirement(data_designer_requirement)
        data_designer_version_specifier = str(data_designer.specifier)
        data_designer_marker = str(data_designer.marker) if data_designer.marker is not None else None

        entry_points = data_designer_entry_points(name, project)
        repository_path = toml_path.parent.relative_to(plugins_dir.parent).as_posix()
        install = install_metadata_for_package(
            catalog_config=catalog_config,
            package_name=name,
        )
        docs_url = catalog_config.docs_url_for_package(name)
        for entry_point_name, entry_point_value in sorted(entry_points.items()):
            entries.append(
                catalog_entry_for_entry_point(
                    package_name=name,
                    version=version,
                    description=description,
                    entry_point_name=entry_point_name,
                    entry_point_value=entry_point_value,
                    package_dir=toml_path.parent,
                    repository_path=repository_path,
                    python_requires=python_requires,
                    data_designer_requirement=data_designer_requirement,
                    data_designer_version_specifier=data_designer_version_specifier,
                    data_designer_marker=data_designer_marker,
                    install=install,
                    docs_url=docs_url,
                )
            )

    return sorted(entries, key=lambda entry: (entry.plugin_package, entry.name))


def catalog_config_for_plugins_dir(plugins_dir: Path) -> CatalogConfig:
    """Load catalog metadata for a plugins directory.

    Args:
        plugins_dir: Repository ``plugins/`` directory.

    Returns:
        Validated catalog metadata for the repository containing the plugins.

    Raises:
        CatalogError: If catalog metadata is missing or malformed.
    """
    try:
        return load_catalog_config(plugins_dir.parent)
    except CatalogConfigError as exc:
        raise CatalogError(f"could not load catalog metadata for catalog generation: {exc}") from exc


def install_metadata_for_package(
    catalog_config: CatalogConfig,
    package_name: str,
) -> dict[str, object]:
    """Return validated catalog install metadata for a package.

    Args:
        catalog_config: Repository-level catalog metadata.
        package_name: Plugin package distribution name.

    Returns:
        Validated install metadata.

    Raises:
        CatalogError: If the generated install object is malformed.
    """
    try:
        install = catalog_config.install_metadata_for_package(package_name)
    except CatalogConfigError as exc:
        raise CatalogError(f"could not generate install metadata for package {package_name!r}: {exc}") from exc
    validate_install_metadata(package_name, install)
    return install


def catalog_entry_for_entry_point(
    package_name: str,
    version: str,
    description: str,
    entry_point_name: str,
    entry_point_value: str,
    package_dir: Path,
    repository_path: str,
    python_requires: str,
    data_designer_requirement: str,
    data_designer_version_specifier: str,
    data_designer_marker: str | None,
    install: dict[str, object],
    docs_url: str,
) -> CatalogEntry:
    """Build a catalog entry from an installed DataDesigner plugin entry point.

    Args:
        package_name: Local plugin package name.
        version: Local plugin package version.
        description: Local plugin package description.
        entry_point_name: Entry point name in the ``data_designer.plugins`` group.
        entry_point_value: Import target registered for the entry point.
        package_dir: Local plugin package directory.
        repository_path: Path to the plugin package from the repository root.
        python_requires: Python version specifier from
            ``[project].requires-python``.
        data_designer_requirement: Direct ``data-designer`` dependency
            requirement string.
        data_designer_version_specifier: Version specifier from the package's
            direct ``data-designer`` dependency.
        data_designer_marker: Environment marker from the package's direct
            ``data-designer`` dependency, or ``None`` when the requirement is
            unconditionally active.
        install: Install requirement metadata for the package.
        docs_url: Absolute documentation URL for the package.

    Returns:
        Catalog entry with runtime plugin metadata.

    Raises:
        CatalogError: If plugin metadata cannot be loaded or read.
    """
    plugin = load_plugin_from_entry_point(
        package_name=package_name,
        entry_point_name=entry_point_name,
        entry_point_value=entry_point_value,
        package_dir=package_dir,
    )
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
        python_requires=python_requires,
        data_designer_requirement=data_designer_requirement,
        data_designer_version_specifier=data_designer_version_specifier,
        data_designer_marker=data_designer_marker,
        install=install,
        docs_url=docs_url,
    )


def project_table_for_pyproject(data: dict[str, Any], toml_path: Path) -> dict[str, Any]:
    """Return the validated ``[project]`` table from a plugin ``pyproject.toml``.

    Args:
        data: Parsed ``pyproject.toml`` content.
        toml_path: Path to the parsed ``pyproject.toml``.

    Returns:
        The ``[project]`` table.

    Raises:
        CatalogError: If the table is missing or malformed.
    """
    project = data.get("project")
    if not isinstance(project, dict):
        raise CatalogError(f"package at {toml_path.parent.as_posix()!r} has invalid [project] table")
    return project


def required_project_string(package_name: str, project: dict[str, Any], key: str) -> str:
    """Return a required non-empty string value from ``[project]``.

    Args:
        package_name: Local plugin package name or directory name.
        project: Parsed ``[project]`` table.
        key: Project metadata key.

    Returns:
        Project metadata string value.

    Raises:
        CatalogError: If the value is missing or not a non-empty string.
    """
    value = project.get(key)
    if not isinstance(value, str) or not value:
        raise CatalogError(f"package {package_name!r} has invalid [project].{key}; expected a non-empty string")
    return value


def optional_project_string(package_name: str, project: dict[str, Any], key: str) -> str:
    """Return an optional string value from ``[project]``.

    Args:
        package_name: Local plugin package name.
        project: Parsed ``[project]`` table.
        key: Project metadata key.

    Returns:
        Project metadata string value, or ``""`` when omitted.

    Raises:
        CatalogError: If the value is present but not a string.
    """
    value = project.get(key, "")
    if not isinstance(value, str):
        raise CatalogError(f"package {package_name!r} has invalid [project].{key}; expected a string")
    return value


def project_version(package_name: str, version: object) -> str:
    """Return a validated PEP 440 package version.

    Args:
        package_name: Local plugin package name or directory name.
        version: Raw ``[project].version`` value.

    Returns:
        Canonical PEP 440 package version.

    Raises:
        CatalogError: If the version is missing, not a string, or not PEP 440.
    """
    if not isinstance(version, str) or not version:
        raise CatalogError(f"package {package_name!r} has invalid [project].version; expected a non-empty string")
    try:
        return str(Version(version))
    except InvalidVersion as exc:
        raise CatalogError(f"package {package_name!r} has invalid [project].version {version!r}: {exc}") from exc


def python_requires_specifier(package_name: str, requires_python: object) -> str:
    """Return a validated Python compatibility specifier.

    Args:
        package_name: Local plugin package name.
        requires_python: Raw ``[project].requires-python`` value.

    Returns:
        Python version specifier string.

    Raises:
        CatalogError: If the specifier is missing, malformed, or empty.
    """
    if not isinstance(requires_python, str) or not requires_python:
        raise CatalogError(
            f"package {package_name!r} has invalid [project].requires-python; expected a non-empty string"
        )
    try:
        specifier = SpecifierSet(requires_python)
    except InvalidSpecifier as exc:
        raise CatalogError(
            f"package {package_name!r} has invalid [project].requires-python {requires_python!r}: {exc}"
        ) from exc
    if not str(specifier):
        raise CatalogError(
            f"package {package_name!r} has invalid [project].requires-python; expected at least one version specifier"
        )
    return str(specifier)


def data_designer_entry_points(package_name: str, project: dict[str, Any]) -> dict[str, str]:
    """Return validated DataDesigner plugin entry points.

    Args:
        package_name: Local plugin package name.
        project: Parsed ``[project]`` table.

    Returns:
        Entry point mapping from entry point names to import targets.

    Raises:
        CatalogError: If the entry point table is missing, empty, or malformed.
    """
    entry_points = project.get("entry-points")
    if not isinstance(entry_points, dict):
        raise CatalogError(f"package {package_name!r} must declare [project.entry-points.{PLUGIN_ENTRY_POINT_GROUP!r}]")

    plugin_entry_points = entry_points.get(PLUGIN_ENTRY_POINT_GROUP)
    if not isinstance(plugin_entry_points, dict) or not plugin_entry_points:
        raise CatalogError(
            f"package {package_name!r} must declare at least one [project.entry-points.{PLUGIN_ENTRY_POINT_GROUP!r}]"
        )

    for entry_point_name, entry_point_value in plugin_entry_points.items():
        if not isinstance(entry_point_name, str) or not entry_point_name:
            raise CatalogError(
                f"package {package_name!r} has invalid {PLUGIN_ENTRY_POINT_GROUP!r} entry point name "
                f"{entry_point_name!r}; expected a non-empty string"
            )
        if not isinstance(entry_point_value, str) or not entry_point_value:
            raise CatalogError(
                f"package {package_name!r} entry point {entry_point_name!r} has invalid value "
                f"{entry_point_value!r}; expected a non-empty string"
            )

    return plugin_entry_points


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


def load_plugin_from_entry_point(
    package_name: str,
    entry_point_name: str,
    entry_point_value: str,
    package_dir: Path,
) -> Any:
    """Load and validate an installed DataDesigner plugin entry point.

    Args:
        package_name: Local plugin package name.
        entry_point_name: Entry point name in the ``data_designer.plugins`` group.
        entry_point_value: Expected import target from the local
            ``pyproject.toml``.
        package_dir: Local plugin package directory.

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

    entry_point = find_installed_entry_point(
        package_name=package_name,
        entry_point_name=entry_point_name,
        entry_point_value=entry_point_value,
        package_dir=package_dir,
    )
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


def find_installed_entry_point(
    package_name: str,
    entry_point_name: str,
    entry_point_value: str,
    package_dir: Path,
) -> importlib.metadata.EntryPoint:
    """Find an installed entry point owned by a local package.

    Args:
        package_name: Local plugin package name.
        entry_point_name: Entry point name in the ``data_designer.plugins`` group.
        entry_point_value: Expected import target from the local
            ``pyproject.toml``.
        package_dir: Local plugin package directory.

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
            validate_installed_entry_point(
                package_name=package_name,
                entry_point=entry_point,
                entry_point_value=entry_point_value,
                package_dir=package_dir,
            )
            return entry_point

    raise CatalogError(
        f"package {package_name!r} entry point {entry_point_name!r} is not installed; "
        "run `make sync` before syncing the catalog"
    )


def validate_installed_entry_point(
    package_name: str,
    entry_point: importlib.metadata.EntryPoint,
    entry_point_value: str,
    package_dir: Path,
) -> None:
    """Validate that an installed entry point matches local package metadata.

    Args:
        package_name: Local plugin package name.
        entry_point: Installed entry point selected by package and name.
        entry_point_value: Expected import target from the local
            ``pyproject.toml``.
        package_dir: Local plugin package directory.

    Raises:
        CatalogError: If the installed entry point target or source path does
            not match the local plugin package.
    """
    if entry_point.value != entry_point_value:
        raise CatalogError(
            f"package {package_name!r} entry point {entry_point.name!r} is stale; installed target "
            f"{entry_point.value!r} does not match pyproject target {entry_point_value!r}. Run `make sync`."
        )

    source_path = entry_point_distribution_source_path(entry_point)
    expected_path = package_dir.resolve()
    if source_path != expected_path:
        raise CatalogError(
            f"package {package_name!r} entry point {entry_point.name!r} is installed from "
            f"{source_path.as_posix() if source_path is not None else 'an unknown source'}, expected "
            f"{expected_path.as_posix()}. Run `make sync`."
        )


def entry_point_distribution_source_path(entry_point: importlib.metadata.EntryPoint) -> Path | None:
    """Return the editable source path for an installed entry point.

    Args:
        entry_point: Installed entry point.

    Returns:
        Source path from ``direct_url.json`` when the entry point distribution
        was installed from a local file URL, otherwise ``None``.
    """
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return None

    for distribution_file in distribution.files or ():
        if not str(distribution_file).endswith("direct_url.json"):
            continue
        direct_url_path = distribution.locate_file(distribution_file)
        try:
            direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        url = direct_url.get("url")
        if not isinstance(url, str):
            return None
        parsed = urlparse(url)
        if parsed.scheme != "file":
            return None
        return Path(unquote(parsed.path)).resolve()

    return None


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


def validate_install_metadata(package_name: str, install: object) -> None:
    """Validate one catalog install object.

    Args:
        package_name: Plugin package distribution name.
        install: Install metadata object to validate.

    Raises:
        CatalogError: If install metadata is malformed or inconsistent with
            package metadata.
    """
    if not isinstance(install, dict):
        raise CatalogError(f"package {package_name!r} has invalid install; expected an object")
    validate_install_keys(package_name, install)

    requirement_text = required_install_string(package_name, install, "requirement")
    try:
        requirement = Requirement(requirement_text)
    except InvalidRequirement as exc:
        raise CatalogError(
            f"package {package_name!r} has invalid install.requirement {requirement_text!r}: {exc}"
        ) from exc

    if canonicalize_name(requirement.name) != canonicalize_name(package_name):
        raise CatalogError(
            f"package {package_name!r} has invalid install.requirement {requirement_text!r}; "
            f"expected a requirement for {package_name!r}"
        )

    index_url = optional_install_string(package_name, install, "index_url")
    if index_url is not None:
        catalog_http_url(f"package {package_name!r} install.index_url", index_url)


def install_target_for_install_metadata(package_name: str, install: object) -> InstallTarget:
    """Derive the default package install target from catalog install metadata.

    Args:
        package_name: Plugin package distribution name.
        install: Install metadata object to derive from.

    Returns:
        Concrete install target and optional index URL.

    Raises:
        CatalogError: If install metadata is malformed.
    """
    validate_install_metadata(package_name, install)
    if not isinstance(install, dict):
        raise CatalogError(f"package {package_name!r} has invalid install; expected an object")
    return InstallTarget(
        target=required_install_string(package_name, install, "requirement"),
        index_url=optional_install_string(package_name, install, "index_url"),
    )


def validate_install_keys(package_name: str, install: dict[str, object]) -> None:
    """Validate fields in a catalog install object.

    Args:
        package_name: Plugin package distribution name.
        install: Install metadata object to validate.

    Raises:
        CatalogError: If required fields are missing or unknown fields exist.
    """
    keys = set(install)
    missing_keys = CATALOG_INSTALL_REQUIRED_KEYS - keys
    extra_keys = keys - CATALOG_INSTALL_REQUIRED_KEYS - CATALOG_INSTALL_OPTIONAL_KEYS
    if missing_keys or extra_keys:
        expected = ", ".join(sorted(CATALOG_INSTALL_REQUIRED_KEYS))
        expected = f"{expected}; optional {{{', '.join(sorted(CATALOG_INSTALL_OPTIONAL_KEYS))}}}"
        actual = ", ".join(sorted(keys))
        raise CatalogError(
            f"package {package_name!r} has invalid install fields; expected {{{expected}}}, got {{{actual}}}"
        )


def required_install_string(package_name: str, install: dict[str, object], key: str) -> str:
    """Return a required non-empty string from an install object.

    Args:
        package_name: Plugin package distribution name.
        install: Install metadata object to read.
        key: Install field to read.

    Returns:
        Non-empty string value.

    Raises:
        CatalogError: If the field is missing or not a non-empty string.
    """
    value = install.get(key)
    if not isinstance(value, str) or not value:
        raise CatalogError(f"package {package_name!r} has invalid install field {key!r}; expected a non-empty string")
    return value


def optional_install_string(package_name: str, install: dict[str, object], key: str) -> str | None:
    """Return an optional non-empty string from an install object.

    Args:
        package_name: Plugin package distribution name.
        install: Install metadata object to read.
        key: Install field to read.

    Returns:
        Non-empty string value, or ``None`` when omitted.

    Raises:
        CatalogError: If the field is present but not a non-empty string.
    """
    if key not in install:
        return None
    return required_install_string(package_name, install, key)


def validate_unique_runtime_plugin_names(entries: list[CatalogEntry]) -> None:
    """Validate that runtime plugin names are unique within the catalog.

    Args:
        entries: Catalog entries to validate.

    Raises:
        CatalogError: If two entries share the same runtime plugin name.
    """
    seen: dict[str, CatalogEntry] = {}
    for entry in entries:
        previous = seen.get(entry.name)
        if previous is not None:
            raise CatalogError(
                f"duplicate runtime plugin name {entry.name!r} from "
                f"{previous.plugin_package!r} entry point {previous.entry_point_name!r} and "
                f"{entry.plugin_package!r} entry point {entry.entry_point_name!r}"
            )
        seen[entry.name] = entry


def validate_catalog_entries(entries: list[CatalogEntry]) -> None:
    """Validate catalog entries before JSON rendering.

    Args:
        entries: Catalog entries to validate.

    Raises:
        CatalogError: If an entry violates the catalog generation contract.
    """
    validate_unique_runtime_plugin_names(entries)
    validate_catalog_package_consistency(entries)
    for entry in entries:
        validate_install_metadata(entry.plugin_package, entry.install)


def validate_catalog_package_consistency(entries: list[CatalogEntry]) -> None:
    """Validate that entries sharing one package agree on package metadata.

    Args:
        entries: Catalog entries to validate.

    Raises:
        CatalogError: If multiple runtime plugin entries disagree about their
            shared package metadata.
    """
    seen: dict[str, CatalogEntry] = {}
    for entry in entries:
        previous = seen.get(entry.plugin_package)
        if previous is None:
            seen[entry.plugin_package] = entry
            continue

        fields = {
            "description": (previous.description, entry.description),
            "python_requires": (previous.python_requires, entry.python_requires),
            "data_designer_requirement": (previous.data_designer_requirement, entry.data_designer_requirement),
            "data_designer_version_specifier": (
                previous.data_designer_version_specifier,
                entry.data_designer_version_specifier,
            ),
            "data_designer_marker": (previous.data_designer_marker, entry.data_designer_marker),
            "install": (previous.install, entry.install),
            "docs_url": (previous.docs_url, entry.docs_url),
        }
        for field, (previous_value, current_value) in fields.items():
            if previous_value != current_value:
                raise CatalogError(
                    f"package {entry.plugin_package!r} has inconsistent catalog {field}: "
                    f"{previous_value!r} and {current_value!r}"
                )


def catalog_entries_by_package(entries: list[CatalogEntry]) -> dict[str, list[CatalogEntry]]:
    """Group catalog entries by package name in deterministic order.

    Args:
        entries: Catalog entries to group.

    Returns:
        Mapping from package name to runtime plugin entries.
    """
    grouped: dict[str, list[CatalogEntry]] = {}
    for entry in sorted(entries, key=lambda item: (item.plugin_package, item.name)):
        grouped.setdefault(entry.plugin_package, []).append(entry)
    return grouped


def render_catalog_package(entries: list[CatalogEntry]) -> dict[str, object]:
    """Render one package object for the catalog.

    Args:
        entries: Runtime plugin entries for a single package.

    Returns:
        JSON-serializable package object.
    """
    entry = entries[0]
    return {
        "name": entry.plugin_package,
        "description": entry.description,
        "install": entry.install,
        "compatibility": {
            "python": {
                "specifier": entry.python_requires,
            },
            "data_designer": {
                "requirement": entry.data_designer_requirement,
                "specifier": entry.data_designer_version_specifier,
                "marker": entry.data_designer_marker,
            },
        },
        "docs": {
            "url": entry.docs_url,
        },
        "plugins": [
            {
                "name": plugin.name,
                "plugin_type": plugin.plugin_type,
                "entry_point": {
                    "group": PLUGIN_ENTRY_POINT_GROUP,
                    "name": plugin.entry_point_name,
                    "value": plugin.entry_point_value,
                },
            }
            for plugin in entries
        ],
    }


def render_catalog_json(entries: list[CatalogEntry]) -> str:
    """Render catalog entries as deterministic JSON.

    Args:
        entries: Catalog entries to render.

    Returns:
        JSON catalog content.
    """
    validate_catalog_entries(entries)
    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "packages": [
            render_catalog_package(package_entries) for package_entries in catalog_entries_by_package(entries).values()
        ],
    }
    return f"{json.dumps(catalog, indent=2)}\n"


if __name__ == "__main__":
    main()
