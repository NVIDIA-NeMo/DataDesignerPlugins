# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate plugin metadata and catalog state before a PyPI release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from ddp import catalog
from ddp._repo import find_repo_root, load_toml
from ddp.codeowners import github_release_owners, owner_tokens_for_codeowners_path
from ddp.tap_config import TapConfig, TapConfigError, load_tap_config

REQUIRED_FIELDS = ("description", "license", "readme", "authors")
CATALOG_PATH = Path("catalog") / "plugins.json"


def main(args: list[str] | None = None) -> int:
    """Validate that a plugin package is release-ready.

    Args:
        args: CLI arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 for success, 1 for validation failure).
    """
    parser = argparse.ArgumentParser(
        prog="validate-release",
        description="Validate plugin metadata and generated catalog state before a PyPI release.",
    )
    parser.add_argument("plugin_name", help="Plugin name (e.g. data-designer-my-plugin)")
    parser.add_argument("tag_version", help="Expected version from the git tag")
    parsed = parser.parse_args(args)

    plugin_name = parsed.plugin_name
    tag_version = parsed.tag_version
    errors = validate_release(find_repo_root(), plugin_name, tag_version)

    if errors:
        print(f"Validation failed for {plugin_name}:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Validation passed for {plugin_name} v{tag_version}")
    return 0


def validate_release(repo_root: Path, plugin_name: str, tag_version: str) -> list[str]:
    """Return release validation errors for a plugin package.

    Args:
        repo_root: Repository root containing ``plugins/`` and ``catalog/``.
        plugin_name: Plugin package directory and expected ``[project].name``.
        tag_version: Version parsed from the release tag without the leading
            ``v``.

    Returns:
        Validation error messages. An empty list means the release contract is
        satisfied.
    """
    errors: list[str] = []
    plugin_dir = repo_root / "plugins" / plugin_name
    toml_path = plugin_dir / "pyproject.toml"

    if not plugin_dir.is_dir():
        errors.append(f"plugin directory must exist at plugins/{plugin_name}")
        return errors
    if not toml_path.is_file():
        errors.append(f"{toml_path} not found")
        return errors

    try:
        tap_config = load_tap_config(repo_root)
    except TapConfigError as exc:
        errors.append(str(exc))
        return errors

    data = load_toml(toml_path)
    project = project_table(data, toml_path, errors)
    if project is None:
        return errors

    project_name = release_project_name(plugin_name, project, errors)
    project_version = release_project_version(project_name, project, tag_version, errors)
    validate_tag_version(tag_version, errors)
    validate_required_metadata(project_name, project, errors)
    description = project.get("description")
    python_requires = release_python_requires(project_name, project, errors)
    data_designer_requirement = release_data_designer_requirement(project_name, project, errors)
    entry_points = release_entry_points(project_name, project, errors)
    release_ref = release_ref_for_cli_args(plugin_name, project_name, project_version, tag_version, tap_config, errors)
    validate_release_codeowners(plugin_name, plugin_dir, errors)

    if (
        project_name is None
        or project_version is None
        or python_requires is None
        or data_designer_requirement is None
        or entry_points is None
        or release_ref is None
        or not isinstance(description, str)
        or not description
    ):
        return errors

    errors.extend(
        validate_catalog_for_release(
            repo_root=repo_root,
            project_name=project_name,
            project_version=project_version,
            package_path=f"plugins/{plugin_name}",
            entry_points=entry_points,
            description=description,
            python_requires=python_requires,
            data_designer_requirement=data_designer_requirement,
            release_ref=release_ref,
            tap_config=tap_config,
        )
    )
    return errors


def project_table(data: dict[str, Any], toml_path: Path, errors: list[str]) -> dict[str, Any] | None:
    """Return the validated project table or record an error.

    Args:
        data: Parsed ``pyproject.toml`` data.
        toml_path: Path to the parsed ``pyproject.toml``.
        errors: Error accumulator.

    Returns:
        Parsed project table, or ``None`` when invalid.
    """
    try:
        return catalog.project_table_for_pyproject(data, toml_path)
    except catalog.CatalogError as exc:
        errors.append(str(exc))
        return None


def release_project_name(plugin_name: str, project: dict[str, Any], errors: list[str]) -> str | None:
    """Validate and return ``[project].name``.

    Args:
        plugin_name: Expected plugin package name from the CLI.
        project: Parsed ``[project]`` table.
        errors: Error accumulator.

    Returns:
        Project name, or ``None`` when invalid.
    """
    try:
        project_name = catalog.required_project_string(plugin_name, project, "name")
    except catalog.CatalogError as exc:
        errors.append(str(exc))
        return None

    if project_name != plugin_name:
        errors.append(f"[project].name is {project_name!r}, expected {plugin_name!r}")
    return project_name


def release_project_version(
    project_name: str | None,
    project: dict[str, Any],
    tag_version: str,
    errors: list[str],
) -> str | None:
    """Validate and return ``[project].version``.

    Args:
        project_name: Parsed project name.
        project: Parsed ``[project]`` table.
        tag_version: Expected version parsed from the release tag.
        errors: Error accumulator.

    Returns:
        Project version string, or ``None`` when invalid.
    """
    package_name = project_name or "unknown"
    version = project.get("version")
    try:
        catalog.project_version(package_name, version)
    except catalog.CatalogError as exc:
        errors.append(str(exc))
        return None

    if version != tag_version:
        errors.append(f"[project].version is {version!r}, expected tag version {tag_version!r}")
    return version if isinstance(version, str) else None


def validate_tag_version(tag_version: str, errors: list[str]) -> None:
    """Validate that the tag version is PEP 440.

    Args:
        tag_version: Version parsed from the release tag.
        errors: Error accumulator.
    """
    try:
        Version(tag_version)
    except InvalidVersion as exc:
        errors.append(f"tag version {tag_version!r} is not valid PEP 440: {exc}")


def validate_required_metadata(project_name: str | None, project: dict[str, Any], errors: list[str]) -> None:
    """Validate required PyPI metadata fields.

    Args:
        project_name: Parsed project name.
        project: Parsed ``[project]`` table.
        errors: Error accumulator.
    """
    package_name = project_name or "unknown"
    for field in REQUIRED_FIELDS:
        if field not in project:
            errors.append(f"Required field project.{field} is missing")

    if "description" in project:
        try:
            catalog.required_project_string(package_name, project, "description")
        except catalog.CatalogError as exc:
            errors.append(str(exc))

    if "license" in project and is_empty_metadata_value(project["license"]):
        errors.append("Required field project.license must not be empty")
    if "readme" in project and is_empty_metadata_value(project["readme"]):
        errors.append("Required field project.readme must not be empty")
    if "authors" in project and not is_non_empty_author_list(project["authors"]):
        errors.append("Required field project.authors must be a non-empty array")


def is_empty_metadata_value(value: object) -> bool:
    """Return whether a PEP 621 metadata field value is empty.

    Args:
        value: Metadata field value.

    Returns:
        ``True`` when the value is an empty string, table, or array.
    """
    return value in ("", [], {})


def is_non_empty_author_list(value: object) -> bool:
    """Return whether a PEP 621 authors value is a non-empty array.

    Args:
        value: ``[project].authors`` value.

    Returns:
        ``True`` when the value is a non-empty list.
    """
    return isinstance(value, list) and bool(value)


def release_python_requires(project_name: str | None, project: dict[str, Any], errors: list[str]) -> str | None:
    """Validate and return the package Python requirement.

    Args:
        project_name: Parsed project name.
        project: Parsed ``[project]`` table.
        errors: Error accumulator.

    Returns:
        Normalized Python specifier, or ``None`` when invalid.
    """
    try:
        return catalog.python_requires_specifier(project_name or "unknown", project.get("requires-python"))
    except catalog.CatalogError as exc:
        errors.append(str(exc))
        return None


def release_data_designer_requirement(
    project_name: str | None,
    project: dict[str, Any],
    errors: list[str],
) -> str | None:
    """Validate and return the package's direct Data Designer dependency.

    Args:
        project_name: Parsed project name.
        project: Parsed ``[project]`` table.
        errors: Error accumulator.

    Returns:
        Requirement string, or ``None`` when invalid.
    """
    try:
        return catalog.data_designer_requirement_for_dependencies(
            package_name=project_name or "unknown",
            dependencies=project.get("dependencies", []),
        )
    except catalog.CatalogError as exc:
        errors.append(str(exc))
        return None


def release_entry_points(
    project_name: str | None,
    project: dict[str, Any],
    errors: list[str],
) -> dict[str, str] | None:
    """Validate and return package Data Designer entry points.

    Args:
        project_name: Parsed project name.
        project: Parsed ``[project]`` table.
        errors: Error accumulator.

    Returns:
        Entry point mapping, or ``None`` when invalid.
    """
    try:
        return catalog.data_designer_entry_points(project_name or "unknown", project)
    except catalog.CatalogError as exc:
        errors.append(str(exc))
        return None


def release_ref_for_cli_args(
    plugin_name: str,
    project_name: str | None,
    project_version: str | None,
    tag_version: str,
    tap_config: TapConfig,
    errors: list[str],
) -> str | None:
    """Validate and return the release ref implied by CLI arguments.

    Args:
        plugin_name: Plugin package name parsed from the tag or Makefile.
        project_name: Parsed project name.
        project_version: Parsed project version.
        tag_version: Version parsed from the tag.
        tap_config: Repository tap configuration.
        errors: Error accumulator.

    Returns:
        Expected release ref, or ``None`` when invalid.
    """
    if project_name is None or project_version is None:
        return None

    try:
        expected_release_ref = tap_config.release_ref_for_package(project_name, project_version)
    except TapConfigError as exc:
        errors.append(str(exc))
        return None

    implied_release_ref = f"{plugin_name}/v{tag_version}"
    if implied_release_ref != expected_release_ref:
        errors.append(
            f"release ref implied by CLI arguments is {implied_release_ref!r}, "
            f"expected {expected_release_ref!r} from [tool.ddp.tap].release-ref-template"
        )
    return expected_release_ref


def validate_release_codeowners(plugin_name: str, plugin_dir: Path, errors: list[str]) -> None:
    """Validate release-eligible CODEOWNERS for a plugin.

    Args:
        plugin_name: Plugin package name.
        plugin_dir: Plugin package directory.
        errors: Error accumulator.
    """
    codeowners_path = plugin_dir / "CODEOWNERS"
    if not codeowners_path.is_file():
        errors.append(f"{codeowners_path} is missing; releases require at least one GitHub CODEOWNER")
        return

    owners = owner_tokens_for_codeowners_path(codeowners_path)
    github_owners = github_release_owners(owners)
    if github_owners:
        return

    if owners:
        listed = ", ".join(owners)
        errors.append(
            f"{plugin_name} CODEOWNERS must include at least one GitHub @user or @org/team owner for releases; "
            f"found only non-GitHub owner(s): {listed}"
        )
        return
    errors.append(f"{plugin_name} CODEOWNERS must include at least one GitHub @user or @org/team owner for releases")


def validate_catalog_for_release(
    repo_root: Path,
    project_name: str,
    project_version: str,
    package_path: str,
    entry_points: dict[str, str],
    description: str,
    python_requires: str,
    data_designer_requirement: str,
    release_ref: str,
    tap_config: TapConfig,
) -> list[str]:
    """Validate checked-in schema v2 catalog entries for a release.

    Args:
        repo_root: Repository root.
        project_name: Plugin package name.
        project_version: Plugin package version.
        package_path: Expected repository-relative plugin package path.
        entry_points: Entry points declared by the package.
        description: Package description.
        python_requires: Normalized package Python specifier.
        data_designer_requirement: Direct Data Designer dependency string.
        release_ref: Expected release ref.
        tap_config: Repository tap configuration.

    Returns:
        Validation error messages.
    """
    errors: list[str] = []
    catalog_path = repo_root / CATALOG_PATH
    data = load_catalog_json(catalog_path, errors)
    if data is None:
        return errors

    plugins = catalog_plugins(data, catalog_path, errors)
    if plugins is None:
        return errors

    package_entries = entries_for_package(plugins, project_name)
    if not package_entries:
        errors.append(f"{CATALOG_PATH.as_posix()} has no schema v2 entries for package {project_name!r}")
        return errors

    matching_version_entries = [
        entry for _entry_index, entry in package_entries if catalog_package_version(entry) == project_version
    ]
    if not matching_version_entries:
        errors.append(
            f"{CATALOG_PATH.as_posix()} has no schema v2 entry for package {project_name!r} version {project_version!r}"
        )

    seen_entry_points: dict[str, str] = {}
    expected_docs_url = tap_config.docs_url_for_package(project_name)
    for entry_index, entry in package_entries:
        errors.extend(
            validate_catalog_entry_for_release(
                entry=entry,
                entry_index=entry_index,
                project_name=project_name,
                project_version=project_version,
                package_path=package_path,
                entry_points=entry_points,
                description=description,
                python_requires=python_requires,
                data_designer_requirement=data_designer_requirement,
                release_ref=release_ref,
                expected_docs_url=expected_docs_url,
                seen_entry_points=seen_entry_points,
            )
        )

    errors.extend(validate_catalog_entry_point_coverage(project_name, entry_points, seen_entry_points))
    return errors


def load_catalog_json(catalog_path: Path, errors: list[str]) -> dict[str, Any] | None:
    """Load checked-in catalog JSON.

    Args:
        catalog_path: Path to ``catalog/plugins.json``.
        errors: Error accumulator.

    Returns:
        Parsed catalog object, or ``None`` when invalid.
    """
    if not catalog_path.is_file():
        errors.append(f"{CATALOG_PATH.as_posix()} is missing")
        return None
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{CATALOG_PATH.as_posix()} is not valid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{CATALOG_PATH.as_posix()} must contain a JSON object")
        return None
    return data


def catalog_plugins(data: dict[str, Any], catalog_path: Path, errors: list[str]) -> list[object] | None:
    """Return schema v2 catalog plugin entries.

    Args:
        data: Parsed catalog JSON.
        catalog_path: Path to the catalog for error messages.
        errors: Error accumulator.

    Returns:
        Plugin entry list, or ``None`` when invalid.
    """
    schema_version = data.get("schema_version")
    if schema_version != catalog.CATALOG_SCHEMA_VERSION:
        errors.append(
            f"{catalog_path} has schema_version {schema_version!r}, expected {catalog.CATALOG_SCHEMA_VERSION}"
        )
        return None

    plugins = data.get("plugins")
    if not isinstance(plugins, list):
        errors.append(f"{catalog_path} has invalid plugins field; expected an array")
        return None
    return plugins


def entries_for_package(plugins: list[object], project_name: str) -> list[tuple[int, dict[str, Any]]]:
    """Return catalog entries for a package name.

    Args:
        plugins: Catalog plugin entries.
        project_name: Package name to select.

    Returns:
        ``(index, entry)`` pairs whose ``package.name`` matches.
    """
    entries: list[tuple[int, dict[str, Any]]] = []
    for index, entry in enumerate(plugins):
        if not isinstance(entry, dict):
            continue
        package = entry.get("package")
        if isinstance(package, dict) and package.get("name") == project_name:
            entries.append((index, entry))
    return entries


def catalog_package_version(entry: dict[str, Any]) -> object:
    """Return the package version from a catalog entry.

    Args:
        entry: Catalog plugin entry.

    Returns:
        ``package.version`` value, or ``None`` when absent.
    """
    package = entry.get("package")
    if not isinstance(package, dict):
        return None
    return package.get("version")


def validate_catalog_entry_for_release(
    entry: dict[str, Any],
    entry_index: int,
    project_name: str,
    project_version: str,
    package_path: str,
    entry_points: dict[str, str],
    description: str,
    python_requires: str,
    data_designer_requirement: str,
    release_ref: str,
    expected_docs_url: str,
    seen_entry_points: dict[str, str],
) -> list[str]:
    """Validate one checked-in catalog entry for a releasing package.

    Args:
        entry: Catalog plugin entry.
        entry_index: Entry index within ``plugins``.
        project_name: Plugin package name.
        project_version: Plugin package version.
        package_path: Expected repository-relative plugin package path.
        entry_points: Entry points declared by the package.
        description: Package description.
        python_requires: Normalized package Python specifier.
        data_designer_requirement: Direct Data Designer dependency string.
        release_ref: Expected release ref.
        expected_docs_url: Expected docs URL.
        seen_entry_points: Mutable mapping of seen entry point names to values.

    Returns:
        Validation error messages.
    """
    errors: list[str] = []
    context = f"{CATALOG_PATH.as_posix()} plugins[{entry_index}]"
    validate_catalog_runtime_fields(entry, context, description, errors)

    package = required_catalog_object(entry, "package", context, errors)
    package_path_value = package_path
    if package is not None:
        package_path_value = validate_catalog_package(
            package=package,
            context=context,
            project_name=project_name,
            project_version=project_version,
            package_path=package_path,
            errors=errors,
        )

    entry_point = required_catalog_object(entry, "entry_point", context, errors)
    if entry_point is not None:
        validate_catalog_entry_point(
            entry_point=entry_point,
            context=context,
            entry_points=entry_points,
            seen_entry_points=seen_entry_points,
            errors=errors,
        )

    compatibility = required_catalog_object(entry, "compatibility", context, errors)
    if compatibility is not None:
        validate_catalog_compatibility(
            compatibility=compatibility,
            context=context,
            python_requires=python_requires,
            data_designer_requirement=data_designer_requirement,
            errors=errors,
        )

    source = entry.get("source")
    if "source" not in entry:
        errors.append(f"{context} is missing source")
    else:
        validate_release_source(
            source=source,
            context=context,
            project_name=project_name,
            release_ref=release_ref,
            package_path=package_path_value,
            errors=errors,
        )

    docs = required_catalog_object(entry, "docs", context, errors)
    if docs is not None:
        validate_catalog_docs(docs, context, expected_docs_url, errors)

    return errors


def validate_catalog_runtime_fields(
    entry: dict[str, Any],
    context: str,
    description: str,
    errors: list[str],
) -> None:
    """Validate schema v2 runtime fields that are present in every entry.

    Args:
        entry: Catalog plugin entry.
        context: Error context.
        description: Expected package description.
        errors: Error accumulator.
    """
    required_catalog_string(entry, "name", context, errors)
    required_catalog_string(entry, "plugin_type", context, errors)
    catalog_description = required_catalog_string(entry, "description", context, errors)
    if catalog_description is not None and catalog_description != description:
        errors.append(f"{context} description is stale; expected {description!r}")


def validate_catalog_package(
    package: dict[str, Any],
    context: str,
    project_name: str,
    project_version: str,
    package_path: str,
    errors: list[str],
) -> str:
    """Validate a catalog package object.

    Args:
        package: Catalog ``package`` object.
        context: Error context.
        project_name: Plugin package name.
        project_version: Plugin package version.
        package_path: Expected repository-relative plugin package path.
        errors: Error accumulator.

    Returns:
        Catalog package path value when valid enough to compare to source
        metadata, otherwise the expected package path.
    """
    package_name = required_catalog_string(package, "name", f"{context}.package", errors)
    version = required_catalog_string(package, "version", f"{context}.package", errors)
    path = required_catalog_string(package, "path", f"{context}.package", errors)

    if package_name is not None and package_name != project_name:
        errors.append(f"{context}.package.name is {package_name!r}, expected {project_name!r}")
    if version is not None and version != project_version:
        errors.append(f"{context}.package.version is {version!r}, expected {project_version!r}")
    if path is not None:
        if path != package_path:
            errors.append(f"{context}.package.path is {path!r}, expected {package_path!r}")
        try:
            catalog.validate_package_path(project_name, path, "package.path")
        except catalog.CatalogError as exc:
            errors.append(str(exc))
        return path
    return package_path


def validate_catalog_entry_point(
    entry_point: dict[str, Any],
    context: str,
    entry_points: dict[str, str],
    seen_entry_points: dict[str, str],
    errors: list[str],
) -> None:
    """Validate a catalog entry point object.

    Args:
        entry_point: Catalog ``entry_point`` object.
        context: Error context.
        entry_points: Entry points declared by ``pyproject.toml``.
        seen_entry_points: Mutable mapping of seen entry point names to values.
        errors: Error accumulator.
    """
    group = required_catalog_string(entry_point, "group", f"{context}.entry_point", errors)
    name = required_catalog_string(entry_point, "name", f"{context}.entry_point", errors)
    value = required_catalog_string(entry_point, "value", f"{context}.entry_point", errors)

    if group is not None and group != catalog.PLUGIN_ENTRY_POINT_GROUP:
        errors.append(f"{context}.entry_point.group is {group!r}, expected {catalog.PLUGIN_ENTRY_POINT_GROUP!r}")
    if name is None or value is None:
        return

    seen_entry_points[name] = value
    expected_value = entry_points.get(name)
    if expected_value is None:
        errors.append(f"{context}.entry_point.name {name!r} is not declared by [project.entry-points]")
        return
    if value != expected_value:
        errors.append(f"{context}.entry_point.value for {name!r} is {value!r}, expected {expected_value!r}")


def validate_catalog_compatibility(
    compatibility: dict[str, Any],
    context: str,
    python_requires: str,
    data_designer_requirement: str,
    errors: list[str],
) -> None:
    """Validate catalog compatibility metadata.

    Args:
        compatibility: Catalog ``compatibility`` object.
        context: Error context.
        python_requires: Expected Python specifier.
        data_designer_requirement: Expected Data Designer dependency string.
        errors: Error accumulator.
    """
    python = required_catalog_object(compatibility, "python", f"{context}.compatibility", errors)
    if python is not None:
        validate_catalog_python_compatibility(python, context, python_requires, errors)

    data_designer = required_catalog_object(compatibility, "data_designer", f"{context}.compatibility", errors)
    if data_designer is not None:
        validate_catalog_data_designer_compatibility(
            data_designer=data_designer,
            context=context,
            data_designer_requirement=data_designer_requirement,
            errors=errors,
        )


def validate_catalog_python_compatibility(
    python: dict[str, Any],
    context: str,
    python_requires: str,
    errors: list[str],
) -> None:
    """Validate catalog Python compatibility metadata.

    Args:
        python: Catalog ``compatibility.python`` object.
        context: Error context.
        python_requires: Expected Python specifier.
        errors: Error accumulator.
    """
    specifier = required_catalog_string(python, "specifier", f"{context}.compatibility.python", errors)
    if specifier is None:
        return
    try:
        parsed = SpecifierSet(specifier)
    except InvalidSpecifier as exc:
        errors.append(f"{context}.compatibility.python.specifier is invalid: {exc}")
        return
    if str(parsed) != python_requires:
        errors.append(f"{context}.compatibility.python.specifier is {specifier!r}, expected {python_requires!r}")


def validate_catalog_data_designer_compatibility(
    data_designer: dict[str, Any],
    context: str,
    data_designer_requirement: str,
    errors: list[str],
) -> None:
    """Validate catalog Data Designer compatibility metadata.

    Args:
        data_designer: Catalog ``compatibility.data_designer`` object.
        context: Error context.
        data_designer_requirement: Expected Data Designer dependency string.
        errors: Error accumulator.
    """
    requirement = required_catalog_string(
        data_designer,
        "requirement",
        f"{context}.compatibility.data_designer",
        errors,
    )
    specifier = required_catalog_string(
        data_designer,
        "specifier",
        f"{context}.compatibility.data_designer",
        errors,
    )
    if "marker" not in data_designer:
        errors.append(f"{context}.compatibility.data_designer is missing marker")
        marker = None
    else:
        marker = data_designer["marker"]
        if marker is not None and not isinstance(marker, str):
            errors.append(f"{context}.compatibility.data_designer.marker must be a string or null")

    if requirement is None or specifier is None:
        return

    try:
        parsed = Requirement(requirement)
    except InvalidRequirement as exc:
        errors.append(f"{context}.compatibility.data_designer.requirement is invalid: {exc}")
        return

    expected = Requirement(data_designer_requirement)
    expected_marker = str(expected.marker) if expected.marker is not None else None
    if requirement != data_designer_requirement:
        errors.append(
            f"{context}.compatibility.data_designer.requirement is {requirement!r}, "
            f"expected {data_designer_requirement!r}"
        )
    parsed_specifier = str(parsed.specifier)
    expected_specifier = str(expected.specifier)
    if specifier != parsed_specifier:
        errors.append(
            f"{context}.compatibility.data_designer.specifier is {specifier!r}, "
            f"expected the specifier from requirement {parsed_specifier!r}"
        )
    if specifier != expected_specifier:
        errors.append(
            f"{context}.compatibility.data_designer.specifier is {specifier!r}, expected {expected_specifier!r}"
        )
    if marker != expected_marker:
        errors.append(f"{context}.compatibility.data_designer.marker is {marker!r}, expected {expected_marker!r}")


def validate_release_source(
    source: object,
    context: str,
    project_name: str,
    release_ref: str,
    package_path: str,
    errors: list[str],
) -> None:
    """Validate release-safe catalog source metadata.

    Args:
        source: Catalog ``source`` object.
        context: Error context.
        project_name: Plugin package name.
        release_ref: Expected release ref.
        package_path: Catalog package path.
        errors: Error accumulator.
    """
    try:
        catalog.validate_source_metadata(project_name, source)
    except catalog.CatalogError as exc:
        errors.append(str(exc))
        return

    if not isinstance(source, dict):
        errors.append(f"{context}.source must be an object")
        return

    source_type = source.get("type")
    if source_type == "path":
        errors.append(f"{context}.source.type 'path' cannot be used for release validation")
        return
    if source_type == "git":
        ref = source.get("ref")
        subdirectory = source.get("subdirectory")
        if ref != release_ref:
            errors.append(f"{context}.source.ref is {ref!r}, expected release ref {release_ref!r}")
        if subdirectory != package_path:
            errors.append(f"{context}.source.subdirectory is {subdirectory!r}, expected package.path {package_path!r}")


def validate_catalog_docs(
    docs: dict[str, Any],
    context: str,
    expected_docs_url: str,
    errors: list[str],
) -> None:
    """Validate catalog docs metadata.

    Args:
        docs: Catalog ``docs`` object.
        context: Error context.
        expected_docs_url: Expected docs URL.
        errors: Error accumulator.
    """
    docs_url = required_catalog_string(docs, "url", f"{context}.docs", errors)
    if docs_url is not None and docs_url != expected_docs_url:
        errors.append(f"{context}.docs.url is {docs_url!r}, expected {expected_docs_url!r}")


def validate_catalog_entry_point_coverage(
    project_name: str,
    entry_points: dict[str, str],
    seen_entry_points: dict[str, str],
) -> list[str]:
    """Validate catalog coverage for all package entry points.

    Args:
        project_name: Plugin package name.
        entry_points: Entry points declared by ``pyproject.toml``.
        seen_entry_points: Entry points found in catalog entries for the package.

    Returns:
        Validation error messages.
    """
    errors: list[str] = []
    for entry_point_name in sorted(set(entry_points) - set(seen_entry_points)):
        errors.append(f"{CATALOG_PATH.as_posix()} is missing package {project_name!r} entry point {entry_point_name!r}")
    return errors


def required_catalog_object(
    data: dict[str, Any],
    key: str,
    context: str,
    errors: list[str],
) -> dict[str, Any] | None:
    """Return a required object field from a catalog object.

    Args:
        data: Catalog object.
        key: Field name.
        context: Error context.
        errors: Error accumulator.

    Returns:
        Object value, or ``None`` when invalid.
    """
    if key not in data:
        errors.append(f"{context} is missing {key}")
        return None
    value = data[key]
    if not isinstance(value, dict):
        errors.append(f"{context}.{key} must be an object")
        return None
    return value


def required_catalog_string(
    data: dict[str, Any],
    key: str,
    context: str,
    errors: list[str],
) -> str | None:
    """Return a required non-empty string field from a catalog object.

    Args:
        data: Catalog object.
        key: Field name.
        context: Error context.
        errors: Error accumulator.

    Returns:
        String value, or ``None`` when invalid.
    """
    if key not in data:
        errors.append(f"{context} is missing {key}")
        return None
    value = data[key]
    if not isinstance(value, str) or not value:
        errors.append(f"{context}.{key} must be a non-empty string")
        return None
    return value


if __name__ == "__main__":
    sys.exit(main())
