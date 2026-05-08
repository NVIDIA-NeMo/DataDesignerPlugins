# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate repository-level Data Designer plugin tap metadata."""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from ddp._repo import find_repo_root, load_toml

TAP_CONFIG_PATH = "[tool.ddp.tap]"
URL_FIELDS = ("catalog-url", "repository-url", "docs-base-url", "package-index-url", "package-assets-url")
REQUIRED_FIELDS = (
    "catalog-url",
    "repository-url",
    "docs-base-url",
    "package-prefix",
    "package-index-url",
    "package-assets-url",
    "package-assets-release-tag",
    "default-data-designer-requirement",
    "author-name",
)
DATA_DESIGNER_DISTRIBUTION_NAME = "data-designer"


class TapConfigError(RuntimeError):
    """Raised when repository tap metadata is missing or malformed."""


@dataclass(frozen=True)
class TapConfig:
    """Validated repository-level Data Designer plugin tap metadata.

    Args:
        catalog_url: Default catalog URL advertised for this tap.
        repository_url: Human-facing repository URL.
        repository_git_url: Optional clone/install URL retained for release ref
            validation and human repository metadata.
        docs_base_url: Base documentation URL for this tap.
        package_prefix: Distribution package prefix used by scaffolded plugins.
        package_index_url: Python Simple API index URL for packages released by
            this tap.
        package_assets_url: Package file base URL used by ``dumb-pypi`` rows
            released by this tap.
        package_assets_release_tag: GitHub Release tag that stores package
            files and the mutable package-list metadata asset.
        release_ref_template: Template used to derive per-plugin release refs.
        default_data_designer_requirement: Dependency string used by scaffolds.
        author_name: Default package author used by scaffolds.
    """

    catalog_url: str
    repository_url: str
    repository_git_url: str | None
    docs_base_url: str
    package_prefix: str
    package_index_url: str
    package_assets_url: str
    package_assets_release_tag: str
    release_ref_template: str | None
    default_data_designer_requirement: str
    author_name: str

    def package_name_for_slug(self, slug: str) -> str:
        """Return the distribution package name for a plugin slug.

        Args:
            slug: Kebab-case plugin slug without the configured prefix.

        Returns:
            Distribution package name.
        """
        return f"{self.package_prefix}{slug}"

    def docs_url(self, path: str = "") -> str:
        """Return a documentation URL under the configured docs base.

        Args:
            path: Optional URL path below the configured docs base.

        Returns:
            Absolute documentation URL.
        """
        if not path:
            return f"{self.docs_base_url.rstrip('/')}/"
        return f"{self.docs_base_url.rstrip('/')}/{path.lstrip('/')}"

    def docs_url_for_package(self, package_name: str) -> str:
        """Return the generated plugin documentation URL for a package.

        Args:
            package_name: Plugin distribution package name.

        Returns:
            Absolute plugin documentation URL.
        """
        return self.docs_url(f"plugins/{normalize_docs_slug(package_name)}/")

    def release_ref_for_package(self, package_name: str, version: str) -> str:
        """Return the configured release ref for a package version.

        Args:
            package_name: Plugin distribution package name.
            version: Plugin package version.

        Returns:
            Release ref generated from ``release_ref_template``.
        """
        if self.release_ref_template is None:
            raise TapConfigError("release validation requires [tool.ddp.tap].release-ref-template")
        return self.release_ref_template.format(package=package_name, version=version)

    def install_metadata_for_package(self, package_name: str, version: str) -> dict[str, object]:
        """Return reusable install metadata for a plugin package.

        Args:
            package_name: Plugin distribution package name.
            version: Plugin package version.

        Returns:
            Catalog install metadata using this tap's Simple API index.
        """
        return {
            "requirement": f"{package_name}=={version}",
            "index_url": self.package_index_url,
        }


def load_tap_config(repo_root: Path | None = None) -> TapConfig:
    """Load validated tap metadata from a repository root.

    Args:
        repo_root: Repository root containing ``pyproject.toml``. When omitted,
            the current working directory is searched for the repository root.

    Returns:
        Validated tap configuration.

    Raises:
        TapConfigError: If ``[tool.ddp.tap]`` is missing or malformed.
    """
    root = repo_root or find_repo_root()
    pyproject_path = root / "pyproject.toml"
    data = load_toml(pyproject_path)
    return tap_config_from_pyproject_data(data, pyproject_path)


def tap_config_from_pyproject_data(data: dict[str, Any], pyproject_path: Path) -> TapConfig:
    """Build tap metadata from parsed root ``pyproject.toml`` data.

    Args:
        data: Parsed root ``pyproject.toml`` content.
        pyproject_path: Path used in deterministic error messages.

    Returns:
        Validated tap configuration.

    Raises:
        TapConfigError: If ``[tool.ddp.tap]`` is missing or malformed.
    """
    tool = data.get("tool")
    ddp = tool.get("ddp") if isinstance(tool, dict) else None
    table = ddp.get("tap") if isinstance(ddp, dict) else None
    if not isinstance(table, dict):
        raise TapConfigError(f"{pyproject_path} is missing required {TAP_CONFIG_PATH} table")
    return tap_config_from_table(table, pyproject_path)


def tap_config_from_table(table: dict[str, Any], pyproject_path: Path) -> TapConfig:
    """Build tap metadata from a parsed ``[tool.ddp.tap]`` table.

    Args:
        table: Parsed ``[tool.ddp.tap]`` table.
        pyproject_path: Path used in deterministic error messages.

    Returns:
        Validated tap configuration.

    Raises:
        TapConfigError: If a field is missing or malformed.
    """
    values = {field: required_tap_string(table, field, pyproject_path) for field in REQUIRED_FIELDS}
    for field in URL_FIELDS:
        validate_url(field, values[field], pyproject_path)
    validate_data_designer_requirement(values["default-data-designer-requirement"], pyproject_path)

    repository_git_url = optional_tap_string(table, "repository-git-url", pyproject_path)
    release_ref_template = optional_tap_string(table, "release-ref-template", pyproject_path)

    if repository_git_url is not None:
        validate_url("repository-git-url", repository_git_url, pyproject_path)
    if release_ref_template is not None:
        validate_release_ref_template(release_ref_template, pyproject_path)

    return TapConfig(
        catalog_url=values["catalog-url"],
        repository_url=values["repository-url"],
        repository_git_url=repository_git_url,
        docs_base_url=values["docs-base-url"],
        package_prefix=values["package-prefix"],
        package_index_url=values["package-index-url"],
        package_assets_url=values["package-assets-url"],
        package_assets_release_tag=values["package-assets-release-tag"],
        release_ref_template=release_ref_template,
        default_data_designer_requirement=values["default-data-designer-requirement"],
        author_name=values["author-name"],
    )


def required_tap_string(table: dict[str, Any], field: str, pyproject_path: Path) -> str:
    """Return a required non-empty tap config string.

    Args:
        table: Parsed ``[tool.ddp.tap]`` table.
        field: Field name to read.
        pyproject_path: Path used in deterministic error messages.

    Returns:
        Non-empty string field value.

    Raises:
        TapConfigError: If the field is missing or is not a non-empty string.
    """
    value = table.get(field)
    if not isinstance(value, str) or not value:
        raise TapConfigError(f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.{field}; expected a non-empty string")
    return value


def optional_tap_string(table: dict[str, Any], field: str, pyproject_path: Path) -> str | None:
    """Return an optional non-empty tap config string.

    Args:
        table: Parsed ``[tool.ddp.tap]`` table.
        field: Field name to read.
        pyproject_path: Path used in deterministic error messages.

    Returns:
        Non-empty string field value, or ``None`` when omitted.

    Raises:
        TapConfigError: If the field is present but is not a non-empty string.
    """
    if field not in table:
        return None

    value = table.get(field)
    if not isinstance(value, str) or not value:
        raise TapConfigError(
            f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.{field}; expected a non-empty string when provided"
        )
    return value


def validate_url(field: str, value: str, pyproject_path: Path) -> None:
    """Validate a tap config URL field.

    Args:
        field: Field name being validated.
        value: URL value.
        pyproject_path: Path used in deterministic error messages.

    Raises:
        TapConfigError: If the value is not an absolute URL.
    """
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        raise TapConfigError(f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.{field}; expected an absolute URL")


def validate_release_ref_template(value: str, pyproject_path: Path) -> None:
    """Validate the configured release ref template.

    Args:
        value: Release ref template string.
        pyproject_path: Path used in deterministic error messages.

    Raises:
        TapConfigError: If the template cannot be formatted with package and
            version values, or if it produces an empty ref.
    """
    supported_fields = {"package", "version"}
    field_names = {field_name for _, field_name, _, _ in string.Formatter().parse(value) if field_name}
    unsupported_fields = sorted(field_names - supported_fields)
    if unsupported_fields:
        formatted_fields = ", ".join(repr(field) for field in unsupported_fields)
        raise TapConfigError(
            f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.release-ref-template; "
            f"unsupported field(s): {formatted_fields}"
        )
    try:
        release_ref = value.format(package="data-designer-example", version="0.1.0")
    except (IndexError, KeyError, ValueError) as exc:
        raise TapConfigError(f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.release-ref-template: {exc}") from exc
    if not release_ref:
        raise TapConfigError(
            f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.release-ref-template; expected a non-empty ref"
        )


def validate_data_designer_requirement(value: str, pyproject_path: Path) -> None:
    """Validate the configured scaffold Data Designer dependency.

    Args:
        value: Requirement string.
        pyproject_path: Path used in deterministic error messages.

    Raises:
        TapConfigError: If the requirement is malformed, is not for
        ``data-designer``, or has no version specifier.
    """
    try:
        requirement = Requirement(value)
    except InvalidRequirement as exc:
        raise TapConfigError(
            f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.default-data-designer-requirement: {exc}"
        ) from exc
    if canonicalize_name(requirement.name) != DATA_DESIGNER_DISTRIBUTION_NAME:
        raise TapConfigError(
            f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.default-data-designer-requirement; "
            f"expected a {DATA_DESIGNER_DISTRIBUTION_NAME!r} requirement"
        )
    if not requirement.specifier:
        raise TapConfigError(
            f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.default-data-designer-requirement; "
            "expected a version specifier"
        )


def normalize_docs_slug(package_name: str) -> str:
    """Normalize a package name into the docs URL slug used by generated docs.

    Args:
        package_name: Python package distribution name.

    Returns:
        URL-safe slug suitable for ``docs/plugins/<slug>``.
    """
    slug = "".join(
        character if character.isalnum() or character in "_.-" else "-" for character in package_name.lower()
    )
    slug = slug.strip("-")
    return slug or "plugin"
