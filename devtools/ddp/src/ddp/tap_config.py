# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate repository-level Data Designer plugin tap metadata."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from ddp._repo import find_repo_root, load_toml

TAP_CONFIG_PATH = "[tool.ddp.tap]"
DEFAULT_SOURCE_VALUES = ("pypi", "git", "path")
URL_FIELDS = ("catalog-url", "repository-url", "repository-git-url", "docs-base-url")
REQUIRED_FIELDS = (
    "catalog-url",
    "repository-url",
    "repository-git-url",
    "docs-base-url",
    "package-prefix",
    "default-source",
    "release-ref-template",
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
        repository_git_url: Clone/install URL for git source metadata.
        docs_base_url: Base documentation URL for this tap.
        package_prefix: Distribution package prefix used by scaffolded plugins.
        default_source: Default source type for catalog source metadata.
        release_ref_template: Template used to derive per-plugin release refs.
        default_data_designer_requirement: Dependency string used by scaffolds.
        author_name: Default package author used by scaffolds.
    """

    catalog_url: str
    repository_url: str
    repository_git_url: str
    docs_base_url: str
    package_prefix: str
    default_source: Literal["pypi", "git", "path"]
    release_ref_template: str
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
        return self.release_ref_template.format(package=package_name, version=version)

    def source_metadata_for_package(self, package_name: str, version: str, repository_path: str) -> dict[str, object]:
        """Return reusable source metadata for a plugin package.

        Args:
            package_name: Plugin distribution package name.
            version: Plugin package version.
            repository_path: Repository-relative package path.

        Returns:
            Source metadata derived from ``default_source``.
        """
        if self.default_source == "pypi":
            return {
                "type": "pypi",
                "package": package_name,
            }
        if self.default_source == "git":
            return {
                "type": "git",
                "url": self.repository_git_url,
                "ref": self.release_ref_for_package(package_name, version),
                "subdirectory": repository_path,
            }
        return {
            "type": "path",
            "path": repository_path,
            "editable": True,
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
    validate_default_source(values["default-source"], pyproject_path)
    validate_release_ref_template(values["release-ref-template"], pyproject_path)
    validate_data_designer_requirement(values["default-data-designer-requirement"], pyproject_path)

    return TapConfig(
        catalog_url=values["catalog-url"],
        repository_url=values["repository-url"],
        repository_git_url=values["repository-git-url"],
        docs_base_url=values["docs-base-url"],
        package_prefix=values["package-prefix"],
        default_source=cast(Literal["pypi", "git", "path"], values["default-source"]),
        release_ref_template=values["release-ref-template"],
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


def validate_default_source(value: str, pyproject_path: Path) -> None:
    """Validate the configured default source type.

    Args:
        value: Source type value.
        pyproject_path: Path used in deterministic error messages.

    Raises:
        TapConfigError: If the source type is unsupported.
    """
    if value not in DEFAULT_SOURCE_VALUES:
        choices = ", ".join(repr(choice) for choice in DEFAULT_SOURCE_VALUES)
        raise TapConfigError(
            f"{pyproject_path} has invalid {TAP_CONFIG_PATH}.default-source; expected one of {choices}"
        )


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
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", package_name.lower()).strip("-")
    return slug or "plugin"
