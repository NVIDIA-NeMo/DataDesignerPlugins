# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import codecs
from pathlib import Path
from typing import ClassVar, Literal

from data_designer.config.base import ConfigBase
from data_designer.config.seed_source import FileSystemSeedSource
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

DEFAULT_CODE_EXTENSIONS = [
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".kts",
    ".lua",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
    ".zsh",
]

DEFAULT_CODE_FILENAMES = [
    "Dockerfile",
    "Makefile",
]

DEFAULT_EXCLUDE_PATTERNS = [
    ".git/*",
    ".git/**",
    ".mypy_cache/*",
    ".pytest_cache/*",
    ".ruff_cache/*",
    ".tox/*",
    ".venv/*",
    "__pycache__/*",
    "build/*",
    "dist/*",
    "node_modules/*",
    "venv/*",
    "**/.git/*",
    "**/.git/**",
    "**/.mypy_cache/*",
    "**/.pytest_cache/*",
    "**/.ruff_cache/*",
    "**/.tox/*",
    "**/.venv/*",
    "**/__pycache__/*",
    "**/build/*",
    "**/dist/*",
    "**/node_modules/*",
    "**/venv/*",
]


class GitHubSeedSource(FileSystemSeedSource, ConfigBase):
    """Seed source for reading code files from GitHub and local git repositories."""

    seed_type: Literal["github"] = "github"

    path: str | None = Field(
        None,
        description=(
            "Optional local git repository path, or a directory whose immediate children are git repositories. "
            "Relative paths are resolved from the current working directory when the config is loaded."
        ),
    )
    repositories: list[str] = Field(
        default_factory=list,
        description=(
            "GitHub repositories to clone before reading. Each entry may be 'owner/name', "
            "'https://github.com/owner/name', or 'https://github.com/owner/name.git'."
        ),
    )
    repository_paths: list[str] = Field(
        default_factory=list,
        description="Additional local git repository paths to read.",
    )
    ref: str | None = Field(
        None,
        description="Optional branch, tag, or commit to check out after cloning GitHub repositories.",
    )
    clone_depth: int | None = Field(
        1,
        ge=1,
        description="Depth for GitHub clones. Set to null for a full clone.",
    )
    clone_timeout_seconds: int = Field(
        300,
        ge=1,
        description="Timeout for each git clone or checkout operation.",
    )
    include_extensions: list[str] | None = Field(
        default_factory=lambda: list(DEFAULT_CODE_EXTENSIONS),
        description=(
            "Lowercase file extensions to include. Values may include or omit the leading dot. "
            "Set to null to include every extension."
        ),
    )
    include_file_names: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CODE_FILENAMES),
        description="Extensionless file names to include, such as Dockerfile or Makefile.",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS),
        description="Relative path glob patterns to exclude from repository scans.",
    )
    max_file_size_bytes: int = Field(
        1_000_000,
        ge=1,
        description="Maximum file size to hydrate into the content column.",
    )
    encoding: str = Field(
        "utf-8",
        description="Text encoding used when hydrating repository file contents.",
    )

    _source_fields: ClassVar[tuple[str, ...]] = ("path", "repositories", "repository_paths")

    @model_validator(mode="after")
    def validate_has_repository_source(self) -> Self:
        """Ensure the seed source has at least one repository source."""
        if self.path is None and not self.repositories and not self.repository_paths:
            fields = ", ".join(self._source_fields)
            raise ValueError(f"At least one of {fields} must be provided.")
        return self

    @field_validator("encoding", mode="after")
    @classmethod
    def validate_encoding(cls, value: str) -> str:
        """Validate that the configured text encoding exists."""
        try:
            codecs.lookup(value)
        except LookupError as error:
            raise ValueError(f"Unknown encoding: {value!r}. Use a valid Python codec name.") from error
        return value

    @field_validator("include_extensions", mode="after")
    @classmethod
    def normalize_include_extensions(cls, value: list[str] | None) -> list[str] | None:
        """Normalize configured extensions to lowercase dotted values."""
        if value is None:
            return None

        normalized: list[str] = []
        for extension in value:
            stripped = extension.strip().lower()
            if not stripped:
                raise ValueError("include_extensions cannot contain empty values.")
            normalized.append(stripped if stripped.startswith(".") else f".{stripped}")
        return sorted(set(normalized))

    @field_validator("include_file_names", "exclude_patterns", mode="after")
    @classmethod
    def validate_non_empty_strings(cls, value: list[str]) -> list[str]:
        """Validate string list fields do not contain blank entries."""
        for item in value:
            if not item.strip():
                raise ValueError("String lists cannot contain empty values.")
        return value

    @field_validator("repositories", mode="after")
    @classmethod
    def validate_repositories(cls, value: list[str]) -> list[str]:
        """Validate repository specs do not contain blank entries."""
        for repository in value:
            if not repository.strip():
                raise ValueError("repositories cannot contain empty values.")
        return value

    @field_validator("repository_paths", mode="after")
    @classmethod
    def validate_repository_paths(cls, value: list[str]) -> list[str]:
        """Validate explicit local repository paths exist."""
        for repository_path in value:
            path = Path(repository_path).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f"Repository path {path} is not a directory.")
        return value

    @property
    def runtime_path(self) -> str:
        """Return the resolved local scan root after a reader has prepared it."""
        if self._runtime_path is not None:
            return self._runtime_path
        if self.path is None:
            raise ValueError("GitHubSeedSource.runtime_path is available after the seed reader is attached.")
        self._runtime_path = str(Path(self.path).expanduser().resolve())
        return self._runtime_path
