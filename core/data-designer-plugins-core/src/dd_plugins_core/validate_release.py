# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate plugin metadata before a PyPI release."""

from __future__ import annotations

import sys

from dd_plugins_core._repo import find_repo_root, load_toml

REQUIRED_FIELDS = ("description", "license", "readme", "authors")


def main() -> int:
    """Validate that a plugin's pyproject.toml is release-ready.

    Expects two positional CLI arguments: ``<plugin-name>`` and
    ``<tag-version>``.

    Returns:
        Exit code (0 for success, 1 for validation failure, 2 for usage error).
    """
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <plugin-name> <tag-version>", file=sys.stderr)
        return 2

    plugin_name = sys.argv[1]
    tag_version = sys.argv[2]

    repo_root = find_repo_root()
    toml_path = repo_root / "plugins" / plugin_name / "pyproject.toml"

    # 1. Check pyproject.toml exists
    if not toml_path.exists():
        print(f"Error: {toml_path} not found", file=sys.stderr)
        return 1

    data = load_toml(toml_path)

    project = data.get("project", {})
    errors: list[str] = []

    # 2. Version match
    file_version = project.get("version")
    if file_version is None:
        errors.append("project.version is missing")
    elif file_version != tag_version:
        errors.append(f"Version mismatch: pyproject.toml has '{file_version}', expected '{tag_version}'")

    # 3. Required PyPI metadata
    for field in REQUIRED_FIELDS:
        if field not in project:
            errors.append(f"Required field project.{field} is missing")

    if errors:
        print(f"Validation failed for {plugin_name}:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Validation passed for {plugin_name} v{tag_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
