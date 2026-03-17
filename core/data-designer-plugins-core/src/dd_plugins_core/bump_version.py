# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bump a plugin's semantic version in its pyproject.toml."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dd_plugins_core._repo import find_repo_root, load_toml

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
VERSION_LINE_RE = re.compile(r'(version\s*=\s*")([^"]+)(")')


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a strict ``X.Y.Z`` version string into its components.

    Args:
        version: A semantic version string such as ``"1.2.3"``.

    Returns:
        A tuple of (major, minor, patch) integers.

    Raises:
        SystemExit: If the version contains a pre-release suffix or is
            otherwise malformed.
    """
    if re.match(r"^\d+\.\d+\.\d+[-+.]", version):
        print(
            f"Error: Pre-release version '{version}' is not supported by bump-version.\n"
            "Edit pyproject.toml manually for pre-release versions.",
            file=sys.stderr,
        )
        sys.exit(1)

    match = SEMVER_RE.match(version)
    if not match:
        print(f"Error: '{version}' is not a valid X.Y.Z version.", file=sys.stderr)
        sys.exit(1)

    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump_semver(major: int, minor: int, patch: int, part: str) -> tuple[int, int, int]:
    """Apply a semantic version bump.

    Args:
        major: Current major version.
        minor: Current minor version.
        patch: Current patch version.
        part: Which component to bump (``"major"``, ``"minor"``, or
            ``"patch"``).

    Returns:
        A tuple of (major, minor, patch) after the bump.
    """
    if part == "major":
        return (major + 1, 0, 0)
    if part == "minor":
        return (major, minor + 1, 0)
    return (major, minor, patch + 1)


def read_version(toml_path: Path) -> str:
    """Read ``project.version`` from a pyproject.toml file.

    Args:
        toml_path: Path to the pyproject.toml file.

    Returns:
        The version string.

    Raises:
        SystemExit: If the file does not exist or ``project.version`` is
            missing.
    """
    if not toml_path.exists():
        print(f"Error: {toml_path} not found", file=sys.stderr)
        sys.exit(1)

    data = load_toml(toml_path)
    version = data.get("project", {}).get("version")
    if version is None:
        print(f"Error: project.version not found in {toml_path}", file=sys.stderr)
        sys.exit(1)

    return version


def replace_version_in_file(toml_path: Path, old: str, new: str) -> None:
    """Replace the version string in a pyproject.toml file using raw text.

    Performs a targeted regex substitution of ``version = "old"`` to
    ``version = "new"`` on the raw file text, preserving all other
    formatting.

    Args:
        toml_path: Path to the pyproject.toml file.
        old: The expected current version string.
        new: The replacement version string.

    Raises:
        SystemExit: If the expected version line is not found in the file
            (guards against TOCTOU races or unexpected formatting).
    """
    text = toml_path.read_text()
    pattern = re.compile(r'(version\s*=\s*")' + re.escape(old) + r'(")')
    if not pattern.search(text):
        print(
            f"Error: Could not find 'version = \"{old}\"' in {toml_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    updated = pattern.sub(rf"\g<1>{new}\2", text, count=1)
    toml_path.write_text(updated)


def main() -> int:
    """Bump a plugin's semantic version in its pyproject.toml.

    Parses CLI arguments for the plugin name and version part to bump,
    then rewrites the version in-place and prints next-step instructions.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="bump-version",
        description="Bump the semantic version of a data-designer plugin.",
    )
    parser.add_argument("plugin", help="Plugin name (e.g. data-designer-my-plugin)")
    parser.add_argument(
        "part",
        choices=("major", "minor", "patch"),
        help="Version component to bump",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    toml_path = repo_root / "plugins" / args.plugin / "pyproject.toml"

    old_version = read_version(toml_path)
    major, minor, patch = parse_semver(old_version)
    new_major, new_minor, new_patch = bump_semver(major, minor, patch, args.part)
    new_version = f"{new_major}.{new_minor}.{new_patch}"

    replace_version_in_file(toml_path, old_version, new_version)

    print(f"Bumped {args.plugin}: {old_version} → {new_version}")
    print()
    print("Next steps:")
    print(f"  git add plugins/{args.plugin}/pyproject.toml")
    print(f'  git commit -m "chore({args.plugin}): bump version to {new_version}"')
    print(f"  make release PLUGIN={args.plugin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
