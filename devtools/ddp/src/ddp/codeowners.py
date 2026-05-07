# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate and parse per-plugin CODEOWNERS files."""

from __future__ import annotations

import re
from pathlib import Path

from ddp._repo import find_repo_root

GITHUB_USER_OWNER_PATTERN = re.compile(r"^@[A-Za-z0-9][A-Za-z0-9-]*$")
GITHUB_TEAM_OWNER_PATTERN = re.compile(r"^@[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9_-]*$")


def main() -> None:
    """Generate a unified CODEOWNERS file from per-plugin CODEOWNERS files."""
    repo_root = find_repo_root()
    plugins_dir = repo_root / "plugins"

    entries: list[tuple[str, str]] = []

    for codeowners_path in sorted(plugins_dir.glob("*/CODEOWNERS")):
        plugin_dir_name = codeowners_path.parent.name
        owners = owner_tokens_for_codeowners_path(codeowners_path)

        if owners:
            entries.append((plugin_dir_name, " ".join(owners)))

    lines = [
        "# Auto-generated from per-plugin CODEOWNERS files. Do not edit manually.",
        "# Run: uv run ddp codeowners > .github/CODEOWNERS",
        "",
        "# Infrastructure",
        "* @NVIDIA-NeMo/data_designer_reviewers",
        "/devtools/ @NVIDIA-NeMo/data_designer_reviewers",
        "/.github/ @NVIDIA-NeMo/data_designer_reviewers",
        "",
        "# Plugins",
    ]

    for plugin_dir_name, owners in entries:
        lines.append(f"/{plugins_dir.name}/{plugin_dir_name}/ {owners}")

    print("\n".join(lines))


def owner_tokens_for_codeowners_path(codeowners_path: Path) -> list[str]:
    """Return owner tokens from a per-plugin CODEOWNERS file.

    Args:
        codeowners_path: Path to a per-plugin CODEOWNERS file.

    Returns:
        Owner tokens in file order. Tokens may be GitHub users, GitHub teams,
        or email addresses.
    """
    return owner_tokens_from_codeowners_text(codeowners_path.read_text(encoding="utf-8"))


def owner_tokens_from_codeowners_text(codeowners_text: str) -> list[str]:
    """Return owner tokens from CODEOWNERS text.

    Args:
        codeowners_text: CODEOWNERS file content.

    Returns:
        Owner tokens in file order. Lines with a pattern return all tokens after
        the pattern; a single bare token is treated as an owner for the simple
        per-plugin format used by this repository.
    """
    owners: list[str] = []
    for line in codeowners_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            owners.extend(parts[1:])
        elif len(parts) == 1 and not parts[0].startswith("*"):
            owners.append(parts[0])
    return owners


def github_release_owners(owners: list[str]) -> list[str]:
    """Return GitHub user and team CODEOWNERS tokens eligible for releases.

    Args:
        owners: Owner tokens parsed from CODEOWNERS.

    Returns:
        Tokens that are GitHub users (``@user``) or teams (``@org/team``).
    """
    return [owner for owner in owners if is_github_release_owner(owner)]


def is_github_release_owner(owner: str) -> bool:
    """Return whether an owner token is a GitHub user or team.

    Args:
        owner: CODEOWNERS owner token.

    Returns:
        ``True`` when the token is a GitHub ``@user`` or ``@org/team`` owner.
    """
    return bool(GITHUB_USER_OWNER_PATTERN.fullmatch(owner) or GITHUB_TEAM_OWNER_PATTERN.fullmatch(owner))


if __name__ == "__main__":
    main()
