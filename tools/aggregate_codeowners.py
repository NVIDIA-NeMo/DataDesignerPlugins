# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-plugin CODEOWNERS files into a single root CODEOWNERS."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    plugins_dir = repo_root / "plugins"

    entries: list[tuple[str, str]] = []

    for codeowners_path in sorted(plugins_dir.glob("*/CODEOWNERS")):
        plugin_dir_name = codeowners_path.parent.name
        owners: list[str] = []
        for line in codeowners_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Lines like "* @user" or "* user@example.com" — take everything after the pattern
            parts = stripped.split()
            if len(parts) >= 2:
                owners.extend(parts[1:])
            elif len(parts) == 1 and not parts[0].startswith("*"):
                owners.append(parts[0])

        if owners:
            entries.append((plugin_dir_name, " ".join(owners)))

    lines = [
        "# Auto-generated from per-plugin CODEOWNERS files. Do not edit manually.",
        "# Run: python tools/aggregate_codeowners.py > CODEOWNERS",
        "",
        "# Infrastructure",
        "* @etramel",
        "/core/ @etramel",
        "/.gitlab-ci.yml @etramel",
        "/tools/ @etramel",
        "",
        "# Plugins",
    ]

    for plugin_dir_name, owners in entries:
        lines.append(f"/{plugins_dir.name}/{plugin_dir_name}/ {owners}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
