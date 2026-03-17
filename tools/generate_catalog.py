# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a markdown plugin catalog from plugin metadata."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        print(
            "Error: Python 3.11+ (tomllib) or the 'tomli' package is required.\nInstall with: pip install tomli",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    plugins_dir = repo_root / "plugins"

    rows: list[tuple[str, str, str, str]] = []

    for toml_path in sorted(plugins_dir.glob("*/pyproject.toml")):
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})
        name = project.get("name", toml_path.parent.name)
        version = project.get("version", "unknown")
        description = project.get("description", "")

        entry_points = project.get("entry-points", {}).get("data_designer.plugins", {})

        if entry_points:
            for ep_key in sorted(entry_points):
                rows.append((name, version, ep_key, description))
        else:
            rows.append((name, version, "", description))

    rows.sort(key=lambda r: (r[0], r[2]))

    lines = [
        "# Plugin Catalog",
        "",
        "Auto-generated from plugin metadata. Do not edit manually.",
        "",
        "| Plugin | Version | Column Type | Description |",
        "|--------|---------|-------------|-------------|",
    ]
    for name, version, column_type, description in rows:
        ct = f"`{column_type}`" if column_type else ""
        lines.append(f"| {name} | {version} | {ct} | {description} |")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
