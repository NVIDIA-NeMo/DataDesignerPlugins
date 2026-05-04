# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a markdown plugin catalog from plugin metadata."""

from __future__ import annotations

from html import escape

from ddp._repo import find_repo_root, load_toml


def format_catalog_row(name: str, version: str, column_type: str, description: str) -> str:
    """Format a plugin metadata row for the catalog table.

    Args:
        name: Python package name for the plugin.
        version: Plugin package version.
        column_type: Data Designer column type entry point key.
        description: Plugin package description.

    Returns:
        An HTML table row for the plugin catalog.
    """
    column_type_cell = f"<code>{escape(column_type)}</code>" if column_type else ""
    return (
        "  <tr>\n"
        f'    <td class="plugin-catalog__plugin">{escape(name)}</td>\n'
        f'    <td class="plugin-catalog__version">{escape(version)}</td>\n'
        f'    <td class="plugin-catalog__column">{column_type_cell}</td>\n'
        f'    <td class="plugin-catalog__description">{escape(description)}</td>\n'
        "  </tr>"
    )


def format_catalog(rows: list[tuple[str, str, str, str]]) -> str:
    """Format plugin metadata as the Markdown catalog page.

    Args:
        rows: Plugin catalog rows as package, version, column type, and description.

    Returns:
        Markdown with an HTML table for the generated catalog page.
    """
    lines = [
        "# Plugin Catalog",
        "",
        '<div class="plugin-catalog-table">',
        '<table class="plugin-catalog">',
        "  <thead>",
        "  <tr>",
        "    <th>Plugin</th>",
        "    <th>Version</th>",
        "    <th>Column Type</th>",
        "    <th>Description</th>",
        "  </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    lines.extend(format_catalog_row(*row) for row in rows)
    lines.extend(
        [
            "  </tbody>",
            "</table>",
            "</div>",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Generate a markdown table of all plugins and print to stdout."""
    repo_root = find_repo_root()
    plugins_dir = repo_root / "plugins"

    rows: list[tuple[str, str, str, str]] = []

    for toml_path in sorted(plugins_dir.glob("*/pyproject.toml")):
        data = load_toml(toml_path)

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

    print(format_catalog(rows))


if __name__ == "__main__":
    main()
