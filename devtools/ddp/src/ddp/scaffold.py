# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tool to scaffold a new Data Designer plugin."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from ddp._repo import SPDX_HEADER
from ddp.tap_config import TapConfig, TapConfigError, load_tap_config


def to_underscored(slug: str) -> str:
    return slug.replace("-", "_")


def to_pascal(slug: str) -> str:
    return "".join(word.capitalize() for word in slug.split("-"))


def validate_slug(slug: str) -> str | None:
    """Validate kebab-case slug. Returns error message or None."""
    if not re.fullmatch(r"[a-z][a-z0-9]*(-[a-z0-9]+)*", slug):
        return (
            f"Invalid plugin name '{slug}'. "
            "Must be kebab-case: lowercase letters, digits, and hyphens only, "
            "no leading/trailing hyphens, no consecutive hyphens."
        )
    return None


def toml_string(value: str) -> str:
    """Format a string for use in generated TOML.

    Args:
        value: String value.

    Returns:
        Double-quoted TOML-compatible string.
    """
    return json.dumps(value)


def generate_pyproject(package_name: str, slug: str, import_name: str, tap_config: TapConfig) -> str:
    return f"""{SPDX_HEADER}

[project]
name = {toml_string(package_name)}
version = "0.1.0"
description = "Data Designer {slug} plugin"
requires-python = ">=3.10"
dependencies = [
    {toml_string(tap_config.default_data_designer_requirement)},
]
license = "Apache-2.0"
readme = "README.md"
authors = [
    {{name = {toml_string(tap_config.author_name)}}},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
]

[project.entry-points."data_designer.plugins"]
{slug} = "{import_name}.plugin:plugin"

[project.urls]
Repository = {toml_string(tap_config.repository_url)}

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{import_name}"]

[tool.ruff]
extend = "../../pyproject.toml"
"""


def generate_readme(package_name: str, slug: str, tap_config: TapConfig) -> str:
    return f"""# {package_name}

Data Designer {slug} plugin.

## Installation

```bash
uv add data-designer {package_name}
```

## Usage

Once installed, the `{slug}` column type is automatically discovered by
Data Designer.

For the full plugin authoring guide, see the
[main repository docs]({tap_config.docs_url("authoring/")}).

Plugin documentation for the repository site lives in this package's `docs/`
directory.
"""


def generate_docs_index(package_name: str, slug: str, tap_config: TapConfig) -> str:
    return f"""# {package_name}

Data Designer {slug} plugin.

## Installation

```bash
uv add data-designer {package_name}
```

## Usage

Once installed, the `{slug}` column type is automatically discovered by
Data Designer.

For the full plugin authoring guide, see the
[main repository docs]({tap_config.docs_url("authoring/")}).
"""


def generate_init() -> str:
    return f"{SPDX_HEADER}\n"


def generate_config(slug: str, import_name: str, class_prefix: str) -> str:
    return f'''{SPDX_HEADER}

from typing import Literal

from data_designer.config.base import SingleColumnConfig


class {class_prefix}ColumnConfig(SingleColumnConfig):
    """Configuration for the {slug} column generator."""

    column_type: Literal["{slug}"] = "{slug}"

    @staticmethod
    def get_column_emoji() -> str:
        return "\U0001f50c"

    @property
    def required_columns(self) -> list[str]:
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        return []
'''


def generate_impl(slug: str, import_name: str, class_prefix: str) -> str:
    return f"""{SPDX_HEADER}

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from {import_name}.config import {class_prefix}ColumnConfig

if TYPE_CHECKING:
    import pandas as pd


class {class_prefix}ColumnGenerator(ColumnGeneratorFullColumn[{class_prefix}ColumnConfig]):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        # TODO: Implement your column generation logic here
        data[self.config.name] = None
        return data
"""


def generate_plugin(import_name: str, class_prefix: str) -> str:
    return f"""{SPDX_HEADER}

from data_designer.plugins.plugin import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="{import_name}.config.{class_prefix}ColumnConfig",
    impl_qualified_name="{import_name}.impl.{class_prefix}ColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
"""


def _discover_owner() -> str:
    """Best-effort owner discovery from git config."""
    for key in ("user.email", "user.name"):
        try:
            result = subprocess.run(
                ["git", "config", key],
                capture_output=True,
                text=True,
                timeout=5,
            )
            value = result.stdout.strip()
            if value:
                return value
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "# TODO: set owner (GitHub @username, @org/team, or email)"


def generate_codeowners(owner: str) -> str:
    return f"""\
# Owner(s) of this plugin — used to generate the root CODEOWNERS file.
# GitHub accepts @username, @org/team, or email format.
* {owner}
"""


def generate_test(import_name: str) -> str:
    return f"""{SPDX_HEADER}

from data_designer.engine.testing.utils import assert_valid_plugin

from {import_name}.plugin import plugin


def test_valid_plugin():
    assert_valid_plugin(plugin)
"""


def find_plugins_dir() -> Path:
    """Walk up from CWD to find the repo root containing a plugins/ dir."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "plugins"
        if candidate.is_dir():
            return candidate
    # Fallback: assume CWD is the repo root
    return cwd / "plugins"


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new Data Designer plugin")
    parser.add_argument(
        "name",
        help="Plugin name in kebab-case (e.g., my-cool-thing)",
    )
    args = parser.parse_args(args)
    slug: str = args.name

    error = validate_slug(slug)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    plugins_dir = find_plugins_dir()
    try:
        tap_config = load_tap_config(plugins_dir.parent)
    except TapConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    package_name = tap_config.package_name_for_slug(slug)
    import_name = to_underscored(package_name)
    class_prefix = to_pascal(slug)

    plugin_dir = plugins_dir / package_name

    if plugin_dir.exists():
        print(
            f"Error: Plugin directory already exists: {plugin_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    src_dir = plugin_dir / "src" / import_name
    test_dir = plugin_dir / "tests"
    docs_dir = plugin_dir / "docs"

    # Create directories
    src_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)

    # Write files
    owner = _discover_owner()
    files = {
        plugin_dir / "pyproject.toml": generate_pyproject(package_name, slug, import_name, tap_config),
        plugin_dir / "README.md": generate_readme(package_name, slug, tap_config),
        plugin_dir / "CODEOWNERS": generate_codeowners(owner),
        docs_dir / "index.md": generate_docs_index(package_name, slug, tap_config),
        src_dir / "__init__.py": generate_init(),
        src_dir / "config.py": generate_config(slug, import_name, class_prefix),
        src_dir / "impl.py": generate_impl(slug, import_name, class_prefix),
        src_dir / "plugin.py": generate_plugin(import_name, class_prefix),
        test_dir / "test_plugin.py": generate_test(import_name),
    }

    for path, content in files.items():
        path.write_text(content)

    print(f"Created plugin '{package_name}' at {plugin_dir}/")
    print()
    print("Generated files:")
    for path in sorted(files):
        print(f"  {path.relative_to(plugin_dir)}")
    print()
    print("Next steps:")
    print(f"  1. cd {plugin_dir}")
    print(f"  2. Edit src/{import_name}/config.py to define your column config")
    print(f"  3. Edit src/{import_name}/impl.py to implement generation logic")
    print("  4. Edit docs/index.md to document your plugin")
    print("  5. uv sync --all-packages && uv run pytest tests/")
    print(f"  6. make release PLUGIN={package_name}")


if __name__ == "__main__":
    main()
