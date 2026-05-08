# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tool to scaffold a new Data Designer plugin."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ddp._repo import SPDX_HEADER
from ddp.catalog_config import CatalogConfig, CatalogConfigError, load_catalog_config

PluginScaffoldType = Literal["column-generator", "seed-reader", "processor"]
DEFAULT_PLUGIN_SCAFFOLD_TYPE: PluginScaffoldType = "column-generator"
PLUGIN_SCAFFOLD_TYPES: tuple[PluginScaffoldType, ...] = (
    "column-generator",
    "seed-reader",
    "processor",
)


@dataclass(frozen=True)
class ScaffoldSpec:
    """Type-specific scaffold metadata."""

    scaffold_type: PluginScaffoldType
    display_name: str
    config_class_suffix: str
    impl_class_suffix: str
    plugin_type_member: str
    config_next_step: str
    impl_next_step: str

    def config_class_name(self, class_prefix: str) -> str:
        """Return the generated config class name."""
        return f"{class_prefix}{self.config_class_suffix}"

    def impl_class_name(self, class_prefix: str) -> str:
        """Return the generated implementation class name."""
        return f"{class_prefix}{self.impl_class_suffix}"


SCAFFOLD_SPECS: dict[str, ScaffoldSpec] = {
    "column-generator": ScaffoldSpec(
        scaffold_type="column-generator",
        display_name="column generator",
        config_class_suffix="ColumnConfig",
        impl_class_suffix="ColumnGenerator",
        plugin_type_member="COLUMN_GENERATOR",
        config_next_step="Edit src/{import_name}/config.py to define your column config",
        impl_next_step="Edit src/{import_name}/impl.py to implement generation logic",
    ),
    "seed-reader": ScaffoldSpec(
        scaffold_type="seed-reader",
        display_name="seed reader",
        config_class_suffix="SeedSource",
        impl_class_suffix="SeedReader",
        plugin_type_member="SEED_READER",
        config_next_step="Edit src/{import_name}/config.py to define your seed source config",
        impl_next_step="Edit src/{import_name}/impl.py to implement manifest and hydration behavior",
    ),
    "processor": ScaffoldSpec(
        scaffold_type="processor",
        display_name="processor",
        config_class_suffix="ProcessorConfig",
        impl_class_suffix="Processor",
        plugin_type_member="PROCESSOR",
        config_next_step="Edit src/{import_name}/config.py to define your processor config",
        impl_next_step="Edit src/{import_name}/impl.py to implement the processor hook you need",
    ),
}


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


def scaffold_spec_for_type(plugin_type: str) -> ScaffoldSpec:
    """Return scaffold metadata for a validated plugin type."""
    return SCAFFOLD_SPECS[plugin_type]


def generate_pyproject(
    package_name: str,
    slug: str,
    import_name: str,
    catalog_config: CatalogConfig,
    spec: ScaffoldSpec,
) -> str:
    return f"""{SPDX_HEADER}

[project]
name = {toml_string(package_name)}
version = "0.1.0"
description = "Data Designer {slug} {spec.display_name} plugin"
requires-python = ">=3.10"
dependencies = [
    {toml_string(catalog_config.default_data_designer_requirement)},
]
license = "Apache-2.0"
readme = "README.md"
authors = [
    {{name = {toml_string(catalog_config.author_name)}}},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
]

[project.entry-points."data_designer.plugins"]
{slug} = "{import_name}.plugin:plugin"

[project.urls]
Repository = {toml_string(catalog_config.repository_url)}

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{import_name}"]

[tool.ruff]
extend = "../../pyproject.toml"
"""


def generate_readme_usage(slug: str, spec: ScaffoldSpec) -> str:
    """Return type-specific README usage text."""
    if spec.scaffold_type == "column-generator":
        return f"""This package provides a Data Designer column generator plugin.
Once installed, the `{slug}` column type is automatically discovered by
Data Designer."""
    if spec.scaffold_type == "seed-reader":
        return f"""This package provides a Data Designer seed reader plugin.
Once installed, the `{slug}` seed-reader entry point is automatically discovered
by Data Designer. Configure it with `seed_type="{slug}"`.

The generated implementation is a placeholder: fill in `build_manifest` with a
DuckDB-friendly manifest and override `hydrate_row` if file contents need
expensive parsing."""
    return f"""This package provides a Data Designer processor plugin.
Once installed, the `{slug}` processor entry point is automatically discovered
by Data Designer. Configure it with `processor_type="{slug}"`.

The generated implementation is a placeholder: `process_after_batch` currently
returns data unchanged until processor behavior is implemented."""


def generate_docs_usage(slug: str, spec: ScaffoldSpec) -> str:
    """Return type-specific docs index usage text."""
    if spec.scaffold_type == "column-generator":
        return f"""## Column type

Use the `{slug}` column type when a dataset needs this generated output column.

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Output column name. |

## Implementation notes

The generated implementation writes `None` to the configured output column until
generation logic is added."""
    if spec.scaffold_type == "seed-reader":
        return f"""## Plugin type

This scaffold registers a `seed-reader` plugin.

## Entry point

Configure this seed source with `seed_type="{slug}"`.

## Implementation notes

The generated implementation imports and validates, but it intentionally returns
no manifest rows until `build_manifest` is implemented. Build a DuckDB-friendly manifest with scalar values, and override `hydrate_row` when expensive file I/O
or parsing should happen after manifest sampling."""
    return f"""## Plugin type

This scaffold registers a `processor` plugin.

## Entry point

Configure this processor with `processor_type="{slug}"`.

## Implementation notes

The generated `process_after_batch` hook returns data unchanged until processor
behavior is implemented.

Processor validation note: Data Designer 0.5.7 `assert_valid_plugin` validates processor config wiring but does not fully check processor implementation classes,
so the generated test also imports and checks the implementation class."""


def generate_readme(package_name: str, slug: str, catalog_config: CatalogConfig, spec: ScaffoldSpec) -> str:
    return f"""# {package_name}

Data Designer {slug} {spec.display_name} plugin.

## Installation

```bash
uv add data-designer {package_name}
```

## Usage

{generate_readme_usage(slug, spec)}

For the full plugin authoring guide, see the
[main repository docs]({catalog_config.docs_url("authoring/")}).

Plugin documentation for the repository site lives in this package's `docs/`
directory.
"""


def generate_docs_index(package_name: str, slug: str, catalog_config: CatalogConfig, spec: ScaffoldSpec) -> str:
    return f"""# {package_name}

Data Designer {slug} {spec.display_name} plugin.

## Installation

```bash
uv add data-designer {package_name}
```

{generate_docs_usage(slug, spec)}

For the full plugin authoring guide, see the
[main repository docs]({catalog_config.docs_url("authoring/")}).
"""


def generate_init() -> str:
    return f"{SPDX_HEADER}\n"


def generate_column_config(slug: str, class_prefix: str) -> str:
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


def generate_seed_reader_config(slug: str, class_prefix: str) -> str:
    return f'''{SPDX_HEADER}

from typing import Literal

from data_designer.config.base import ConfigBase
from data_designer.config.seed_source import FileSystemSeedSource


class {class_prefix}SeedSource(FileSystemSeedSource, ConfigBase):
    """Configuration for the {slug} filesystem seed reader."""

    seed_type: Literal["{slug}"] = "{slug}"
'''


def generate_processor_config(slug: str, class_prefix: str) -> str:
    return f'''{SPDX_HEADER}

from typing import Literal

from data_designer.config.base import ProcessorConfig


class {class_prefix}ProcessorConfig(ProcessorConfig):
    """Configuration for the {slug} processor."""

    processor_type: Literal["{slug}"] = "{slug}"
'''


def generate_config(slug: str, class_prefix: str, spec: ScaffoldSpec) -> str:
    """Return config.py content for the requested plugin type."""
    if spec.scaffold_type == "column-generator":
        return generate_column_config(slug, class_prefix)
    if spec.scaffold_type == "seed-reader":
        return generate_seed_reader_config(slug, class_prefix)
    return generate_processor_config(slug, class_prefix)


def generate_column_impl(slug: str, import_name: str, class_prefix: str) -> str:
    return f"""{SPDX_HEADER}

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from {import_name}.config import {class_prefix}ColumnConfig

if TYPE_CHECKING:
    import pandas as pd


class {class_prefix}ColumnGenerator(ColumnGeneratorFullColumn[{class_prefix}ColumnConfig]):
    \"\"\"Placeholder column generator for the {slug} column type.\"\"\"

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        # TODO: Implement your column generation logic here
        data[self.config.name] = None
        return data
"""


def generate_seed_reader_impl(slug: str, import_name: str, class_prefix: str) -> str:
    return f"""{SPDX_HEADER}

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from data_designer.engine.resources.seed_reader import FileSystemSeedReader, SeedReaderFileSystemContext

from {import_name}.config import {class_prefix}SeedSource

if TYPE_CHECKING:
    import pandas as pd


class {class_prefix}SeedReader(FileSystemSeedReader[{class_prefix}SeedSource]):
    \"\"\"Placeholder filesystem seed reader for the {slug} entry point.\"\"\"

    def build_manifest(
        self,
        *,
        context: SeedReaderFileSystemContext,
    ) -> pd.DataFrame | list[dict[str, Any]]:
        \"\"\"Return manifest rows for files under the configured seed source path.\"\"\"
        # TODO: Inspect context.root_path and return one DuckDB-friendly dict per logical seed row.
        # Override hydrate_row when file I/O or expensive parsing should run after manifest sampling.
        return []
"""


def generate_processor_impl(slug: str, import_name: str, class_prefix: str) -> str:
    return f"""{SPDX_HEADER}

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor

from {import_name}.config import {class_prefix}ProcessorConfig

if TYPE_CHECKING:
    import pandas as pd


class {class_prefix}Processor(Processor[{class_prefix}ProcessorConfig]):
    \"\"\"Placeholder processor for the {slug} entry point.\"\"\"

    def process_after_batch(
        self,
        data: pd.DataFrame,
        *,
        current_batch_number: int | None,
    ) -> pd.DataFrame:
        \"\"\"Return each generated batch unchanged until processor logic is added.\"\"\"
        # TODO: Implement processor behavior for each generated batch.
        return data
"""


def generate_impl(slug: str, import_name: str, class_prefix: str, spec: ScaffoldSpec) -> str:
    """Return impl.py content for the requested plugin type."""
    if spec.scaffold_type == "column-generator":
        return generate_column_impl(slug, import_name, class_prefix)
    if spec.scaffold_type == "seed-reader":
        return generate_seed_reader_impl(slug, import_name, class_prefix)
    return generate_processor_impl(slug, import_name, class_prefix)


def generate_plugin(import_name: str, class_prefix: str, spec: ScaffoldSpec) -> str:
    config_class_name = spec.config_class_name(class_prefix)
    impl_class_name = spec.impl_class_name(class_prefix)
    return f"""{SPDX_HEADER}

from data_designer.plugins.plugin import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="{import_name}.config.{config_class_name}",
    impl_qualified_name="{import_name}.impl.{impl_class_name}",
    plugin_type=PluginType.{spec.plugin_type_member},
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


def generate_test(import_name: str, class_prefix: str, spec: ScaffoldSpec) -> str:
    if spec.scaffold_type == "processor":
        return f"""{SPDX_HEADER}

from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.plugins.plugin import PluginType

from {import_name}.impl import {class_prefix}Processor
from {import_name}.plugin import plugin


def test_valid_plugin() -> None:
    # Data Designer 0.5.7 assert_valid_plugin validates processor config wiring,
    # but it does not fully check processor implementation classes yet.
    assert_valid_plugin(plugin)
    assert plugin.plugin_type == PluginType.PROCESSOR
    assert plugin.impl_cls is {class_prefix}Processor
    assert issubclass(plugin.impl_cls, Processor)
"""

    return f"""{SPDX_HEADER}

from data_designer.engine.testing.utils import assert_valid_plugin

from {import_name}.plugin import plugin


def test_valid_plugin() -> None:
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
    parser.add_argument(
        "--type",
        choices=PLUGIN_SCAFFOLD_TYPES,
        default=DEFAULT_PLUGIN_SCAFFOLD_TYPE,
        dest="plugin_type",
        help="Plugin type to scaffold (default: column-generator).",
    )
    args = parser.parse_args(args)
    slug: str = args.name
    spec = scaffold_spec_for_type(args.plugin_type)

    error = validate_slug(slug)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    plugins_dir = find_plugins_dir()
    try:
        catalog_config = load_catalog_config(plugins_dir.parent)
    except CatalogConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    package_name = catalog_config.package_name_for_slug(slug)
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
        plugin_dir / "pyproject.toml": generate_pyproject(package_name, slug, import_name, catalog_config, spec),
        plugin_dir / "README.md": generate_readme(package_name, slug, catalog_config, spec),
        plugin_dir / "CODEOWNERS": generate_codeowners(owner),
        docs_dir / "index.md": generate_docs_index(package_name, slug, catalog_config, spec),
        src_dir / "__init__.py": generate_init(),
        src_dir / "config.py": generate_config(slug, class_prefix, spec),
        src_dir / "impl.py": generate_impl(slug, import_name, class_prefix, spec),
        src_dir / "plugin.py": generate_plugin(import_name, class_prefix, spec),
        test_dir / "test_plugin.py": generate_test(import_name, class_prefix, spec),
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
    print(f"  2. {spec.config_next_step.format(import_name=import_name)}")
    print(f"  3. {spec.impl_next_step.format(import_name=import_name)}")
    print("  4. Edit docs/index.md to document your plugin")
    print("  5. uv sync --all-packages && uv run pytest tests/")
    print(f"  6. make release PLUGIN={package_name}")


if __name__ == "__main__":
    main()
