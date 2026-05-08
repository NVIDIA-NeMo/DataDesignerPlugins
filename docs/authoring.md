# Plugin Authoring

Create plugins through the repository tooling first. The scaffold is the source
of truth for package shape, entry point registration, tests, and ownership
files.

## Contributing to the NVIDIA tap

This repository is the NVIDIA-maintained curated first-party plugin tap for
Data Designer. Add a new plugin here when all of these are true:

- NVIDIA will maintain the plugin and its releases.
- The plugin is broadly useful to Data Designer users rather than specific to a
  private workflow, one team, or one downstream deployment.
- The plugin has an accountable CODEOWNER who can review changes, answer
  compatibility questions, and own release follow-through.

Plugins that are external, team-specific, experimental, or community-maintained
can still be useful without landing in DDPlugins. Publish those from an
external tap instead. External taps should expose a schema v2 catalog from an
unauthenticated raw JSON URL, or from a local catalog file path for authoring
and offline workflows. See [Plugin taps](taps.md) for the consumer and
maintainer workflow, and [Tap catalog schema v2](tap-catalog-schema-v2.md) for
the JSON contract and install source metadata.

Adding a tap is a trust decision, not only a discovery preference. A tap is a
pointer to Python packages. Installing from a tap runs package-manager
resolution and imports code after installation. Review the tap URL, package
name, version, source/ref, and install command before confirming installs from
non-default taps.

## Scaffold a plugin

From the repository root:

```bash
make sync
uv run ddp new my-plugin
```

This creates a column generator by default. To scaffold another plugin type,
pass one of the supported `--type` values:

```bash
uv run ddp new my-column-generator --type column-generator
uv run ddp new my-seed-reader --type seed-reader
uv run ddp new my-processor --type processor
```

The supported scaffold types are `column-generator`, `seed-reader`, and
`processor`. The default is `column-generator`, so `uv run ddp new my-plugin`
and `uv run ddp new my-plugin --type column-generator` create the same plugin
type.

The command creates a package named `data-designer-my-plugin`:

```text
plugins/data-designer-my-plugin/
|-- pyproject.toml
|-- README.md
|-- CODEOWNERS
|-- docs/
|   `-- index.md
|-- tests/
|   `-- test_plugin.py
`-- src/
    `-- data_designer_my_plugin/
        |-- __init__.py
        |-- config.py
        |-- impl.py
        `-- plugin.py
```

Use `plugins/data-designer-template/` as the reference implementation before
introducing a new structure.

## Naming and discovery

Plugin packages use the `data-designer-` prefix. The entry point key is the
slug that Data Designer discovers at runtime. For column generators, this is the
column type. For seed readers, use the same slug as the `seed_type`
discriminator. For processors, use the same slug as the `processor_type`
discriminator:

```toml
[project]
name = "data-designer-my-plugin"

[project.entry-points."data_designer.plugins"]
my-plugin = "data_designer_my_plugin.plugin:plugin"
```

The module path must use absolute imports:

```python
from data_designer_my_plugin.config import MyPluginColumnConfig
```

Do not use relative imports in this repository.

Tap discovery and runtime entry-point discovery are related but separate. The
tap catalog lets a consumer find package metadata and installation sources
before a package is installed. Runtime discovery happens later, after
installation, when Data Designer loads the `data_designer.plugins` entry-point
group from the active Python environment.

## How catalog entries are generated

`make catalog` builds `catalog/plugins.json` from installed local entry points,
package metadata, plugin docs metadata, compatibility dependencies, and the
repository-level `[tool.ddp.tap]` table.

| Input | Catalog fields |
| --- | --- |
| `[project].name` | Package `name`, PyPI install target, and generated docs path. |
| `[project].version` | Package `version`; consumers combine this with package `name` for exact PyPI installs. |
| `[project].description` | Package `description`. |
| `[project].requires-python` | `compatibility.python.specifier`. |
| Direct `data-designer` dependency in `[project].dependencies` | `compatibility.data_designer.requirement`, `.specifier`, and `.marker`. |
| `[project.entry-points."data_designer.plugins"]` | Runtime plugin `entry_point.group`, `entry_point.name`, and `entry_point.value`. |
| Loaded plugin object | Runtime plugin `name` and `plugin_type` under the package's `plugins` array. |
| Plugin package directory | Used as Git `source.subdirectory` or path `source.path` when the repo is configured to generate those source types. |
| Plugin docs under `plugins/<package>/docs/` plus `[tool.ddp.tap].docs-base-url` | `docs.url`. |
| `[tool.ddp.tap].default-source`, `repository-git-url`, and `release-ref-template` | The generated `source` object. |

Catalog entries describe installability, but they do not replace runtime entry
points. A plugin becomes available to Data Designer only after the package is
installed and its entry point can be discovered in the active environment.

## Implement the plugin

The scaffold separates each plugin type into three concerns:

| File | Responsibility |
| --- | --- |
| `config.py` | Plugin configuration, discriminator, parameters, dependencies, and metadata. |
| `impl.py` | Runtime column generation, seed reading, or processing logic. |
| `plugin.py` | Data Designer plugin object and entry point target. |

Keep functions and methods short enough to read in one pass. Prefer reusable
helpers over nested private closures. Dependencies should be declared by the
plugin package that needs them, not by another local plugin.

## Test public behavior

The scaffold includes a validation test:

```python
from data_designer.engine.testing.utils import assert_valid_plugin

from data_designer_my_plugin.plugin import plugin


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)
```

Add functional tests for the behavior in `impl.py` and for any meaningful
configuration validation. Tests should exercise public interfaces and expected
Data Designer behavior instead of private implementation details.

Run the isolated plugin test target while developing:

```bash
make test-plugin PLUGIN=data-designer-my-plugin
```

For a faster loop, run the package tests directly:

```bash
uv run pytest plugins/data-designer-my-plugin/tests/ -v
```

## Document the plugin

Each plugin owns its site documentation under its package directory:

```text
plugins/data-designer-my-plugin/docs/
|-- index.md
`-- usage.md
```

The top-level docs build copies this content into the generated
`docs/plugins/` tree and adds it to the Zensical navigation. Keep links and
assets relative to the plugin's `docs/` directory so they continue to work after
generation.

Every plugin gets a generated fallback page from package metadata when it does
not provide docs yet, but plugin-authored pages should be the source of truth
for usage, configuration, and examples.

## Regenerate metadata

When plugin docs, plugin metadata, compatibility dependencies, tap config, or
ownership changes, regenerate the derived files:

```bash
make sync
make plugin-docs
make catalog
make codeowners
```

Generated files and their inputs:

| Output | Regenerate with | Inputs |
| --- | --- | --- |
| `docs/plugins/` | `make plugin-docs` | Plugin package docs and package metadata. |
| Generated plugin nav block in `zensical.toml` | `make plugin-docs` | Generated plugin docs tree. |
| `catalog/plugins.json` | `make catalog` | Installed entry points, package metadata, compatibility dependencies, plugin docs URLs, and `[tool.ddp.tap]`. |
| `.github/CODEOWNERS` | `make codeowners` | Per-plugin `CODEOWNERS` files. |

CI verifies that generated plugin docs, `catalog/plugins.json`, and
`.github/CODEOWNERS` are current. The catalog's
`compatibility.data_designer.requirement` and
`compatibility.data_designer.specifier` fields come from each package's direct
versioned `data-designer` dependency in `[project].dependencies`. The catalog
also publishes the package's `requires-python` specifier and any
`data-designer` dependency environment marker.

## External tap authoring

To author an external tap, copy or fork this repository structure and configure
the root `[tool.ddp.tap]` table for your organization:

```toml
[tool.ddp.tap]
catalog-url = "https://raw.githubusercontent.com/acme/DataDesignerPlugins/main/catalog/plugins.json"
repository-url = "https://github.com/acme/DataDesignerPlugins"
repository-git-url = "https://github.com/acme/DataDesignerPlugins.git"
docs-base-url = "https://acme.github.io/DataDesignerPlugins/"
package-prefix = "data-designer-acme-"
default-source = "pypi"
release-ref-template = "{package}/v{version}"
default-data-designer-requirement = "data-designer>=0.5.7"
author-name = "ACME"
```

Then scaffold, validate, and publish the raw JSON catalog:

```bash
make sync
uv run ddp new my-plugin --type column-generator
make all
```

Publish the generated `catalog/plugins.json` at the `catalog-url` as raw JSON.
Do not point users at an HTML repository page or documentation site as the
machine catalog. Once Data Designer supports tap configuration, tell users to
add that raw catalog URL through the `plugins taps add` flow before discovering
or installing plugins from the external tap.
