# Plugin Authoring

Create plugins through the repository tooling first. The scaffold is the source
of truth for package shape, entry point registration, tests, and ownership
files.

## Scaffold a plugin

From the repository root:

```bash
make sync
uv run ddp new my-plugin
```

This creates a package named `data-designer-my-plugin`:

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
column type slug that Data Designer discovers at runtime:

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

## Implement the plugin

The scaffold separates the plugin into three concerns:

| File | Responsibility |
| --- | --- |
| `config.py` | Column configuration, parameters, dependencies, and metadata. |
| `impl.py` | Runtime generation logic. |
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

When plugin docs, plugin metadata, or ownership changes, regenerate the derived
files:

```bash
make sync
make plugin-docs
make catalog
make codeowners
```

CI verifies that generated plugin docs, `catalog/plugins.json`, and
`.github/CODEOWNERS` are current. The catalog's
`compatibility.data_designer.requirement` and
`compatibility.data_designer.specifier` fields come from each package's direct
versioned `data-designer` dependency in `[project].dependencies`. The catalog
also publishes the package's `requires-python` specifier and any
`data-designer` dependency environment marker.
