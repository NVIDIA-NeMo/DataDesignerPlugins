# Adding a Plugin

## Quick Start

The fastest way to create a plugin is with the scaffold tool:

```bash
uv run scaffold-plugin my-plugin
```

This generates all required files including `pyproject.toml`, the source package, a test stub, and a `CODEOWNERS` file (auto-populated from your git config). See the printed output for next steps.

Alternatively, you can copy the template manually:

1. Copy the template plugin:

```bash
cp -r plugins/data-designer-template plugins/data-designer-my-plugin
```

2. Rename the Python package directory:

```bash
mv plugins/data-designer-my-plugin/src/data_designer_template \
   plugins/data-designer-my-plugin/src/data_designer_my_plugin
```

3. Update `plugins/data-designer-my-plugin/pyproject.toml`:

```toml
[project]
name = "data-designer-my-plugin"
version = "0.1.0"
description = "My custom Data Designer plugin"
requires-python = ">=3.10"
dependencies = [
    "data-designer",
]

[project.entry-points."data_designer.plugins"]
my-column-type = "data_designer_my_plugin.plugin:plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/data_designer_my_plugin"]

[tool.ruff]
extend = "../../pyproject.toml"
```

4. Implement the three Python files:

- **`config.py`** — Subclass `SingleColumnConfig`. Set `column_type` as a `Literal` with your type slug. Define any config params, `required_columns`, and `side_effect_columns`.
- **`impl.py`** — Subclass `ColumnGeneratorFullColumn[YourConfig]` (or `ColumnGeneratorCellByCell[YourConfig]` for row-by-row processing). Implement `generate()`.
- **`plugin.py`** — Create a `Plugin` instance with fully-qualified names for your config and impl classes.

5. Test:

```bash
uv sync --all-packages
uv run pytest plugins/data-designer-my-plugin/tests/ -v
```

## Validation

Use `assert_valid_plugin` to verify your plugin is wired correctly:

```python
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer_my_plugin.plugin import plugin

assert_valid_plugin(plugin)
```

## Column Generator Types

- **`ColumnGeneratorFullColumn`** — receives and returns a full `pd.DataFrame`. Use when the transform operates on the whole column at once.
- **`ColumnGeneratorCellByCell`** — receives and returns a `dict` (single row). Use for row-level transforms, especially those involving model calls.

## Entry Points

Plugins register via `[project.entry-points."data_designer.plugins"]` in their `pyproject.toml`. The key is the plugin slug; the value points to the `Plugin` instance. Data Designer discovers plugins automatically at import time via this mechanism.

## Code Ownership

Each plugin has its own `CODEOWNERS` file (created automatically by `scaffold-plugin`). These per-plugin files are aggregated into the root `CODEOWNERS` by running:

```bash
python tools/aggregate_codeowners.py > CODEOWNERS
```

CI checks that the root file stays in sync. If you add or update a per-plugin `CODEOWNERS`, regenerate the root file before pushing.
