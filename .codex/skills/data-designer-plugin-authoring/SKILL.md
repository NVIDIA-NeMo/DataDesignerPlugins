---
name: data-designer-plugin-authoring
description: Use when creating, updating, documenting, or preparing a pull request for a Data Designer plugin in the NVIDIA-NeMo/DataDesignerPlugins repository, including ddp scaffolding, plugin implementation, validation, and per-plugin Zensical docs.
metadata:
  short-description: Create Data Designer plugins
---

# Data Designer Plugin Authoring

Use this skill for plugin work in the `DataDesignerPlugins` repo. The repo is a `uv` workspace with shared tooling in `devtools/` and one independent package per plugin under `plugins/*`. The Python baseline is 3.10+.

## Context To Load

Before making plugin changes, read the local files that define the current contract:

- `AGENTS.md`
- `README.md`
- `docs/authoring.md`
- `docs/workflow.md`
- `Makefile`
- `zensical.toml`
- `devtools/ddp/src/ddp/scaffold.py`
- `devtools/ddp/src/ddp/plugin_docs.py`
- The reference plugin under `plugins/data-designer-template/`

After `make sync`, inspect Data Designer interfaces directly when signatures matter:

```bash
uv run python -c "import inspect; from data_designer.config.base import SingleColumnConfig; print(inspect.getsource(SingleColumnConfig))"
uv run python -c "import inspect; from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn; print(inspect.getsource(ColumnGeneratorFullColumn))"
```

## Scaffold With `ddp`

The initial plugin structure is owned by the repo's `ddp` CLI. Always invoke the scaffold command; do not create the package directory, `pyproject.toml`, source package, tests, docs, or ownership files by hand.

```bash
make sync
uv run ddp new <slug>
```

Use a kebab-case slug without the `data-designer-` prefix. If you need the exact scaffold behavior, read `devtools/ddp/src/ddp/scaffold.py` or inspect the generated files after running the command. Do not reproduce the scaffold algorithm in this skill; the software encodes that process deterministically.

If the command fails because the scaffold is wrong or incomplete, fix the `ddp` tooling or report the blocker. Do not bypass it by hand-assembling the initial plugin skeleton.

Read the generated files before editing them. If the generated class names stutter because the slug contains words such as `column`, rename the classes and update `plugin.py`.

## Implementation Rules

- Keep plugins self-contained. Do not add local dependencies on another plugin package.
- Use absolute imports such as `from data_designer_my_plugin.config import MyPluginColumnConfig`.
- Use Python 3.10+ annotations: `list[str]`, `A | B`, and `X | None`.
- Subclass `SingleColumnConfig` and define `column_type` as a `Literal["slug"]` with the same default string.
- Add Pydantic validators for structural constraints so bad config fails during config construction.
- Use the appropriate generator base, usually `ColumnGeneratorFullColumn[YourConfig]`.
- Keep logic in top-level reusable functions or modules. Do not use nested helper functions or private closures.
- Prefer vectorized pandas operations, named helpers, `functools.partial`, or dispatch tables over lambda-heavy `apply()` code.
- Use `from __future__ import annotations` and `TYPE_CHECKING` for pandas type-only imports.
- Add Google-style docstrings to public classes, functions, and methods.
- Add new import packages to the root Ruff isort `known-first-party` list.

The scaffold normally creates correct plugin wiring:

```python
from data_designer.plugins.plugin import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="data_designer_my_plugin.config.MyPluginColumnConfig",
    impl_qualified_name="data_designer_my_plugin.impl.MyPluginColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
```

## Tests

Write tests around public behavior:

- `assert_valid_plugin(plugin)` contract validation.
- Config defaults, dependency properties, and validation errors.
- Pure helper function behavior.
- Generator behavior on representative DataFrames.
- Data Designer preview integration when useful.
- Edge cases with `None`, `NaN`, empty strings, numeric values in text columns, and malformed config.

Use `pathlib.Path` for pytest `tmp_path` annotations.

Run plugin tests in isolation:

```bash
make test-plugin PLUGIN=data-designer-my-plugin
```

## Per-Plugin Zensical Docs

Each plugin owns source docs under `plugins/data-designer-<slug>/docs/`. `make plugin-docs` copies those files into the generated `docs/plugins/data-designer-<slug>/` tree and updates the generated nav block in `zensical.toml`.

Do not edit generated plugin docs or the generated nav block directly.

Recommended source docs:

```text
plugins/data-designer-<slug>/docs/
|-- index.md
`-- usage.md
```

Prepare `docs/index.md` as the overview:

- H1: `# data-designer-<slug>`.
- Short description of what the plugin adds.
- Installation command: `uv add data-designer data-designer-<slug>`.
- Column type section naming the entry point users configure.
- Configuration table with `Field`, `Required`, and `Description`.
- Realistic Python or YAML usage example.
- Important behavior notes, output columns, or limitations only when useful.

Prepare `docs/usage.md` when the plugin needs a fuller example. Its H1 becomes the Zensical nav label, so keep it concise. Include expected output shape, before/after behavior, and validation or error cases users need to understand.

Formatting rules:

- Keep links and assets relative to the plugin docs directory.
- Put doc assets under the plugin docs tree, such as `docs/assets/example.png`.
- Use fenced code blocks with language tags.
- Use Markdown tables for config references.
- Keep heading levels hierarchical.
- Make the package `pyproject.toml` description concise and user-facing because it feeds generated plugin cards.
- Verify the `[project.entry-points."data_designer.plugins"]` key is the column type users configure.

Regenerate and validate docs after plugin docs or metadata changes:

```bash
make plugin-docs
make docs
```

## Generated Files

Use current target names:

```bash
make plugin-docs
make codeowners
make update-license-headers
make check
```

There is no `make catalog` target in this repo.

## Validation

Prefer repo targets:

```bash
make format
make lint
make test-plugin PLUGIN=data-designer-my-plugin
make validate
make check
make docs
```

Run `make all` before the PR when feasible. If a full target cannot be run, report exactly what was skipped and why.
