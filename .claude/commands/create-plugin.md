---
description: Create a new Data Designer plugin with correct structure, docs, metadata, and local validation
argument-hint: Plugin slug and description (e.g., "word-count - counts words in text columns")
---

# Data Designer Plugin Authoring

You are creating a Data Designer plugin in the `DataDesignerPlugins` monorepo. Use the repository tooling as the source of truth, keep the plugin self-contained, and prepare the per-plugin documentation so the Zensical site generation stays clean.

**Plugin request:** $ARGUMENTS

---

## Phase 1: Read Current Repo Context

Before editing, read the current repository guidance and reference implementation. Use Claude Read/Glob tools for file exploration when possible.

Required reads:

1. `AGENTS.md` - repo conventions, workflow, PR expectations, and release guardrails.
2. `README.md` - current quick start, Makefile targets, and `ddp` CLI overview.
3. `docs/authoring.md` - plugin authoring guide.
4. `docs/workflow.md` - local checks, generated docs, and CI expectations.
5. `Makefile` - canonical target names.
6. `zensical.toml` - site configuration and generated plugin docs navigation block.
7. `devtools/ddp/src/ddp/scaffold.py` - current scaffold output.
8. `devtools/ddp/src/ddp/plugin_docs.py` - how per-plugin docs become Zensical pages.
9. `plugins/data-designer-template/` - reference package, especially `config.py`, `impl.py`, `plugin.py`, `tests/test_plugin.py`, `pyproject.toml`, and `docs/`.

After `make sync`, inspect the Data Designer interfaces you plan to implement instead of guessing signatures:

```bash
uv run python -c "import inspect; from data_designer.config.base import SingleColumnConfig; print(inspect.getsource(SingleColumnConfig))"
uv run python -c "import inspect; from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn; print(inspect.getsource(ColumnGeneratorFullColumn))"
```

---

## Phase 2: Scaffold With `ddp`

The initial plugin structure is owned by the repo's `ddp` CLI. Always invoke the scaffold command; do not create the package directory, `pyproject.toml`, source package, tests, docs, or ownership files by hand.

```bash
make sync
uv run ddp new <slug>
```

Use the kebab-case slug without the `data-designer-` prefix. If you need to understand exactly what the command creates, read `devtools/ddp/src/ddp/scaffold.py` or inspect the generated files after running the command. The skill should not duplicate the scaffold algorithm; the software encodes that process deterministically.

If the command fails because the scaffold is wrong or incomplete, fix the `ddp` tooling or report the blocker. Do not bypass it by hand-assembling the initial plugin skeleton.

After scaffolding, read the generated files before editing them. If the slug contains words such as `column` and the generated class names stutter, rename the classes and update `plugin.py`.

---

## Phase 3: Implement

### Config

Subclass `SingleColumnConfig`. Use Python 3.10+ annotations and a literal column type default:

```python
from typing import Literal

from data_designer.config.base import SingleColumnConfig


class MyPluginColumnConfig(SingleColumnConfig):
    """Configuration for the my-plugin column generator."""

    column_type: Literal["my-plugin"] = "my-plugin"

    @staticmethod
    def get_column_emoji() -> str:
        return "..."

    @property
    def required_columns(self) -> list[str]:
        return ["source_column"]

    @property
    def side_effect_columns(self) -> list[str]:
        return []
```

Add Pydantic validators for structural constraints such as non-empty lists, parsable patterns, allowed modes, paths, or field combinations. Catch invalid config at construction time, not inside `generate()`.

### Implementation

Use the Data Designer generator base that matches the behavior, usually `ColumnGeneratorFullColumn[YourConfig]` for whole-column transformations.

Implementation rules:

- Keep plugin logic in top-level composable functions or small modules; keep `impl.py` mostly orchestration.
- Do not use relative imports. Import from the package name, for example `from data_designer_my_plugin.config import MyPluginColumnConfig`.
- Do not define private helper closures or functions inside functions.
- Prefer vectorized pandas operations, named helpers, `functools.partial`, or module-level dispatch tables over lambda-heavy `apply()` code.
- Use `from __future__ import annotations` and guard pandas imports with `TYPE_CHECKING` when pandas is only needed for type hints.
- Add Google-style docstrings to public classes, functions, and methods.
- Keep dependencies in the plugin's own `pyproject.toml`. Do not depend on another local plugin package.

If you add a new import package, add it to the root Ruff isort list:

```toml
[tool.ruff.lint.isort]
known-first-party = ["ddp", "data_designer_template", "data_designer_my_plugin"]
```

### Plugin Wiring

The scaffold usually gets `plugin.py` right. Only update it when class or module names change:

```python
from data_designer.plugins.plugin import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="data_designer_my_plugin.config.MyPluginColumnConfig",
    impl_qualified_name="data_designer_my_plugin.impl.MyPluginColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
```

### CODEOWNERS

The scaffold discovers an owner from git config. Check the per-plugin `CODEOWNERS` and prefer the repo convention from the template:

```text
* @NVIDIA-NeMo/data_designer_reviewers
```

Run `make codeowners` after ownership changes so `.github/CODEOWNERS` is regenerated.

---

## Phase 4: Test Public Behavior

Write tests around public interfaces and expected Data Designer behavior:

```python
from data_designer.engine.testing.utils import assert_valid_plugin

from data_designer_my_plugin.plugin import plugin


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)
```

Cover the relevant tiers:

- Config properties, defaults, and validation errors.
- Pure helper functions or parsing/scoring modules.
- Generator behavior against representative DataFrames.
- Data Designer preview integration when the plugin changes user-visible pipeline behavior.
- Edge cases with `None`, `NaN`, empty strings, numeric values in text columns, or malformed config.

If using pytest's `tmp_path`, annotate it as `pathlib.Path`, not `pd.DataFrame`.

Run the isolated plugin test loop while developing:

```bash
make test-plugin PLUGIN=data-designer-my-plugin
```

The target uses `uv venv --clear`, so stale `.venv-data-designer-my-plugin` directories should not need manual cleanup.

---

## Phase 5: Prepare Per-Plugin Zensical Docs

Each plugin owns its source docs under `plugins/data-designer-<slug>/docs/`. The top-level Zensical site is generated from those files and package metadata.

Required source docs:

```text
plugins/data-designer-<slug>/docs/
`-- index.md
```

Recommended docs for user-facing plugins:

```text
plugins/data-designer-<slug>/docs/
|-- index.md
`-- usage.md
```

Write `docs/index.md` as the plugin overview:

- H1: `# data-designer-<slug>`.
- One short paragraph explaining what the plugin adds.
- Installation command using `uv add data-designer data-designer-<slug>`.
- Column type section naming the discovered entry point, for example `` `<slug>` ``.
- Configuration table with `Field`, `Required`, and `Description` columns.
- A realistic Python or Data Designer config example.
- Important behavior notes, limitations, or output columns only when useful.

Write `docs/usage.md` when the plugin needs a fuller example:

- H1: `# Usage` or another concise title; non-index H1 text becomes the Zensical nav label.
- A runnable or realistic example using `DataDesignerConfigBuilder`, a YAML-style config, or both.
- Expected output shape or before/after behavior.
- Error cases and config validation notes that users should know before running a job.

Zensical formatting rules for this repo:

- Keep links and assets relative to the plugin's own `docs/` directory; generated pages are copied to `docs/plugins/data-designer-<slug>/`.
- Store plugin doc assets under the plugin docs tree, for example `plugins/data-designer-<slug>/docs/assets/example.png`.
- Use fenced code blocks with language tags such as `python`, `yaml`, `toml`, or `bash`.
- Use Markdown tables for config references.
- Keep headings hierarchical and avoid skipping from H1 to H3.
- Do not edit `docs/plugins/` directly. It is generated.
- Do not edit the generated plugin nav block in `zensical.toml` directly.
- Remember that package metadata feeds the generated plugin index card: keep `pyproject.toml` `description` concise and user-facing, and verify the `data_designer.plugins` entry point key is the column type users configure.

Regenerate and validate site inputs after plugin docs or metadata change:

```bash
make plugin-docs
make docs
```

---

## Phase 6: Regenerate Derived Files

Use current target names. There is no `make catalog` target in this repo.

When plugin docs or package metadata change:

```bash
make plugin-docs
```

When plugin ownership changes:

```bash
make codeowners
```

When Python files are added or changed:

```bash
make update-license-headers
```

`make check` verifies generated plugin docs, generated CODEOWNERS, and SPDX headers:

```bash
make check
```

---

## Phase 7: Local Validation

Prefer the repo's Makefile targets over ad hoc substitutes.

Fast loop:

```bash
make format
make lint
make test-plugin PLUGIN=data-designer-my-plugin
make validate
make check
make docs
```

Full local CI:

```bash
make all
```

---

## Anti-Pattern Checklist

Before opening the PR, verify you have not done any of these:

- Skipped `uv run ddp new <slug>` and hand-created the plugin structure.
- Treated this command as a copy of the scaffold algorithm instead of delegating the initial structure to `ddp`.
- Used `docs/adding-a-plugin.md`; the current guide is `docs/authoring.md`.
- Used `make catalog`; the current generated docs target is `make plugin-docs`.
- Edited generated files under `docs/plugins/` manually.
- Edited the generated plugin nav block in `zensical.toml` manually.
- Forgot to run `make plugin-docs` after plugin docs or package metadata changes.
- Forgot to run `make codeowners` after per-plugin ownership changes.
- Left `pyproject.toml` with a generic scaffold description.
- Left `docs/index.md` as generic scaffold text for a user-facing plugin.
- Used relative imports.
- Added local plugin-to-plugin dependencies.
- Used `typing.List`, `typing.Optional`, or `typing.Union` instead of Python 3.10+ annotations.
- Added nested helper functions or private helper closures.
- Deferred structural config validation to `generate()`.
- Missed tests for invalid config or null-like source values.
