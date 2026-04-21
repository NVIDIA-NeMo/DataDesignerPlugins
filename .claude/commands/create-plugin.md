---
description: Create a new DataDesigner plugin with correct structure, conventions, and passing CI
argument-hint: Plugin name and description (e.g., "word-count - counts words in text columns")
---

# DataDesigner Plugin Authoring

You are creating a new DataDesigner plugin in the `data-designer-plugins` monorepo. Follow this guide precisely. It encodes lessons learned from prior plugin authoring attempts and addresses common pitfalls.

**Plugin request:** $ARGUMENTS

---

## Phase 1: Understand the Codebase (DO NOT SKIP)

Before writing any code, you must build context. Use dedicated tools (Read, Glob) rather than Bash for file exploration.

**Required reads** (in parallel):

1. `CLAUDE.md` -- repo conventions (already in system context, but re-read for specifics)
2. `plugins/data-designer-template/src/data_designer_template/config.py` -- reference config
3. `plugins/data-designer-template/src/data_designer_template/impl.py` -- reference implementation
4. `plugins/data-designer-template/src/data_designer_template/plugin.py` -- reference wiring
5. `plugins/data-designer-template/tests/test_plugin.py` -- reference tests
6. `plugins/data-designer-template/pyproject.toml` -- reference packaging
7. `docs/adding-a-plugin.md` -- full authoring guide (agents often skip this -- don't)

**Required introspection** (after `make sync`):

```bash
uv run python -c "import inspect; from data_designer.config.base import SingleColumnConfig; print(inspect.getsource(SingleColumnConfig))"
uv run python -c "import inspect; from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn; print(inspect.getsource(ColumnGeneratorFullColumn))"
```

This tells you the exact interface you must implement. Do not guess at method signatures.

---

## Phase 2: Scaffold

Always use the canonical scaffold tool. Never hand-create the plugin directory structure.

```bash
make sync
uv run ddp new <slug>
```

After scaffolding, read all generated files to see what the scaffold provides and what you need to modify.

---

## Phase 3: Implement

### 3a. Config (`config.py`)

Subclass `SingleColumnConfig`. Required elements:

```python
from typing import Literal
from data_designer.config.base import SingleColumnConfig

class MyPluginColumnConfig(SingleColumnConfig):
    column_type: Literal["my-plugin"] = "my-plugin"  # Must be Literal with default

    # Your config fields here (use modern 3.10+ annotations: list[str], X | None)

    @staticmethod
    def get_column_emoji() -> str:
        return "..."  # Single emoji

    @property
    def required_columns(self) -> list[str]:
        return [...]  # Columns that must exist before this one runs

    @property
    def side_effect_columns(self) -> list[str]:
        return []  # Additional columns this generator creates (usually empty)
```

**Common mistake -- class naming**: If your plugin slug already contains "column" (e.g., "hash-column"), the scaffold generates `HashColumnColumnConfig` with a stutter. Rename to `HashColumnConfig` / `HashColumnGenerator` and update `plugin.py` qualified names accordingly.

**Validate early with `field_validator`**: If your config has fields with structural constraints (e.g., a regex that must contain capture groups, a list that must be non-empty, a string that must parse as a certain format), add a Pydantic `@field_validator` so errors are caught at config construction time, not at `generate()` time. Deferring validation to `generate()` means users only discover bad config after they've wired up an entire pipeline.

```python
from pydantic import field_validator

class MyPluginColumnConfig(SingleColumnConfig):
    pattern: str

    @field_validator("pattern")
    @classmethod
    def pattern_must_be_valid(cls, value: str) -> str:
        """Validate the pattern at config construction time."""
        compiled = re.compile(value)
        if compiled.groups < 1:
            raise ValueError(f"Pattern must contain at least one capture group, got: {value!r}")
        return value
```

Do not duplicate this validation logic in `impl.py`. If the config validates on construction, `generate()` can trust the field is valid.

### 3b. Implementation (`impl.py`)

Subclass `ColumnGeneratorFullColumn[YourConfig]` (batch) or `ColumnGeneratorCellByCell[YourConfig]` (row-by-row).

**Critical rules:**

1. **NO lambda closures.** CLAUDE.md bans closures and function-in-function definitions. This is the most common violation.

   BAD (every prior agent did this):
   ```python
   data[self.config.name] = data[col].apply(lambda x: my_func(x, param))
   ```

   GOOD -- use `functools.partial`:
   ```python
   from functools import partial
   data[self.config.name] = data[col].apply(partial(my_func, param=param))
   ```

   GOOD -- use vectorized pandas operations when possible:
   ```python
   data[self.config.name] = data[col].str.upper()
   ```

   GOOD -- use a module-level dispatch dict:
   ```python
   _MODE_FUNCTIONS: dict[str, Callable[[str], int]] = {
       "words": count_words,
       "characters": count_characters,
   }
   # In generate():
   data[self.config.name] = data[col].apply(_MODE_FUNCTIONS[self.config.mode])
   ```

2. **Extract logic into top-level composable functions**, not methods on the generator class. This follows the CLAUDE.md rule: "Favor reusable, composable functions that can be combined in higher-level functions."

   **But avoid leaky abstractions.** If a helper function accepts a compiled object (e.g., `re.Pattern`) but then extracts the raw string from it to pass to another API (e.g., `series.str.extract(pattern.pattern)`), the abstraction is misleading. Either accept the raw form the downstream API needs, or use the compiled object directly.

3. **Use `TYPE_CHECKING` guard for pandas**:
   ```python
   from __future__ import annotations
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       import pandas as pd
   ```

4. **Full Google-style docstrings** on all public functions, methods, and classes.

5. **No relative imports.** Use `from data_designer_my_plugin.config import MyPluginColumnConfig`.

### 3c. Plugin wiring (`plugin.py`)

The scaffold generates this correctly. Only update if you renamed classes:

```python
plugin = Plugin(
    config_qualified_name="data_designer_my_plugin.config.MyPluginColumnConfig",
    impl_qualified_name="data_designer_my_plugin.impl.MyPluginColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
```

### 3d. Extra modules

If your plugin has substantial pure logic (scoring, parsing, transformation), extract it into a separate module (e.g., `scoring.py`). Keep `impl.py` thin -- it should wire config to logic, not contain the logic itself.

### 3e. Root `pyproject.toml`

Add your module to the isort known-first-party list:

```toml
[tool.ruff.lint.isort]
known-first-party = [..., "data_designer_my_plugin"]
```

### 3f. CODEOWNERS

The scaffold generates this from `git config user.email`. Check that it uses `@username` or `@org/team` format (e.g., `* @NVIDIA-NeMo/data_designer_reviewers`), not email format. If it used email, fix it to match the convention in the template's CODEOWNERS.

---

## Phase 4: Test

### Test structure

Write four tiers of tests, matching the template's patterns:

```python
# Tier 1: Plugin contract validation
def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)

# Tier 2: Config unit tests
class TestMyPluginColumnConfig:
    def test_required_columns(self) -> None: ...
    def test_side_effect_columns(self) -> None: ...
    def test_column_emoji(self) -> None: ...
    def test_defaults(self) -> None: ...

# Tier 3: Generator unit tests (using __new__ bypass pattern)
def _make_generator(config: MyPluginColumnConfig) -> MyPluginColumnGenerator:
    generator = MyPluginColumnGenerator.__new__(MyPluginColumnGenerator)
    generator._config = config
    return generator

class TestMyPluginColumnGenerator:
    @pytest.fixture()
    def source_df(self) -> pd.DataFrame:
        return pd.DataFrame({...})

    def test_basic_generation(self, source_df: pd.DataFrame) -> None:
        generator = _make_generator(MyPluginColumnConfig(name="out", ...))
        result = generator.generate(source_df)
        assert "out" in result.columns
        ...

# Tier 4: Integration tests using DataDesigner.preview()
class TestMyPluginPreviewIntegration:
    def test_preview_basic(self, tmp_path: Path) -> None:
        seed_df = pd.DataFrame({...})
        builder = DataDesignerConfigBuilder()
        builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
        builder.add_column(name="out", column_type="my-plugin", ...)
        result = DataDesigner(artifact_path=tmp_path / "artifacts").preview(builder, num_records=3)
        assert result.dataset is not None
        assert "out" in result.dataset.columns
```

### Test pitfalls to avoid

1. **`tmp_path` type annotation**: The pytest `tmp_path` fixture is `pathlib.Path`, NOT `pd.DataFrame`. The template has this bug -- do not copy it.
   ```python
   from pathlib import Path
   def test_preview(self, tmp_path: Path) -> None:  # CORRECT
   # NOT: def test_preview(self, tmp_path: pd.DataFrame) -> None:  # WRONG
   ```

2. **Pandas dtype coercion**: When creating a `pd.Series` with mixed int/float, pandas upcasts ints to floats. `pd.Series({"a": 42, "b": 3.14})` gives `a=42.0`, not `a=42`. Write test expectations accordingly, or use uniform types.

3. **Test composable functions independently**: If you extracted functions to module-level, write dedicated test classes for them (e.g., `TestComputeHash`, `TestTokenize`). This goes beyond the template but produces better coverage.

4. **Test config validation edge cases**: If you added `@field_validator` on config fields, write tests that verify invalid inputs are rejected at construction time:
   ```python
   def test_rejects_invalid_pattern(self) -> None:
       with pytest.raises(ValueError, match="at least one capture group"):
           MyPluginColumnConfig(name="out", source_column="src", pattern=r"\d+")

   def test_rejects_malformed_input(self) -> None:
       with pytest.raises(Exception):  # re.error or ValidationError
           MyPluginColumnConfig(name="out", source_column="src", pattern=r"(unclosed")
   ```

5. **Test edge cases with None and non-string source values**: DataFrames in the wild often have `None`, `NaN`, or numeric values in text columns. Write at least one test that exercises your generator on a DataFrame with `None` values in the source column to verify graceful handling.

6. **Stale venv on test re-run**: `make test-plugin` creates `.venv-{plugin-name}` and fails if it already exists from a prior failed run. If tests fail and you need to re-run:
   ```bash
   rm -rf .venv-data-designer-my-plugin && make test-plugin PLUGIN=data-designer-my-plugin
   ```

---

## Phase 5: Format and Lint First

Import sort order (isort) is the most common lint failure. **Always run `make format` before `make lint`** to avoid wasting a cycle:

```bash
make sync
make format                                            # Fix import order and formatting FIRST
make lint                                              # Should pass after format
```

---

## Phase 6: Test in Isolation

```bash
make test-plugin PLUGIN=data-designer-my-plugin        # Isolated venv test
```

If tests fail and you need to re-run, **delete the stale venv first** (the Makefile does not auto-clean on failure):

```bash
rm -rf .venv-data-designer-my-plugin && make test-plugin PLUGIN=data-designer-my-plugin
```

---

## Phase 7: Validate and Check

```bash
make validate                                          # Entry point + assert_valid_plugin
make catalog && make codeowners && make update-license-headers  # Regenerate derived files
make check                                             # Verify derived files match
make lint                                              # Final lint confirmation
```

---

## Anti-Pattern Checklist

Before declaring done, verify you have NOT done any of these:

- [ ] Lambda closures in `generate()` or anywhere else (use `functools.partial` or dispatch dicts)
- [ ] Relative imports (`from .config import ...`)
- [ ] `tmp_path: pd.DataFrame` in test signatures (should be `from pathlib import Path` then `tmp_path: Path`)
- [ ] Missing SPDX headers on any `.py` file
- [ ] Email format in CODEOWNERS instead of `@username` (read template's CODEOWNERS to match)
- [ ] Missing docstrings on public functions/classes
- [ ] Private helper closures or nested function definitions
- [ ] `typing.List[str]` instead of `list[str]` (3.10+ style required)
- [ ] Missing `from __future__ import annotations` when using `TYPE_CHECKING`
- [ ] Skipped reading `docs/adding-a-plugin.md`
- [ ] Used `find` or `ls` via Bash instead of Glob/Read tools
- [ ] Forgot to add module to `known-first-party` in root `pyproject.toml`
- [ ] Forgot to run `make catalog && make codeowners && make update-license-headers`
- [ ] Forgot to run `make format` BEFORE `make lint` (isort failures are the #1 lint issue)
- [ ] Forgot to delete stale `.venv-*` before re-running `make test-plugin` after a failure
- [ ] Config fields with structural constraints lack `@field_validator` (validate at construction, not at `generate()` time)
- [ ] Helper function accepts a compiled object but then extracts the raw form to pass to a downstream API (leaky abstraction)
- [ ] No tests for `None`/`NaN` values in the source column
- [ ] No tests verifying that invalid config field values are rejected at construction time
