# Adding a Plugin

## 1. Scaffold

```bash
uv run scaffold-plugin my-plugin
```

Generates `plugins/data-designer-my-plugin/` with all required files. Your git email is auto-detected for the CODEOWNERS file.

Generated structure:

```
plugins/data-designer-my-plugin/
├── pyproject.toml              # Package metadata + entry point registration
├── CODEOWNERS                  # Plugin ownership (auto-populated)
├── tests/
│   └── test_plugin.py          # Validation test stub
└── src/
    └── data_designer_my_plugin/
        ├── __init__.py
        ├── config.py           # Column config (params, dependencies, emoji)
        ├── impl.py             # Generation logic
        └── plugin.py           # Wires config + impl for discovery
```

## 2. Implement

Three files need your logic:

**config.py**: Subclass `SingleColumnConfig`. Define your `column_type` as a `Literal` string, add config parameters, and declare column dependencies.

```python
from typing import Literal
from data_designer.config.base import SingleColumnConfig

class MyPluginColumnConfig(SingleColumnConfig):
    column_type: Literal["my-plugin"] = "my-plugin"

    source_column: str
    threshold: float = 0.5

    @staticmethod
    def get_column_emoji() -> str:
        return "🔌"

    @property
    def required_columns(self) -> list[str]:
        return [self.source_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []
```

**impl.py**: Subclass a generator base class and implement `generate()`.

```python
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn
from data_designer_my_plugin.config import MyPluginColumnConfig

class MyPluginColumnGenerator(ColumnGeneratorFullColumn[MyPluginColumnConfig]):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.config.name] = data[self.config.source_column] * self.config.threshold
        return data
```

Two base classes are available.

`ColumnGeneratorFullColumn` receives and returns a `pd.DataFrame`. Use it for vectorized column operations. `ColumnGeneratorCellByCell` receives and returns a `dict` (one row) and supports `max_parallel_requests` for concurrency. Use it for row-level transforms, especially model calls.

**plugin.py** is already wired by the scaffolder. Update the qualified names only if you rename your classes.

## 3. Test

```bash
uv sync --all-packages
uv run pytest plugins/data-designer-my-plugin/tests/ -v
```

The scaffolded test validates plugin structure via `assert_valid_plugin`. Add functional tests for your generation logic.

```python
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer_my_plugin.plugin import plugin

def test_valid_plugin():
    assert_valid_plugin(plugin)
```

## 4. Regenerate Metadata

```bash
python tools/generate_catalog.py > docs/catalog.md
python tools/aggregate_codeowners.py > CODEOWNERS
```

CI will reject your MR if these are stale.

## 5. Submit

```bash
git checkout -b feature/my-plugin
git add plugins/data-designer-my-plugin/ docs/catalog.md CODEOWNERS
git commit -m "feat: add my-plugin"
git push -u origin feature/my-plugin
glab mr create
```

CI runs four checks on your MR: lint (ruff), isolated install + pytest per plugin, `assert_valid_plugin` on all entry points, and catalog/CODEOWNERS freshness.

## Entry Point Discovery

Plugins register via `[project.entry-points."data_designer.plugins"]` in `pyproject.toml`. The key is your column type slug; the value points to the `Plugin` instance. Data Designer discovers all installed plugins automatically through this mechanism.

```toml
[project.entry-points."data_designer.plugins"]
my-plugin = "data_designer_my_plugin.plugin:plugin"
```
