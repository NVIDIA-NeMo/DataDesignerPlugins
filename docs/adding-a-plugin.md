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

## 6. Release to PyPI

Once your plugin's MR is merged to `main`, you can publish it to PyPI. CI handles the actual upload; you just bump the version, tag, and push.

### First-time setup: register your package name

PyPI claims package names on first upload (there's no separate registration step). Before your first release, verify the name is available:

1. Go to `https://pypi.org/project/data-designer-my-plugin/` and confirm you get a 404 (name is free). All plugins in this repo use the `data-designer-` prefix.

2. Check your `pyproject.toml` metadata. The scaffolder populates the required fields (`description`, `license`, `readme`, `authors`), but review them before your first publish:

   ```toml
   [project]
   name = "data-designer-my-plugin"
   version = "0.1.0"
   description = "One-line description of what your plugin does"
   license = "Apache-2.0"
   readme = "README.md"
   authors = [
       {name = "NVIDIA Corporation"},
   ]
   ```

3. Write a `README.md` in your plugin directory. This is what users see on the PyPI page. The scaffolder creates one, but customize it with usage examples specific to your plugin.

4. Do a local dry-run to make sure everything builds:

   ```bash
   make build-plugin PLUGIN=data-designer-my-plugin
   ```

   This validates metadata and produces a wheel and sdist in `dist/`. Inspect the wheel if you want to verify contents:

   ```bash
   unzip -l dist/data_designer_my_plugin-0.1.0-py3-none-any.whl
   ```

### Publishing a release

```bash
make release PLUGIN=data-designer-my-plugin
git push origin data-designer-my-plugin/v0.1.0
```

`make release` runs the full local pipeline (ownership check, tests, validation, build), creates a git tag, and prints the push command. Pushing the tag triggers the CI publish job, which uploads to PyPI.

The same process works for all subsequent releases. Bump `version` in `pyproject.toml`, merge to `main`, then run `make release` again.

### What `make release` does

First it compares your git email against the plugin's `CODEOWNERS` and warns if you're not listed. Then it installs the plugin in an isolated venv and runs pytest, validates that `pyproject.toml` has all required PyPI fields with a consistent version, and builds the wheel and sdist into `dist/`. Finally it creates a git tag `data-designer-my-plugin/v<version>` from the version in `pyproject.toml`.

### What CI does when you push the tag

1. Validates the plugin directory exists and the tagged commit is on `main`.
2. Runs `validate_release.py` (version match + metadata check).
3. Checks CODEOWNERS (hard gate). The tag pusher must be listed in the plugin's `CODEOWNERS` file.
4. Installs and tests the plugin in an isolated venv.
5. Builds and uploads to PyPI using the repo's `PYPI_TOKEN`.

### Tag convention

Tags follow the pattern `<plugin-package-name>/v<version>`:

```
data-designer-my-plugin/v0.1.0
data-designer-my-plugin/v0.2.0a1
data-designer-my-plugin/v1.0.0
```

Each plugin is tagged and released independently. Releasing one plugin doesn't affect any others.

### Pre-release versions

Use PEP 440 pre-release suffixes in your `pyproject.toml` version:

```toml
version = "0.2.0a1"   # alpha
version = "0.2.0b1"   # beta
version = "0.2.0rc1"  # release candidate
```

Pre-release versions won't be installed by default with `pip install`. Users must explicitly request them with `pip install data-designer-my-plugin==0.2.0a1` or `pip install --pre data-designer-my-plugin`.

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR: PLUGIN_DIR not found` | Check `PLUGIN=` matches the directory name under `plugins/` |
| Version mismatch | The tag version must match `project.version` in `pyproject.toml` |
| CODEOWNERS failure in CI | Add your GitLab `@username` or email to the plugin's `CODEOWNERS` file |
| `ERROR: tagged commit is not on main` | Tags must point to commits on the `main` branch |
| Package name taken on PyPI | Choose a different name; all plugins must use the `data-designer-` prefix |
| `PYPI_TOKEN` not set | A repo maintainer needs to add the `PYPI_TOKEN` CI/CD variable in GitLab project settings |

## Entry Point Discovery

Plugins register via `[project.entry-points."data_designer.plugins"]` in `pyproject.toml`. The key is your column type slug; the value points to the `Plugin` instance. Data Designer discovers all installed plugins automatically through this mechanism.

```toml
[project.entry-points."data_designer.plugins"]
my-plugin = "data_designer_my_plugin.plugin:plugin"
```
