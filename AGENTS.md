# Data Designer Plugins

# Repo Facts

This repo is a `uv` workspace with shared tooling in `core/` and one independent package per plugin under `plugins/*`.
The Python compatibility baseline here is 3.10+, not 3.11-only.
Use `plugins/data-designer-template/` as the reference implementation before inventing new structure.

Example layout:

```text
core/data-designer-plugins-core/
plugins/data-designer-template/
plugins/data-designer-my-plugin/
docs/
```

# Creating a New Plugin

If asked to create a new plugin, use this repo's tooling first and refer to `README.md` and `docs/` only as supporting detail.
Prefer the scaffold CLI over hand-creating files.
All plugins should be entirely self-contained and manage themselves.
Plugins should not depend on one another locally, however plugins may depend on publicly released PyPI `data-designer-*` plugins.

Canonical scaffold flow:

```bash
make sync
uv run ddp new my-plugin
```

This creates a package like:

```text
plugins/data-designer-my-plugin/
src/data_designer_my_plugin/
tests/test_plugin.py
CODEOWNERS
```

Naming convention example:

```toml
[project]
name = "data-designer-my-plugin"

[project.entry-points."data_designer.plugins"]
my-plugin = "data_designer_my_plugin.plugin:plugin"
```

# Understanding DataDesigner

If in doubt, refer to the DataDesigner codebase and implementation over documentation.
Rather than conducting slow review via Github urls, clone the latest (or appropriate) release of DataDesigner to a permission-allowed, accessible tmp directory for exploration.


# Development Workflow

When creating or updating a plugin, make your edits within a worktree.
Prefer the repo's canonical `Makefile` targets over ad hoc substitutes.
Test your changes locally and ensure that you have a locally green CI by running the Makefile CI commands.
Upon completion, submit a pull request using the `gh` cli.

Canonical local workflow:

```bash
make sync
make lint
make test
make validate
make check
make all
```

Notes:

- `make test` installs each plugin in an isolated venv and runs its tests there.
- `make validate` discovers installed `data_designer.plugins` entry points and runs `assert_valid_plugin`.
- `make check` verifies generated metadata and SPDX headers.

If you change plugin metadata or ownership, regenerate the derived files:

```bash
make catalog
make codeowners
make check-license-headers
# or, to fix headers:
make update-license-headers
```

Pull request example:

```bash
gh pr create
```

# Development Style

Tests should be written around public interfaces.
Use modern Python 3.10+ style type annotations such as `list[str]`, `A | B`, and `X | None`.
Utilize full Google-style docstrings for implemented functionality.
Do not use private helper closures or function-in-function definitions.
Favor reusable, composable functions that can be combined in higher-level functions.
Keep function and method definitions short and legible, deferring to composition rather than nesting.
Use Ruff via the repo targets for linting and formatting.
Relative imports are banned in this repo.

Validation test example:

```python
from data_designer.engine.testing.utils import assert_valid_plugin

from data_designer_my_plugin.plugin import plugin


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)
```

Import style example:

```python
from data_designer_my_plugin.config import MyPluginColumnConfig
# Avoid:
# from .config import MyPluginColumnConfig
```

# Version Bumps and Releases

Never release or publish a plugin version as a tag or to PyPI without being asked or having express permission from the user.

Use `make bump` to increment the version, then `make release` to tag. Do not edit `pyproject.toml` version strings by hand.

```bash
# 1. Bump version (defaults to patch; use PART=minor or PART=major as needed)
make bump PLUGIN=data-designer-my-plugin PART=patch

# 2. Commit the version change
git add plugins/data-designer-my-plugin/pyproject.toml
git commit -m "chore(data-designer-my-plugin): bump version to 0.2.0"

# 3. Tag and release (runs tests, validation, and build)
make release PLUGIN=data-designer-my-plugin
git push origin data-designer-my-plugin/v0.2.0
```

Release facts:

- Tags are per-plugin: `data-designer-my-plugin/v0.1.0`.
- Release CI expects the tagged commit to be on `main`.
- The tag pusher must be listed in that plugin's `CODEOWNERS`.
- Pre-release versions (e.g. `0.2.0a1`) are not supported by `ddp bump`; edit `pyproject.toml` manually for those.

# References

- [DataDesigner GitHub](https://github.com/NVIDIA-NeMo/DataDesigner)
- [DataDesigner Latest Release Notes](https://github.com/NVIDIA-NeMo/DataDesigner/releases/latest)
- [DataDesigner Plugin Authoring Guide](https://nvidia-nemo.github.io/DataDesigner/latest/plugins/overview/)
- [data-designer-plugins authoring guide](docs/adding-a-plugin.md)
- [data-designer-plugins plugin catalog](docs/catalog.md)
