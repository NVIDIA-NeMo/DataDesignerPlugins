# NeMo Data Designer Plugins

[![Documentation](https://img.shields.io/badge/docs-documentation-blue)](https://nvidia-nemo.github.io/DataDesignerPlugins/)

First-class NVIDIA-provided plugins for [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).
This repository is the NVIDIA-maintained curated first-party plugin catalog for
Data Designer.

## Quick Start

```bash
git clone git@github.com:NVIDIA-NeMo/DataDesignerPlugins.git
cd DataDesignerPlugins
make sync
```

Create a new plugin:

```bash
uv run ddp new my-plugin
```

This generates a column generator by default. Pass `--type seed-reader` or
`--type processor` to scaffold those plugin types instead. Each scaffold creates
a complete plugin skeleton under `plugins/data-designer-my-plugin/` with config,
implementation, entry point, docs, tests, and CODEOWNERS. See
[docs/authoring.md](docs/authoring.md) for the full authoring guide.

## Repository Structure

```
DataDesignerPlugins/
|-- devtools/
|   `-- ddp/                          # Monorepo management tooling (ddp CLI, dev-only)
|-- catalog/                          # Machine-consumable plugin catalog data
|-- plugins/                          # One directory per plugin (auto-discovered by uv)
|   `-- data-designer-template/       # Reference implementation
`-- docs/                             # Zensical documentation source
```

Each plugin is an independent Python package with its own `pyproject.toml`, docs, tests, and CODEOWNERS. The root workspace auto-discovers plugins via `plugins/*`.

## Plugin Catalog

The default NVIDIA plugin catalog URL is:

```text
https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json
```

`catalog/plugins.json` is the generated JSON catalog artifact for Data Designer.
The published Pages site serves that catalog together with the static Python
package index at `https://nvidia-nemo.github.io/DataDesignerPlugins/simple/`.
See [docs/catalogs.md](docs/catalogs.md) for catalog discovery, catalog
fields, `install` metadata, trust expectations, and external catalog setup
guidance.

## Development

Use the repo's `Makefile` targets for all development tasks:

```bash
make sync               # Install all packages (uv sync --all-packages)
make lint               # Lint and format check (ruff)
make format             # Auto-fix lint issues and reformat
make test               # Test each plugin in an isolated venv
make validate           # Run assert_valid_plugin on all entry points
make check              # Verify generated plugin docs, catalog, CODEOWNERS, and license headers are up to date
make plugin-docs        # Regenerate docs/plugins/ from per-plugin docs and metadata
make catalog            # Regenerate catalog/plugins.json
make package-index      # Add catalog JSON and the static package index to site/
make qa-package-index   # Build/install a plugin through a scratch local package index
make docs               # Build the Zensical documentation site
make docs-server        # Serve docs locally at http://localhost:8000
make all                # lint + test + validate + check + docs (full local CI)
```

To test a single plugin in isolation:

```bash
make test-plugin PLUGIN=data-designer-my-plugin
```

If you change plugin docs, plugin metadata, or ownership, regenerate derived files:

```bash
make plugin-docs              # Regenerate plugin documentation site inputs
make catalog                  # Regenerate catalog/plugins.json
make codeowners               # Regenerate CODEOWNERS
make update-license-headers   # Fix SPDX headers
```

## The `ddp` CLI

The `ddp` command manages the monorepo. Run `uv run ddp --help` to see all subcommands:

| Command | Description |
|---------|-------------|
| `ddp new <name>` | Scaffold a new plugin |
| `ddp sync catalog` | Sync the static plugin catalog JSON |
| `ddp validate` | Validate all installed plugins |
| `ddp plugin-docs` | Generate plugin docs site inputs |
| `ddp package-index` | Build, validate, merge, and QA the static package index |
| `ddp codeowners` | Aggregate CODEOWNERS to stdout |
| `ddp license-headers` | Add or check SPDX license headers |
| `ddp bump <plugin> <part>` | Bump a plugin's semantic version |
| `ddp check-release <plugin> <version>` | Validate plugin metadata for release |

## Releasing

```bash
make bump PLUGIN=data-designer-my-plugin PART=patch   # Bump version (major/minor/patch)
git add plugins/data-designer-my-plugin/pyproject.toml
git commit -m "chore(data-designer-my-plugin): bump version to 0.1.1"
make release PLUGIN=data-designer-my-plugin            # Build and tag for GitHub Release publishing
git push origin data-designer-my-plugin/v0.1.1
gh release create data-designer-my-plugin/v0.1.1       # Triggers CI publish
```

See [docs/releasing.md](docs/releasing.md) for the full release guide.

## License

Apache-2.0. See [LICENSE](LICENSE).
