# 🔌 🎨 NeMo Data Designer Plugins

First-class NVIDIA-provided plugins for [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).

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

This generates a complete plugin skeleton under `plugins/data-designer-my-plugin/` with config, implementation, entry point, tests, and CODEOWNERS. See [docs/adding-a-plugin.md](docs/adding-a-plugin.md) for the full authoring guide.

## Repository Structure

```
DataDesignerPlugins/
├── devtools/
│   └── ddp/                          # Monorepo management tooling (ddp CLI, dev-only)
├── plugins/                          # One directory per plugin (auto-discovered by uv)
│   └── data-designer-template/       # Reference implementation
└── docs/                             # Authoring guide, plugin catalog
```

Each plugin is an independent Python package with its own `pyproject.toml`, tests, and CODEOWNERS. The root workspace auto-discovers plugin packages via `plugins/*`. A package can provide one or more Data Designer plugins through the `data_designer.plugins` group.

## Development

Use the repo's `Makefile` targets for all development tasks:

```bash
make sync               # Install all packages (uv sync --all-packages)
make lint               # Lint and format check (ruff)
make format             # Auto-fix lint issues and reformat
make test               # Test each plugin in an isolated venv
make validate           # Run assert_valid_plugin on all entry points
make check              # Verify catalog, CODEOWNERS, and license headers are up to date
make all                # lint + test + validate + check (full local CI)
```

To test a single plugin in isolation:

```bash
make test-plugin PLUGIN=data-designer-my-plugin
```

If you change plugin metadata or ownership, regenerate derived files:

```bash
make catalog                  # Regenerate docs/catalog.md
make codeowners               # Regenerate CODEOWNERS
make update-license-headers   # Fix SPDX headers
```

## The `ddp` CLI

The `ddp` command manages the monorepo. Run `uv run ddp --help` to see all subcommands:

| Command | Description |
|---------|-------------|
| `ddp new <name>` | Scaffold a new plugin |
| `ddp validate` | Validate all installed plugins |
| `ddp catalog` | Generate plugin catalog to stdout |
| `ddp codeowners` | Aggregate CODEOWNERS to stdout |
| `ddp license-headers` | Add or check SPDX license headers |
| `ddp bump <plugin> <part>` | Bump a plugin's semantic version |
| `ddp check-release <plugin> <version>` | Validate plugin metadata for release |

## Releasing

```bash
make bump PLUGIN=data-designer-my-plugin PART=patch   # Bump version (major/minor/patch)
git add plugins/data-designer-my-plugin/pyproject.toml
git commit -m "chore(data-designer-my-plugin): bump version to 0.1.1"
make release PLUGIN=data-designer-my-plugin            # Tag + build
git push origin data-designer-my-plugin/v0.1.1         # Triggers CI publish
```

See [docs/adding-a-plugin.md](docs/adding-a-plugin.md) for the full release guide.

## License

Apache-2.0. See [LICENSE](LICENSE).
