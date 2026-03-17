# Data Designer Plugins

First-class NVIDIA-provided plugins for [NeMo Data Designer](https://github.com/NVIDIA/NeMo-Data-Designer).

## Quick Start

```bash
git clone git@gitlab-master.nvidia.com:etramel/data-designer-plugins.git
cd data-designer-plugins
uv sync --all-packages
```

Create a new plugin:

```bash
uv run scaffold-plugin my-plugin
```

This generates a complete plugin skeleton under `plugins/data-designer-my-plugin/` with config, implementation, entry point, tests, and CODEOWNERS. See [docs/adding-a-plugin.md](docs/adding-a-plugin.md) for the full authoring guide.

## Repository Structure

```
data-designer-plugins/
├── core/                     # Shared utilities (scaffold CLI, zero DD deps)
├── plugins/                  # One directory per plugin (auto-discovered by uv)
│   └── data-designer-template/   # Reference implementation
├── tools/                    # Repo automation (catalog, CODEOWNERS aggregation)
└── docs/                     # Authoring guide, plugin catalog
```

Each plugin is an independent Python package with its own `pyproject.toml`, tests, and CODEOWNERS. The root workspace auto-discovers plugins via `plugins/*`.

## Development

```bash
uv sync --all-packages          # Install everything
uv run pytest plugins/ -v       # Test all plugins
uv run ruff check .             # Lint
uv run ruff format --check .    # Format check
```

## License

Apache-2.0. See [LICENSE](LICENSE).
