# Data Designer Plugins

First-class NVIDIA-provided plugins for [NeMo Data Designer](https://github.com/NVIDIA/NeMo-Data-Designer).

## Setup

```bash
# Clone alongside DataDesigner
git clone <repo-url> data-designer-plugins
cd data-designer-plugins

# Install all packages in development mode
uv sync --all-packages

# Run tests
uv run pytest plugins/data-designer-template/tests/ -v

# Lint
uv run ruff check .
uv run ruff format --check .
```

Requires a local checkout of DataDesigner at `../DataDesigner/`.

## Adding a Plugin

See [docs/adding-a-plugin.md](docs/adding-a-plugin.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
