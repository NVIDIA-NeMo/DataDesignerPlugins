# NeMo Data Designer Plugins

[![Documentation](https://img.shields.io/badge/docs-documentation-blue)](https://nvidia-nemo.github.io/DataDesignerPlugins/)

First-class NVIDIA-provided plugins for [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).
This repository is the NVIDIA-maintained curated first-party plugin tap for
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

## Plugin Tap Catalog

The default NVIDIA plugin tap catalog URL is:

```text
https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json
```

`catalog/plugins.json` is generated, checked in, and consumable by Data Designer
from the raw GitHub URL with unauthenticated HTTP. It does not require GitHub
API credentials or GitHub Pages.

The `/main/catalog/plugins.json` URL tracks the accepted `main` branch and is
therefore mutable. For an immutable snapshot, use a tag or commit SHA in the raw
URL, for example:

```text
https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/<tag-or-sha>/catalog/plugins.json
```

Release assets can be added later if raw Git snapshots are not sufficient.
External taps may use any unauthenticated raw JSON endpoint or a local catalog
file path.

### Tap governance and trust

DDPlugins is the NVIDIA-maintained curated first-party tap. New plugins belong
in this repository when they are NVIDIA-maintained, broadly useful to Data
Designer users, and have an accountable CODEOWNER who can review changes,
answer compatibility questions, and own releases.

External, team-specific, experimental, or community-maintained plugins do not
need to land in DDPlugins to be useful. Publish them from an external tap
instead: generate a schema v2 catalog and expose it from an unauthenticated raw
JSON URL or a local catalog file path. See
[docs/tap-catalog-schema-v2.md](docs/tap-catalog-schema-v2.md) for the tap
catalog contract.

Adding a tap is a trust decision, not only a discovery preference. A tap is a
pointer to Python packages. Installing from a tap runs package-manager
resolution and imports code after installation. Review the tap URL, package
name, version, source/ref, and install command before confirming installs from
non-default taps.

Data Designer CLI install flows should distinguish these cases:

- Default NVIDIA tap: this is the curated first-party source, so normal install
  confirmation rules apply.
- Non-default tap: users must explicitly opt in with `plugins taps add` before
  plugins from that tap are discoverable.
- Install from a non-default tap: show the tap URL, package name, package
  version, source URL/ref/path, and exact package-manager command; require
  confirmation unless `--yes` is passed.

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

See [docs/releasing.md](docs/releasing.md) for the full release guide.

## License

Apache-2.0. See [LICENSE](LICENSE).
