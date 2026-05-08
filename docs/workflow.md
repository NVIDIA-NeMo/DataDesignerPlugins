# Development Workflow

Use the Makefile targets as the local interface for this repository. They match
the checks used in GitHub CI and keep plugin packages isolated from each other.

## Local setup

```bash
make sync
```

This syncs the `uv` workspace, including the shared `ddp` development tooling
and the documentation build tool.

## Local checks

Run the full local pipeline before opening a pull request:

```bash
make all
```

The target runs:

| Target | What it verifies |
| --- | --- |
| `make lint` | Ruff linting and formatting. |
| `make test` | Each plugin's tests in an isolated virtual environment. |
| `make validate` | Installed `data_designer.plugins` entry points with `assert_valid_plugin`. |
| `make check` | Generated plugin docs, generated catalog, package-index metadata, generated CODEOWNERS, and SPDX license headers. |
| `make docs` | Zensical builds the documentation site in strict mode and adds catalog/package-index files to `site/`. |
| `make docs-server` | Zensical serves the documentation site locally while you edit. |

## Documentation

Repository documentation source lives under `docs/`, and plugin-specific site
pages live under each plugin's `docs/` directory. The top-level docs build
regenerates `docs/plugins/` before running [Zensical](https://zensical.org/):

```bash
make docs
```

The build writes static output to `site/`, which is ignored by git. Zensical
validates internal links during the build, and this repository runs the build
with `--strict` so documentation warnings fail CI. The docs target also runs
`make package-index` so the deployable site contains `catalog/plugins.json`,
`packages.json`, `simple/`, and `pypi/`.

To preview documentation while editing:

```bash
make docs-server
```

The server listens at `http://localhost:8000` by default. Override the address
when needed:

```bash
make docs-server DOCS_DEV_ADDR=localhost:8080
```

## Generated files

Generated site inputs and catalog metadata come from repository metadata, plugin
docs, package metadata, installed entry points, catalog config, and ownership files:

```bash
make plugin-docs
make catalog
make package-index
make codeowners
```

`docs/plugins/` and the plugin section of `zensical.toml` are generated from
plugin package metadata and `plugins/*/docs/`. Do not edit generated plugin site
pages directly.

`catalog/plugins.json` is the generated machine-readable catalog artifact. It is
generated from installed local plugin entry points, package metadata,
`[tool.ddp.catalog]`, docs URL configuration, and direct `data-designer` dependency
specifiers for compatibility checks by Data Designer and external tools.

`catalog/plugins.json` is also checked in as the machine-readable NVIDIA
catalog. The default catalog URL is the GitHub Pages catalog URL:

```text
https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json
```

This URL is deployed with the documentation site and the static package index
at `https://nvidia-nemo.github.io/DataDesignerPlugins/simple/`.
External catalogs may use any unauthenticated raw JSON endpoint or a local catalog
file path, and may serve packages from any Python package index or direct
reference.

Consumers should fetch the configured JSON URL or local file path, not a GitHub
HTML file browser view.

Use focused regeneration and validation targets when only one generated surface
changed:

| Change | Regenerate | Validate |
| --- | --- | --- |
| Plugin docs or docs metadata | `make plugin-docs` | `make check-plugin-docs` or `make check` |
| Package metadata, entry points, compatibility dependency, or `[tool.ddp.catalog]` | `make catalog` | `make check-catalog` or `make check` |
| Package-list metadata or site package-index wiring | `make package-index` | `make check-package-index` or `make check` |
| Per-plugin ownership | `make codeowners` | `make check-codeowners` or `make check` |
| SPDX headers | `make update-license-headers` | `make check-license-headers` or `make check` |

## GitHub CI

Pull requests run the main CI workflow:

- lint
- isolated plugin tests
- plugin validation
- generated metadata and license header checks, including `make check-catalog`
  for checked-in catalog freshness and catalog shape, plus
  `make check-package-index` for package-list and static index generation

Documentation changes also run the documentation workflow. On pull requests the
workflow builds the site and uploads a preview artifact. On pushes to `main`, or
when a package release dispatches the workflow after updating package metadata,
it builds the same site and deploys `site/` to GitHub Pages. Every Pages deploy
includes documentation, `catalog/plugins.json`, and the static package index so
docs-only deploys do not remove installer-facing files.
