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
| `make check` | Generated plugin docs, generated catalog, generated CODEOWNERS, and SPDX license headers. |
| `make docs` | Zensical builds the documentation site in strict mode. |
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
with `--strict` so documentation warnings fail CI.

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

Generated site inputs come from repository metadata and plugin docs:

```bash
make plugin-docs
make catalog
make codeowners
```

`docs/plugins/` and the plugin section of `zensical.toml` are generated from
plugin package metadata and `plugins/*/docs/`. Do not edit generated plugin site
pages directly. `catalog/plugins.json` is generated from installed local plugin
entry points, package metadata, and direct `data-designer` dependency specifiers
for compatibility checks by external tools.

## GitHub CI

Pull requests run the main CI workflow:

- lint
- isolated plugin tests
- plugin validation
- generated metadata and license header checks

Documentation changes also run the documentation workflow. On pull requests the
workflow builds the site and uploads a preview artifact. On pushes to `main`, it
builds the same site and deploys `site/` to GitHub Pages.
