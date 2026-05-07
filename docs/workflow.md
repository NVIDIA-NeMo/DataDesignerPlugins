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

Generated site inputs and tap metadata come from repository metadata, plugin
docs, package metadata, installed entry points, tap config, and ownership files:

```bash
make plugin-docs
make catalog
make codeowners
```

`docs/plugins/` and the plugin section of `zensical.toml` are generated from
plugin package metadata and `plugins/*/docs/`. Do not edit generated plugin site
pages directly.

`catalog/plugins.json` is the generated machine-readable tap artifact. It is
generated from installed local plugin entry points, package metadata,
`[tool.ddp.tap]`, docs URL configuration, and direct `data-designer` dependency
specifiers for compatibility checks by Data Designer and external tools.

`catalog/plugins.json` is also checked in as the machine-readable NVIDIA tap
catalog. The default tap URL is the unauthenticated raw GitHub URL:

```text
https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json
```

This URL tracks accepted `main` and is mutable. Tag or commit SHA raw URLs are
immutable snapshots, for example
`https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/<tag-or-sha>/catalog/plugins.json`.
Release assets can be added later only if a distribution workflow needs them.
External taps may use any unauthenticated raw JSON endpoint or a local catalog
file path.

Human documentation can link to the raw JSON catalog, but the documentation site
does not deliver the machine catalog. Consumers should fetch the configured raw
JSON URL or local file path, not a GitHub Pages page, Zensical page, or GitHub
HTML file browser view.

Use focused regeneration and validation targets when only one generated surface
changed:

| Change | Regenerate | Validate |
| --- | --- | --- |
| Plugin docs or docs metadata | `make plugin-docs` | `make check-plugin-docs` or `make check` |
| Package metadata, entry points, compatibility dependency, or `[tool.ddp.tap]` | `make catalog` | `make check-catalog` or `make check` |
| Per-plugin ownership | `make codeowners` | `make check-codeowners` or `make check` |
| SPDX headers | `make update-license-headers` | `make check-license-headers` or `make check` |

## GitHub CI

Pull requests run the main CI workflow:

- lint
- isolated plugin tests
- plugin validation
- generated metadata and license header checks, including `make check-catalog`
  for checked-in catalog freshness and schema v2 shape

Documentation changes also run the documentation workflow. On pull requests the
workflow builds the site and uploads a preview artifact. On pushes to `main`, it
builds the same site and deploys `site/` to GitHub Pages. GitHub Pages is only
for human-readable documentation; the machine-readable catalog is exposed from
the raw `catalog/plugins.json` URL above.
