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
| `make check` | Generated catalog, generated CODEOWNERS, and SPDX license headers. |
| `make docs` | Zensical builds the documentation site in strict mode. |

## Documentation

Documentation source lives under `docs/` and is built by
[Zensical](https://zensical.org/):

```bash
make docs
```

The build writes static output to `site/`, which is ignored by git. Zensical
validates internal links during the build, and this repository runs the build
with `--strict` so documentation warnings fail CI.

## Generated files

Two files are generated from repository metadata:

```bash
make catalog
make codeowners
```

`docs/catalog.md` is part of the documentation site, but it is generated from
plugin package metadata. Do not edit it manually.

## GitHub CI

Pull requests run the main CI workflow:

- lint
- isolated plugin tests
- plugin validation
- generated metadata and license header checks

Documentation changes also run the documentation workflow. On pull requests the
workflow builds the site and uploads a preview artifact. On pushes to `main`, it
builds the same site and deploys `site/` to GitHub Pages.

