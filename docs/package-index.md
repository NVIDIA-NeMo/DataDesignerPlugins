# Static Package Index Spec

This spec defines how Data Designer plugin packages are distributed without
checking built wheels or source distributions into git.

The design uses standard Python package installation wherever possible:

- GitHub Release assets store built package files.
- `dumb-pypi` generates a static PyPI-compatible Simple API index.
- GitHub Pages serves the Simple API index and Data Designer catalog JSON.
- The Data Designer catalog carries plugin discovery metadata and a standard
  install requirement, not package files.

## Goals

- Keep wheel and source distribution files out of git.
- Make package installation work with standard installers such as `uv` and
  `pip`.
- Keep the Data Designer catalog focused on plugin discovery.
- Support monorepo packages that release independently.
- Preserve a fully static public surface that does not require a package index
  server process.
- Make release CI reproducible and fail safely when an artifact name already
  exists with different content.

## Non-Goals

- Do not replace PyPI for packages that are published there.
- Do not make `dumb-pypi` the Data Designer plugin catalog.
- Do not require all plugin packages to live in this repository.
- Do not check generated wheels, source distributions, or package index files
  into git.

## Public Surfaces

The package distribution surface is split into three locations.

| Surface | URL | Owner |
| --- | --- | --- |
| Data Designer catalog | `https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json` | GitHub Pages |
| Simple package index | `https://nvidia-nemo.github.io/DataDesignerPlugins/simple/` | GitHub Pages |
| Package files | `https://github.com/NVIDIA-NeMo/DataDesignerPlugins/releases/download/ddp-package-assets/<filename>` | GitHub Release assets |

The raw checked-in catalog at
`https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json`
may remain available for review and automation, but the canonical installer
surface should be the GitHub Pages URL because the catalog, docs, and Simple
index are deployed together.

## GitHub Release Asset Bucket

Built wheels and source distributions are uploaded to a dedicated GitHub
Release:

```text
tag: ddp-package-assets
release title: Data Designer plugin package assets
latest: false
```

Package files use their standard distribution filenames:

```text
data_designer_template-0.1.0-py3-none-any.whl
data_designer_template-0.1.0.tar.gz
```

The resulting download URLs are:

```text
https://github.com/NVIDIA-NeMo/DataDesignerPlugins/releases/download/ddp-package-assets/data_designer_template-0.1.0-py3-none-any.whl
https://github.com/NVIDIA-NeMo/DataDesignerPlugins/releases/download/ddp-package-assets/data_designer_template-0.1.0.tar.gz
```

Release CI must never overwrite a wheel or source distribution with different
bytes:

- If the asset name does not exist, upload it.
- If the asset name exists and the SHA256 matches, treat the upload as already
  complete.
- If the asset name exists and the SHA256 differs, fail the release.

The asset bucket release may contain one mutable metadata asset named
`packages.json`. That file is the `dumb-pypi` JSON-lines package list used to
rebuild the static index. CI may replace `packages.json` after adding new
package rows. CI must not replace package files.

GitHub Release assets have product limits. At the time this spec was written,
GitHub documents up to 1000 assets per release, each file under 2 GiB, and no
total release size or bandwidth quota for release assets. If this repository
approaches the per-release asset count limit, move package files to GitHub
Packages, S3, or another object store, or shard the asset bucket and replace
`dumb-pypi` with tooling that supports per-file base URLs.

## Data Designer Catalog Shape

The catalog is still schema v2 and package-first. The package install metadata
should use an `install` object:

```json
{
  "schema_version": 2,
  "packages": [
    {
      "name": "data-designer-template",
      "version": "0.1.0",
      "description": "Template Data Designer plugin",
      "install": {
        "requirement": "data-designer-template==0.1.0",
        "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
      },
      "compatibility": {
        "python": {
          "specifier": ">=3.10"
        },
        "data_designer": {
          "requirement": "data-designer>=0.5.7",
          "specifier": ">=0.5.7",
          "marker": null
        }
      },
      "docs": {
        "url": "https://nvidia-nemo.github.io/DataDesignerPlugins/plugins/data-designer-template/"
      },
      "plugins": [
        {
          "name": "text-transform",
          "plugin_type": "column-generator",
          "entry_point": {
            "group": "data_designer.plugins",
            "name": "text-transform",
            "value": "data_designer_template.plugin:plugin"
          }
        }
      ]
    }
  ]
}
```

Catalog validation rules:

- `install.requirement` must parse as a PEP 508 requirement.
- For packages hosted in the static Simple index, `install.requirement` should
  be an exact `name==version` requirement.
- `install.index_url` is required for packages hosted in this repository's
  static Simple index.
- Direct references may omit `install.index_url`.
- Runtime plugin names must remain unique across the catalog.
- The catalog must not include artifact bytes or GitHub release asset IDs.

Direct reference examples:

```text
data-designer-example @ git+https://github.com/acme/plugin.git@v1.0.0
data-designer-example @ https://packages.example.test/data_designer_example-0.1.0-py3-none-any.whl
```

## Installer Behavior

For an index-backed package, an NDD CLI using `uv` should install with public
PyPI plus the plugin index:

```bash
uv pip install \
  --default-index https://pypi.org/simple/ \
  --index https://nvidia-nemo.github.io/DataDesignerPlugins/simple/ \
  data-designer-template==0.1.0
```

For a direct requirement, the CLI can install the requirement directly:

```bash
uv pip install \
  "data-designer-example @ https://packages.example.test/data_designer_example-0.1.0-py3-none-any.whl"
```

After installation, Data Designer discovers runtime plugins through the
installed package's `data_designer.plugins` entry points. The tap catalog does
not make a plugin available by itself.

## `dumb-pypi` Index Generation

`dumb-pypi` takes a package list and emits static Simple API files. It does not
need local package files if the package filenames and metadata are known.

The package list is JSON lines:

```json
{"filename":"data_designer_template-0.1.0-py3-none-any.whl","hash":"sha256=<hex>","requires_python":">=3.10"}
{"filename":"data_designer_template-0.1.0.tar.gz","hash":"sha256=<hex>","requires_python":">=3.10"}
```

Generate the package index with the GitHub Release asset download URL as the
package file base:

```bash
dumb-pypi \
  --package-list-json packages.json \
  --packages-url https://github.com/NVIDIA-NeMo/DataDesignerPlugins/releases/download/ddp-package-assets/ \
  --output-dir /tmp/ddp-package-index
```

`dumb-pypi` writes a root `index.html`, so it must not run directly against the
final docs site output. Generate into a scratch directory, then copy only the
package-index outputs needed by installers:

```text
/tmp/ddp-package-index/simple/       -> site/simple/
/tmp/ddp-package-index/pypi/         -> site/pypi/
/tmp/ddp-package-index/packages.json -> site/packages.json
```

The generated root `index.html` and `changelog/` should not replace the
documentation site unless the project explicitly chooses to expose them under a
separate subdirectory.

## Release CI Flow

For a plugin release tag such as `data-designer-template/v0.1.0`, CI should:

1. Parse the package name and version from the tag.
2. Verify the tagged commit is reachable from `main`.
3. Validate plugin metadata, CODEOWNERS, generated catalog metadata, and tests.
4. Build only the releasing plugin's wheel and source distribution.
5. Run package metadata checks such as `twine check`.
6. Compute SHA256 hashes for the built files.
7. Ensure the `ddp-package-assets` release exists and is marked non-latest.
8. Upload new wheel and sdist assets to `ddp-package-assets`.
9. Refuse to overwrite an existing package asset unless the SHA256 matches.
10. Download the current `packages.json` metadata asset from
    `ddp-package-assets`, or start with an empty package list if it does not
    exist.
11. Append or confirm JSON-lines rows for the newly built files.
12. Replace the `packages.json` metadata asset on `ddp-package-assets`.
13. Run `dumb-pypi` into a scratch directory from the updated `packages.json`.
14. Build the documentation site.
15. Copy `catalog/plugins.json` into `site/catalog/plugins.json`.
16. Copy `simple/`, `pypi/`, and `packages.json` from the scratch
    `dumb-pypi` output into the site output.
17. Deploy the complete site through GitHub Pages.

Every GitHub Pages deployment workflow must include the package index. GitHub
Pages artifact deployments replace the published site, so a docs-only deploy
that omits `simple/` would remove the package index. Use one shared
site-building script for docs deploys and release deploys.

## Main Branch Docs Flow

On pushes to `main`, the documentation workflow should also rebuild the package
index from the `packages.json` metadata asset:

1. Build docs into `site/`.
2. Download `packages.json` from the `ddp-package-assets` release if it exists.
3. Run `dumb-pypi` into a scratch directory.
4. Copy `simple/`, `pypi/`, and `packages.json` into `site/`.
5. Copy `catalog/plugins.json` into `site/catalog/plugins.json`.
6. Deploy the complete `site/` artifact to GitHub Pages.

This keeps documentation changes from deleting the Simple API index.

## Repository Tooling

Add repo tooling around the package index:

```bash
make package-index
make check-package-index
```

Expected behavior:

- `make package-index` builds a local package index from a supplied or
  downloaded `packages.json` package list.
- `make check-package-index` verifies that package-list rows are valid, hashes
  are well formed, package filenames parse, and generated Simple index files
  can be consumed by `uv`.
- The root `make all` should run package-index validation, but it should not
  require GitHub credentials or network access to GitHub Release assets.

Use local scratch fixtures for CI tests. Networked release asset upload belongs
only in release workflows.

## Local QA

The implementation must include an end-to-end scratch test that does not touch
GitHub:

1. Build local wheel and sdist files for at least `data-designer-template`.
2. Place them in `/tmp/.../packages/`.
3. Generate a `packages.json` JSON-lines file with SHA256 hashes.
4. Run `dumb-pypi` with a local package file URL:

   ```bash
   dumb-pypi \
     --package-list-json /tmp/.../packages.json \
     --packages-url file:///tmp/.../packages/ \
     --output-dir /tmp/.../index
   ```

5. Create a clean virtual environment.
6. Install through the local Simple API:

   ```bash
   uv pip install \
     --python /tmp/.../venv/bin/python \
     --default-index https://pypi.org/simple/ \
     --index file:///tmp/.../index/simple/ \
     data-designer-template==0.1.0
   ```

7. Verify the installed `data_designer.plugins` entry point is discoverable.
8. Verify the Data Designer catalog install metadata points at the same
   requirement and index URL shape used by the installer.

## Migration From Current Schema v2 Work

The current package-first catalog direction is still correct, but custom
`source` objects should be replaced with `install` metadata:

```json
"install": {
  "requirement": "data-designer-template==0.1.0",
  "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
}
```

Implementation tasks:

- Replace `source` validation with PEP 508 `install.requirement` validation.
- Add optional `install.index_url` URL validation.
- Generate exact `name==version` requirements for local packages in this repo.
- Add `dumb-pypi` as a dev/release tooling dependency.
- Add package-list generation with SHA256 hashes.
- Add package-index generation and checks.
- Update the release workflow to upload wheels and sdists to the
  `ddp-package-assets` release.
- Update the Pages workflow so every deployment includes docs, catalog JSON,
  and the Simple API package index.
- Update docs and fixtures to describe `install`, `requirement`, and
  `index_url`.

## References

- [`dumb-pypi`](https://github.com/chriskuehl/dumb-pypi)
- [PyPA Simple Repository API](https://packaging.python.org/en/latest/specifications/simple-repository-api/)
- [PyPA dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)
- [PyPA entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/)
- [GitHub Releases storage and bandwidth quotas](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas)
