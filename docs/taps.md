# Plugin Taps

Plugin taps are JSON catalogs that let Data Designer find plugin packages
before those packages are installed. The default NVIDIA tap catalog URL is:

```text
https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json
```

That URL returns the JSON catalog body from GitHub Pages. It is deployed with
the documentation site and the static Python Simple API package index.

## Discovery Layers

Tap discovery and runtime entry-point discovery are separate layers.

Tap discovery reads configured catalog files or remote tap URLs. A consumer can
validate `schema_version`, inspect compatibility metadata, show docs links, and
derive install targets without importing plugin packages.

Runtime entry-point discovery happens only after a plugin package is installed
in a Python environment. Data Designer discovers installed plugins from the
`data_designer.plugins` entry-point group. A tap package can describe those
entry points, but it does not make the plugins available at runtime by itself.

## Catalog Artifact

The NVIDIA catalog is the generated `catalog/plugins.json` file served through
GitHub Pages:

```text
https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json
```

The generated package index for catalog packages released by this tap is served
beside it:

```text
https://nvidia-nemo.github.io/DataDesignerPlugins/simple/
```

External taps may use any unauthenticated raw JSON URL or a local filesystem
path whose content matches the catalog schema. A local path is useful for
authoring and offline checks; published taps should give users a raw JSON URL
and, when needed, an accompanying Python package index.

## Catalog Example

Catalog documents contain `schema_version` and `packages`. Each package entry
describes one installable Python package and the runtime plugins it exposes
after installation.

```json
{
  "schema_version": 2,
  "packages": [
    {
      "description": "Read local documents as chunked seed records",
      "name": "data-designer-retrieval-sdg",
      "version": "0.1.0",
      "install": {
        "requirement": "data-designer-retrieval-sdg==0.1.0",
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
        "url": "https://nvidia-nemo.github.io/DataDesignerPlugins/plugins/data-designer-retrieval-sdg/"
      },
      "plugins": [
        {
          "name": "document-chunker",
          "plugin_type": "seed-reader",
          "entry_point": {
            "group": "data_designer.plugins",
            "name": "document-chunker",
            "value": "data_designer_retrieval_sdg.plugins:document_chunker_plugin"
          }
        }
      ]
    }
  ]
}
```

## Required Fields

Top-level fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | Yes | Literal `2`. Consumers must reject unsupported schema versions. |
| `packages` | Yes | Array of package entries sorted deterministically by package name. |

Package entry fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `description` | Yes | Package description from `[project].description`. |
| `name` | Yes | Python distribution package name from `[project].name`. |
| `version` | Yes | Package version from `[project].version`. |
| `install.requirement` | Yes | PEP 508 package requirement to install. |
| `install.index_url` | No | Python Simple API index URL for index-backed packages. |
| `compatibility.python.specifier` | Yes | Python requirement from `[project].requires-python`. |
| `compatibility.data_designer.requirement` | Yes | Direct `data-designer` dependency string. |
| `compatibility.data_designer.specifier` | Yes | Parsed version specifier from that dependency. |
| `compatibility.data_designer.marker` | Yes | Environment marker from that dependency, or `null`. |
| `docs.url` | Yes | Absolute HTTP(S) URL for human-readable plugin docs. |
| `plugins` | Yes | Non-empty array of runtime plugin entries exposed by this package. |

Runtime plugin entry fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | Yes | Runtime plugin name. It must be unique within one catalog. |
| `plugin_type` | Yes | One of `column-generator`, `seed-reader`, or `processor`. |
| `entry_point.group` | Yes | Literal `data_designer.plugins`. |
| `entry_point.name` | Yes | Entry-point key from `[project.entry-points."data_designer.plugins"]`. |
| `entry_point.value` | Yes | Entry-point import target from package metadata. |

For the full field contract, validation rules, and fixture descriptions, see
[Catalog schema](catalog-schema.md).

## Install Metadata

Each package has exactly one `install` object. Consumers pass
`install.requirement` to `pip`, `uv`, or another standard Python installer. If
`install.index_url` is present, consumers include it as an additional package
index.

| Shape | Required fields | Install meaning |
| --- | --- | --- |
| Static index | `requirement`, `index_url` | Install an exact package version from a Simple API index, for example `data-designer-example==0.1.0`. |
| Default installer index | `requirement` | Install an exact package version from the installer's configured default index. |
| Direct reference | `requirement` | Install from a PEP 508 direct reference such as Git or an HTTP(S) wheel URL. |

The default NVIDIA catalog uses exact package requirements plus the
DataDesignerPlugins Simple API index. External catalogs may use direct
references when packages are hosted elsewhere.

## Trust Model

DDPlugins is the NVIDIA-maintained curated first-party tap. New plugins belong
in this repository when they are NVIDIA-maintained, broadly useful to Data
Designer users, and have an accountable CODEOWNER who can review changes,
answer compatibility questions, and own releases.

External, team-specific, experimental, or community-maintained plugins do not
need to land in DDPlugins to be useful. Publish them from an external tap
instead.

Adding a tap is a trust decision, not only a discovery preference. A tap points
to Python packages. Installing from a tap runs package-manager resolution and
imports code after installation. Data Designer install flows should show the tap
URL, package name, package version, requirement, index URL or direct reference,
and exact install command before installing from a non-default tap.

## External Tap Repositories

A copied or forked tap repository should configure `[tool.ddp.tap]` in the root
`pyproject.toml`, scaffold plugins with the repo tooling, and publish the
generated raw JSON catalog.

Minimal flow:

```bash
make sync
uv run ddp new my-plugin --type column-generator
make all
```

Configure these tap-level fields for the fork or copied repository:

| Field | Purpose |
| --- | --- |
| `catalog-url` | Raw JSON URL users should add as the tap. |
| `repository-url` | Human-facing repository URL used in generated package metadata. |
| `repository-git-url` | Optional human/release metadata for repositories that also use Git refs. |
| `docs-base-url` | Base URL for generated human docs links. |
| `package-prefix` | Prefix used by `ddp new` when creating package names. |
| `package-index-url` | Simple API index URL written to generated `install.index_url` values. |
| `package-assets-url` | Base URL where package files named by `packages.json` are served. |
| `package-assets-release-tag` | GitHub Release tag used as the package asset bucket. |
| `release-ref-template` | Template for release refs, commonly `{package}/v{version}`. |
| `default-data-designer-requirement` | Data Designer dependency written by scaffolds and surfaced in compatibility metadata. |
| `author-name` | Default author written into scaffolded packages. |

After publishing the JSON catalog and package index, tell users the tap name
and catalog URL. Once Data Designer tap configuration support is available,
users should add that catalog URL through the `plugins taps add` flow before
discovering or installing plugins from the non-default tap.

## Maintainer Workflow

Regenerate generated files whenever the inputs change:

| Input changed | Regenerate | Generated output |
| --- | --- | --- |
| Plugin docs or docs metadata | `make plugin-docs` | `docs/plugins/` and the generated plugin nav block in `zensical.toml`. |
| Plugin package metadata, entry points, compatibility dependency, or tap config | `make catalog` | `catalog/plugins.json`. |
| Package-list metadata or catalog deployment | `make package-index` | `site/simple/`, `site/pypi/`, `site/packages.json`, and `site/catalog/plugins.json`. |
| Per-plugin `CODEOWNERS` | `make codeowners` | `.github/CODEOWNERS`. |
| License headers | `make update-license-headers` | SPDX headers in source files. |

Validate with `make check-catalog` for catalog-only changes,
`make check-package-index` for package-index metadata, `make check` for all
generated metadata, and `make docs` for the strict documentation build.
