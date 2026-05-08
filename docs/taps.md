# Plugin Taps

Plugin taps are schema v2 JSON catalogs that let Data Designer find plugin
packages before those packages are installed. The default NVIDIA tap catalog URL
is:

```text
https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json
```

That URL must return the raw JSON catalog body. It is not a GitHub repository
browser page, a GitHub Pages documentation page, or a docs-only artifact.

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

The NVIDIA catalog is the checked-in `catalog/plugins.json` file served through
raw GitHub:

```text
https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json
```

The `/main/catalog/plugins.json` URL tracks the accepted `main` branch and is
therefore mutable. Use a tag or commit SHA in the raw URL when users need an
immutable snapshot:

```text
https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/<tag-or-sha>/catalog/plugins.json
```

External taps may use any unauthenticated raw JSON URL or a local filesystem
path whose content matches schema v2. A local path is useful for authoring and
offline checks; published taps should give users a raw JSON URL.

## Schema v2 Example

Schema v2 documents contain `schema_version` and `packages`. Each package entry
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
      "source": {
        "type": "pypi"
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
| `source` | Yes | One install source object from the schema v2 source union. |
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
[Tap catalog schema v2](tap-catalog-schema-v2.md).

## Source Objects

Each package has exactly one `source` object. Consumers use it with package
`name` and `version` to derive an install target.

| Source type | Required fields | Install meaning |
| --- | --- | --- |
| `pypi` | `type` | Install the exact package version from PyPI, for example `data-designer-example==0.1.0`. |
| `git` | `type`, `url`, `ref` | Install from a Git ref using a PEP 508 direct reference. Optional `subdirectory` supports packages below the repository root. |
| `url` | `type`, `url` | Install from a direct HTTP(S) wheel or source distribution URL. |
| `path` | `type`, `path`, `editable` | Install a local package path, usually for tap authoring or offline testing. |

The default NVIDIA raw catalog must use release-safe sources such as `pypi` or
tagged `git` refs. It must not publish `path` sources.

## Trust Model

DDPlugins is the NVIDIA-maintained curated first-party tap. New plugins belong
in this repository when they are NVIDIA-maintained, broadly useful to Data
Designer users, and have an accountable CODEOWNER who can review changes,
answer compatibility questions, and own releases.

External, team-specific, experimental, or community-maintained plugins do not
need to land in DDPlugins to be useful. Publish them from an external schema v2
tap instead.

Adding a tap is a trust decision, not only a discovery preference. A tap points
to Python packages. Installing from a tap runs package-manager resolution and
imports code after installation. Data Designer install flows should show the tap
URL, package name, package version, source URL/ref/path, and exact install
command before installing from a non-default tap.

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
| `repository-git-url` | Git clone/install URL used when `default-source = "git"`. |
| `docs-base-url` | Base URL for generated human docs links. |
| `package-prefix` | Prefix used by `ddp new` when creating package names. |
| `default-source` | Catalog source type generated for packages: `pypi`, `git`, or `path`. |
| `release-ref-template` | Template for release refs, commonly `{package}/v{version}`. |
| `default-data-designer-requirement` | Data Designer dependency written by scaffolds and surfaced in compatibility metadata. |
| `author-name` | Default author written into scaffolded packages. |

After publishing the raw JSON catalog, tell users the tap name and raw URL. Once
Data Designer tap configuration support is available, users should add that raw
catalog URL through the `plugins taps add` flow before discovering or installing
plugins from the non-default tap.

## Maintainer Workflow

Regenerate generated files whenever the inputs change:

| Input changed | Regenerate | Generated output |
| --- | --- | --- |
| Plugin docs or docs metadata | `make plugin-docs` | `docs/plugins/` and the generated plugin nav block in `zensical.toml`. |
| Plugin package metadata, entry points, compatibility dependency, or tap config | `make catalog` | `catalog/plugins.json`. |
| Per-plugin `CODEOWNERS` | `make codeowners` | `.github/CODEOWNERS`. |
| License headers | `make update-license-headers` | SPDX headers in source files. |

Validate with `make check-catalog` for catalog-only changes, `make check` for
all generated metadata, and `make docs` for the strict documentation build.
