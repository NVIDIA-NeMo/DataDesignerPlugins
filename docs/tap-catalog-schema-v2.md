# Tap Catalog Schema v2

Schema v2 is the concrete JSON contract for Data Designer plugin tap catalogs.
It extends the current read-only catalog shape with installation source metadata
and documentation URLs so consumers can implement plugin info and install flows.

This page defines the v2 contract implemented by the checked-in
`catalog/plugins.json` file.

## Default NVIDIA Tap

The default NVIDIA tap catalog URL is:

```text
https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json
```

That URL points at the generated and checked-in catalog for the accepted `main`
branch. Data Designer can read it as ordinary unauthenticated JSON over HTTP; it
does not require GitHub API authentication and does not depend on GitHub Pages.

The `/main/catalog/plugins.json` URL is mutable because it tracks the current
accepted `main` branch. Use a tag or commit SHA in the raw URL when consumers
need an immutable snapshot:

```text
https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/<tag-or-sha>/catalog/plugins.json
```

Release assets can be added later if raw tag or SHA snapshots are not enough for
a downstream distribution workflow.

External taps are not required to use GitHub. They may expose schema v2 from any
unauthenticated raw JSON endpoint, or from a local catalog file path for
authoring and offline workflows. The configured URL must return the raw JSON
catalog body, not an HTML documentation page or repository browser page.

## Tap Governance and Trust

DDPlugins is the NVIDIA-maintained curated first-party plugin tap for Data
Designer. New plugins belong in this repository when they are
NVIDIA-maintained, broadly useful to Data Designer users, and have an
accountable CODEOWNER. External, team-specific, experimental, or
community-maintained plugins do not need to land in DDPlugins to be useful;
publish them from an external schema v2 tap instead.

Adding a tap is a trust decision, not only a discovery preference. A tap is a
pointer to Python packages. Installing from a tap runs package-manager
resolution and imports code after installation. Review the tap URL, package
name, version, source/ref, and install command before confirming installs from
non-default taps.

Future Data Designer CLI install flows should use warning and confirmation
language with these semantics:

| Flow | Expected user-facing behavior |
| --- | --- |
| Default NVIDIA tap | Present it as the curated first-party source and apply normal confirmation rules. |
| `plugins taps add` for a non-default tap | Treat this as explicit opt-in to trust and discover a new tap. Show the tap name and raw catalog URL before saving it. |
| Install from a non-default tap | Show the tap URL, package name, package version, source URL/ref/path, and exact package-manager command. Require confirmation unless `--yes` is passed. |

For non-default taps, confirmation copy should make the install target concrete,
for example:

```text
Install plugin from non-default tap?
Tap: https://example.com/data-designer/plugins.json
Package: data-designer-example==0.1.0
Source: git https://github.com/example/data-designer-example.git @ v0.1.0
Command: uv pip install "data-designer-example @ git+https://github.com/example/data-designer-example.git@v0.1.0"
```

## Layering

Tap discovery and runtime entry-point discovery are separate layers.

Tap discovery finds and reads catalog documents from configured catalog files or
remote tap URLs. A tap catalog is data: consumers should read its
`schema_version`, validate the JSON shape, and inspect install metadata without
importing plugin packages.

Runtime entry-point discovery happens after a package is installed in a Python
environment. Data Designer discovers installed plugins from the
`data_designer.plugins` entry-point group. Schema v2 records the entry-point
group, key, and import target as catalog metadata, but a catalog entry does not
make a plugin available at runtime by itself.

Catalog delivery is separate from documentation delivery. Plugin entries include
`docs.url` for human-readable docs, but consumers should read machine-readable
catalog data from the configured local file path or raw tap URL, not from a
documentation page.

`package.path` is repo-local metadata only. It identifies where the package
lives in the repository that produced the catalog, for example
`plugins/data-designer-template`; it must not be treated as an install source
for remote taps. Consumers must use the `source` object to derive installation
targets.

Multi-plugin packages are represented as multiple entries in `plugins`. Each
runtime plugin gets its own entry, and those entries share the same `package`
and `source` objects.

## Document Shape

The top-level document must contain `schema_version` and `plugins`:

```json
{
  "schema_version": 2,
  "plugins": [
    {
      "name": "document-chunker",
      "plugin_type": "seed-reader",
      "description": "Read local documents as chunked seed records",
      "package": {
        "name": "data-designer-retrieval-sdg",
        "version": "0.1.0",
        "path": "plugins/data-designer-retrieval-sdg"
      },
      "entry_point": {
        "group": "data_designer.plugins",
        "name": "document-chunker",
        "value": "data_designer_retrieval_sdg.plugins:document_chunker_plugin"
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
      "source": {
        "type": "pypi",
        "package": "data-designer-retrieval-sdg"
      },
      "docs": {
        "url": "https://nvidia-nemo.github.io/DataDesignerPlugins/plugins/data-designer-retrieval-sdg/"
      }
    }
  ]
}
```

## Top-Level Fields

| Field | Type | Required | Contract |
| --- | --- | --- | --- |
| `schema_version` | integer | Yes | Literal `2`. Consumers must reject unsupported schema versions. |
| `plugins` | array | Yes | Plugin entries sorted deterministically by `package.name`, then runtime plugin `name`. |

## Plugin Entry Fields

Every item in `plugins` must contain these fields:

| Field | Type | Required | Contract |
| --- | --- | --- | --- |
| `name` | string | Yes | Non-empty runtime plugin name, equal to `Plugin.name` from the loaded entry point. Runtime plugin names must be unique within one catalog. |
| `plugin_type` | string | Yes | One of `column-generator`, `seed-reader`, or `processor`. |
| `description` | string | Yes | Package-level `[project].description` string. |
| `package.name` | string | Yes | PEP 503-compatible package name from `[project].name`. |
| `package.version` | string | Yes | PEP 440 package version from `[project].version`. |
| `package.path` | string | Yes | Repo-relative package directory, such as `plugins/data-designer-template`. This is repo-local metadata, not a remote tap install source. |
| `entry_point.group` | string | Yes | Literal `data_designer.plugins`. |
| `entry_point.name` | string | Yes | Installed entry-point key from `[project.entry-points."data_designer.plugins"]`. |
| `entry_point.value` | string | Yes | Import target from `[project.entry-points."data_designer.plugins"]`. |
| `compatibility.python.specifier` | string | Yes | Validated `[project].requires-python` specifier. |
| `compatibility.data_designer.requirement` | string | Yes | Exact direct dependency string for `data-designer`. |
| `compatibility.data_designer.specifier` | string | Yes | Parsed version specifier from the direct `data-designer` dependency. |
| `compatibility.data_designer.marker` | string or null | Yes | Environment marker from the direct `data-designer` dependency, or `null`. |
| `source` | object | Yes | Exactly one install source object from the source union. |
| `docs.url` | string | Yes | Absolute HTTP(S) URL for the plugin documentation page. |

## Source Objects

Each plugin entry must include exactly one `source` object. The source object is
the install source for the plugin package.

### PyPI

Use `pypi` for released packages on PyPI:

```json
{
  "type": "pypi",
  "package": "data-designer-example"
}
```

Required fields:

| Field | Type | Contract |
| --- | --- | --- |
| `type` | string | Literal `pypi`. |
| `package` | string | PEP 503-compatible package name to install from PyPI. |

Consumers should derive the default exact install target from `source.package`
plus `package.version`, for example `data-designer-example==0.1.0`, unless the
user explicitly requests a resolver-driven or latest-version workflow.

### Git

Use `git` for direct Git subdirectory installs:

```json
{
  "type": "git",
  "url": "https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git",
  "ref": "data-designer-example/v0.1.0",
  "subdirectory": "plugins/data-designer-example"
}
```

Required fields:

| Field | Type | Contract |
| --- | --- | --- |
| `type` | string | Literal `git`. |
| `url` | string | HTTP(S) Git repository URL. |
| `ref` | string | Git branch, tag, or commit to install. |
| `subdirectory` | string | Repository-relative Python package directory. |

Consumers should derive a PEP 508 direct reference from the package name and the
source fields:

```text
data-designer-example @ git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git@data-designer-example/v0.1.0#subdirectory=plugins/data-designer-example
```

### Path

Use `path` only for local catalog-file authoring workflows:

```json
{
  "type": "path",
  "path": "plugins/data-designer-example",
  "editable": true
}
```

Required fields:

| Field | Type | Contract |
| --- | --- | --- |
| `type` | string | Literal `path`. |
| `path` | string | Local filesystem path to the package directory. |
| `editable` | boolean | Whether consumers should install the package in editable mode. |

The default NVIDIA raw catalog must not use `path` sources.

## Contract Fixtures

Small checked-in JSON fixtures live in
`devtools/ddp/tests/fixtures/catalogs/`:

- `schema-v2-valid.json` is a valid schema v2 catalog with compatible PyPI and
  Git entries, incompatible Python and Data Designer compatibility entries,
  docs URLs, and a multi-plugin package represented by two entries sharing the
  same `package` and `source`.
- `schema-v2-invalid-source.json` is a schema v2 catalog with a malformed Git
  source object.
- `schema-v2-unsupported-version.json` uses an unsupported `schema_version`.

These fixtures are intended for Data Designer CLI and downstream consumer
contract tests. Consumers should load them as raw JSON from the repository path
or raw file content and validate their own catalog parsing, compatibility
filtering, source handling, docs URL handling, and multi-plugin package logic.
Do not import DDPlugins devtool modules, plugin packages, or runtime entry
points when using the fixtures; the fixture contract is JSON-only.

## Validation Rules

- Consumers must reject catalogs whose `schema_version` is unsupported.
- `entry_point.group` must be exactly `data_designer.plugins`.
- Runtime plugin names must be unique within one catalog.
- Multiple entries may share the same `package` and `source`; this is how
  multi-plugin packages are represented.
- Invalid PEP 440 versions, invalid specifiers, missing direct `data-designer`
  dependencies, stale installed entry points, and malformed source objects must
  fail catalog generation or schema validation.
