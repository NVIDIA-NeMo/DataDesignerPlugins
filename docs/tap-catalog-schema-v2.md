# Tap Catalog Schema v2

Schema v2 is the concrete JSON contract for Data Designer plugin tap catalogs.
It extends the current read-only catalog shape with installation source metadata
and documentation URLs so consumers can implement plugin info and install flows.

This page defines the v2 contract implemented by the checked-in
`catalog/plugins.json` file.

For tap discovery workflow, the default NVIDIA raw catalog URL, mutability,
external tap setup, and trust guidance, see [Plugin taps](taps.md).

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
