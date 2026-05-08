# Tap Catalog Schema v2

Schema v2 is the concrete JSON contract for Data Designer plugin tap catalogs.
It is package-first: a catalog lists installable Python packages, and each
package lists the Data Designer runtime plugins exposed after installation.

This package-first shape keeps the catalog generic. The JSON catalog can be
hosted anywhere that serves raw JSON, and each package can be installed from
PyPI, Git, a direct URL, or a local path without assuming the package lives in
the same repository as the catalog.

For tap discovery workflow, the default NVIDIA raw catalog URL, mutability,
external tap setup, and trust guidance, see [Plugin taps](taps.md).

## Document Shape

The top-level document must contain `schema_version` and `packages`:

```json
{
  "schema_version": 2,
  "packages": [
    {
      "name": "data-designer-retrieval-sdg",
      "version": "0.1.0",
      "description": "Retriever SDG toolkit",
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

## Top-Level Fields

| Field | Type | Required | Contract |
| --- | --- | --- | --- |
| `schema_version` | integer | Yes | Literal `2`. Consumers must reject unsupported schema versions. |
| `packages` | array | Yes | Package entries sorted deterministically by package name. |

## Package Fields

Every item in `packages` must contain these fields:

| Field | Type | Required | Contract |
| --- | --- | --- | --- |
| `name` | string | Yes | PEP 503-compatible Python distribution package name. |
| `version` | string | Yes | PEP 440 package version. |
| `description` | string | Yes | Package-level description. |
| `source` | object | Yes | Exactly one install source object from the source union. |
| `compatibility.python.specifier` | string | Yes | Python version specifier. |
| `compatibility.data_designer.requirement` | string | Yes | Exact direct dependency string for `data-designer`. |
| `compatibility.data_designer.specifier` | string | Yes | Parsed version specifier from the direct `data-designer` dependency. |
| `compatibility.data_designer.marker` | string or null | Yes | Environment marker from the direct `data-designer` dependency, or `null`. |
| `docs.url` | string | Yes | Absolute HTTP(S) URL for package documentation. |
| `plugins` | array | Yes | Non-empty array of runtime plugin entries exposed by this package. |

## Runtime Plugin Fields

Every item in a package's `plugins` array must contain these fields:

| Field | Type | Required | Contract |
| --- | --- | --- | --- |
| `name` | string | Yes | Non-empty runtime plugin name. Runtime plugin names must be unique within one catalog. |
| `plugin_type` | string | Yes | One of `column-generator`, `seed-reader`, or `processor`. |
| `entry_point.group` | string | Yes | Literal `data_designer.plugins`. |
| `entry_point.name` | string | Yes | Installed entry-point key from `[project.entry-points."data_designer.plugins"]`. |
| `entry_point.value` | string | Yes | Import target from `[project.entry-points."data_designer.plugins"]`. |

## Source Objects

Each package must include exactly one `source` object. Consumers combine the
package `name`, package `version`, and `source` object to derive an install
target.

### PyPI

Use `pypi` for released packages whose package name is installable from PyPI:

```json
{
  "type": "pypi"
}
```

Required fields:

| Field | Type | Contract |
| --- | --- | --- |
| `type` | string | Literal `pypi`. |

Consumers should derive the default exact install target from package `name`
and `version`, for example `data-designer-example==0.1.0`, unless the user
explicitly requests a resolver-driven or latest-version workflow.

### Git

Use `git` for direct Git installs. `subdirectory` is optional so packages can
live either at the Git repository root or below it.

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

Optional fields:

| Field | Type | Contract |
| --- | --- | --- |
| `subdirectory` | string | Repository-relative Python package directory. Omit for packages at the Git repository root. |

Consumers should derive a PEP 508 direct reference from the package name and the
source fields:

```text
data-designer-example @ git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git@data-designer-example/v0.1.0#subdirectory=plugins/data-designer-example
```

For repository-root packages, omit the `#subdirectory=...` fragment.

### Direct URL

Use `url` for direct wheel or source distribution URLs:

```json
{
  "type": "url",
  "url": "https://packages.example.test/data_designer_example-0.1.0-py3-none-any.whl"
}
```

Required fields:

| Field | Type | Contract |
| --- | --- | --- |
| `type` | string | Literal `url`. |
| `url` | string | Absolute HTTP(S) wheel or source distribution URL. |

Consumers should derive a PEP 508 direct reference:

```text
data-designer-example @ https://packages.example.test/data_designer_example-0.1.0-py3-none-any.whl
```

### Path

Use `path` only for local catalog-file authoring workflows:

```json
{
  "type": "path",
  "path": "/tmp/data-designer-example",
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

- `schema-v2-valid.json` is a valid schema v2 catalog with compatible PyPI,
  Git, and direct URL package sources, incompatible Python and Data Designer
  compatibility entries, docs URLs, and a multi-plugin package represented as
  one package with two runtime plugin entries.
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
- A multi-plugin package is represented as one package object with multiple
  runtime plugin entries in `plugins`.
- Invalid PEP 440 versions, invalid specifiers, missing direct `data-designer`
  dependencies, stale installed entry points, and malformed source objects must
  fail catalog generation or schema validation.
