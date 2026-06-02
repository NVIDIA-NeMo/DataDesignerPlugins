# Catalog Schema

The catalog schema is the concrete JSON contract for Data Designer plugin
catalogs. It is package-first: a catalog lists installable Python packages, and
each package lists the Data Designer runtime plugins exposed after installation.

This package-first shape keeps the catalog generic. The JSON catalog can be
hosted anywhere that serves raw JSON, and each package can be installed from
any Python package index or PEP 508 direct reference without assuming the
package lives in the same repository as the catalog.

For catalog discovery workflow, the default NVIDIA catalog URL, external
catalog setup, and trust guidance, see [Plugin catalogs](catalogs.md).

## Document Shape

The top-level document must contain `schema_version` and `packages`:

```json
{
  "schema_version": 2,
  "packages": [
    {
      "name": "data-designer-retrieval-sdg",
      "description": "Retriever SDG toolkit",
      "install": {
        "requirement": "data-designer-retrieval-sdg",
        "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
      },
      "compatibility": {
        "python": {
          "specifier": ">=3.10"
        },
        "data_designer": {
          "requirement": "data-designer>=0.6.1",
          "specifier": ">=0.6.1",
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
| `description` | string | Yes | Package-level description. |
| `install.requirement` | string | Yes | PEP 508 requirement used to install the package. |
| `install.index_url` | string | No | Absolute HTTP(S) Python Simple API index URL for index-backed packages. |
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

## Install Metadata

Each package must include exactly one `install` object. Consumers use
`install.requirement` as the installer requirement. `install.index_url` is
optional and points at the Python Simple API index that should be added for
index-backed packages.

### Static Index Package

Use a package-name requirement plus an index URL for packages released through
this repository's static package index:

```json
{
  "requirement": "data-designer-example",
  "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
}
```

For packages hosted in this catalog's static index, `requirement` should usually be
the package `name` without a version specifier so the package manager can
resolve installs and upgrades from the Simple API index.

### Default Installer Index

Catalogs may omit `index_url` for packages available from the installer's
configured default package index:

```json
{
  "requirement": "data-designer-example"
}
```

### Direct Reference

Catalogs may use any valid PEP 508 direct reference, such as Git or an HTTP(S)
wheel URL:

```json
{
  "requirement": "data-designer-example @ git+https://github.com/acme/plugin.git@v1.0.0"
}
```

```json
{
  "requirement": "data-designer-example @ https://packages.example.test/data_designer_example-0.1.0-py3-none-any.whl"
}
```

## Contract Fixtures

Small checked-in JSON fixtures live in
`devtools/ddp/tests/fixtures/catalogs/`:

- `catalog-valid.json` is a valid catalog fixture with compatible
  index-backed, Git direct-reference, and HTTP direct-reference package
  installs, incompatible Python and Data Designer compatibility entries, docs
  URLs, and a multi-plugin package represented as one package with two runtime
  plugin entries.
- `catalog-invalid-install.json` is a catalog fixture with malformed
  install metadata.
- `catalog-unsupported-version.json` uses an unsupported `schema_version`.

These fixtures are intended for Data Designer CLI and downstream consumer
contract tests. Consumers should load them as raw JSON from the repository path
or raw file content and validate their own catalog parsing, compatibility
filtering, install handling, docs URL handling, and multi-plugin package logic.
Do not import DDPlugins devtool modules, plugin packages, or runtime entry
points when using the fixtures; the fixture contract is JSON-only.

## Validation Rules

- Consumers must reject catalogs whose `schema_version` is unsupported.
- `entry_point.group` must be exactly `data_designer.plugins`.
- Runtime plugin names must be unique within one catalog.
- A multi-plugin package is represented as one package object with multiple
  runtime plugin entries in `plugins`.
- Invalid PEP 440 versions, invalid specifiers, missing direct `data-designer`
  dependencies, stale installed entry points, and malformed install objects must
  fail catalog generation or schema validation.
