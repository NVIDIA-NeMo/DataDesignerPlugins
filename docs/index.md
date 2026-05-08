# Data Designer Plugins

This repository contains first-class NVIDIA-provided plugins for
[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner). It is the
NVIDIA-maintained curated first-party plugin catalog for Data Designer. Use these
docs when you need to create, review, validate, or release a plugin.

## What lives here

Each plugin is an independent Python package under `plugins/`. The root
workspace only provides shared development tooling and CI; plugins should not
depend on each other through local paths.

```text
DataDesignerPlugins/
|-- devtools/
|   `-- ddp/                    # Repo management CLI
|-- plugins/
|   `-- data-designer-template/ # Reference plugin implementation
`-- docs/                       # Zensical documentation source
```

## Start here

- [Plugin authoring](authoring.md) covers the scaffold flow, package layout,
  entry point contract, implementation expectations, and test shape.
- [Development workflow](workflow.md) covers local checks, generated metadata,
  documentation builds, and GitHub CI.
- [Plugin catalogs](catalogs.md) covers catalog discovery, the default NVIDIA catalog URL,
  catalog fields, install metadata, trust expectations, and external catalog
  publishing.
- [Catalog schema](catalog-schema.md) is the schema reference for catalog
  validation details and fixtures.
- [Releasing](releasing.md) covers version bumps, tags, ownership checks, and
  package publishing.
- [Plugins](plugins/index.md) lists generated plugin pages assembled from each
  plugin package's own docs and metadata.

## Repository contract

- Use the `ddp` CLI to scaffold new plugins.
- Keep plugins self-contained.
- Target Python 3.10 and newer.
- Write tests around public interfaces.
- Regenerate generated metadata when plugin docs or ownership changes.
- Register a package in `catalog/plugins.json` only when preparing its first
  release, and use targeted replacement only for intentional catalog metadata
  corrections.
- Treat the default NVIDIA catalog URL as the published catalog at
  `https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json`,
  served from GitHub Pages beside the static package index.
- Treat DDPlugins as the curated NVIDIA first-party catalog. New plugins belong here
  when they are NVIDIA-maintained, broadly useful, and owned by an accountable
  CODEOWNER.
- Publish unrelated external, team-specific, experimental, or
  community-maintained plugins from external catalogs rather than requiring them to
  land in this repository.
- Treat adding a catalog as a trust decision, not only a discovery preference.
- Treat catalog discovery and runtime entry-point discovery as separate
  layers.
- Run the Makefile targets locally before opening or updating a pull request.
