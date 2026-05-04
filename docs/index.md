# Data Designer Plugins

This repository contains first-class NVIDIA-provided plugins for
[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner). Use these
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
- [Releasing](releasing.md) covers version bumps, tags, ownership checks, and
  PyPI publishing.
- [Plugin catalog](catalog.md) lists the plugins currently discovered from
  package metadata.

## Repository contract

- Use the `ddp` CLI to scaffold new plugins.
- Keep plugins self-contained.
- Target Python 3.10 and newer.
- Write tests around public interfaces.
- Regenerate generated metadata when plugin metadata or ownership changes.
- Run the Makefile targets locally before opening or updating a pull request.
