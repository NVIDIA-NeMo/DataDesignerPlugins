# Releasing Plugins

Do not publish a plugin version unless the release has been requested and you
have permission to publish it.

## Version bump

Use the repo tooling to bump released plugin versions:

```bash
make bump PLUGIN=data-designer-my-plugin PART=patch
```

`PART` can be `patch`, `minor`, or `major`. Commit the resulting
`pyproject.toml` change:

```bash
git add plugins/data-designer-my-plugin/pyproject.toml
git commit -m "chore(data-designer-my-plugin): bump version to 0.2.0"
```

Pre-release versions such as `0.2.0a1` are not supported by `ddp bump`; edit the
plugin `pyproject.toml` manually only when a pre-release is explicitly needed.

## Release from main

After the version bump is merged to `main`, create the release tag:

```bash
make release PLUGIN=data-designer-my-plugin
```

Release tags are per plugin package and use this format:

```text
{package}/v{version}
```

For example, package `data-designer-my-plugin` version `0.2.0` releases from
tag `data-designer-my-plugin/v0.2.0`. The tag format comes from
`[tool.ddp.tap].release-ref-template`, which defaults to
`{package}/v{version}`.

The release target:

- warns if your git email is not listed in the plugin `CODEOWNERS`;
- runs that plugin's isolated tests;
- validates release metadata;
- builds the wheel and source distribution;
- creates a tag named `data-designer-my-plugin/v<version>`.

Push the tag printed by the release command:

```bash
git push origin data-designer-my-plugin/v0.2.0
```

## Tap discoverability

A released plugin becomes discoverable through a tap when the generated
`catalog/plugins.json` on the tap's published branch includes an entry for the
package version and runtime entry point. For the NVIDIA tap, that means the
checked-in raw JSON catalog at:

```text
https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json
```

must include the released package version, entry-point metadata, compatibility
metadata, docs URL, and install `source` object. Data Designer can discover the
entry from the tap catalog before installation, then install the package from
the entry's `source`, and finally discover the runtime plugin from the installed
package's `data_designer.plugins` entry point.

Tap discovery is not runtime discovery. A catalog entry makes a plugin visible
as installable metadata; it does not make the plugin available in a Python
environment until the package is installed.

## Install sources

The current NVIDIA tap uses PyPI source metadata for released packages:

```json
{
  "source": {
    "type": "pypi",
    "package": "data-designer-my-plugin"
  },
  "package": {
    "version": "0.2.0"
  }
}
```

Consumers derive the exact PyPI install target from `source.package` and
`package.version`, for example `data-designer-my-plugin==0.2.0`.

External taps can use Git source metadata. When `[tool.ddp.tap].default-source`
is `git`, catalog generation uses `repository-git-url` as `source.url`, the
release ref generated from `release-ref-template` as `source.ref`, and
`package.path` as `source.subdirectory`:

```json
{
  "source": {
    "type": "git",
    "url": "https://github.com/acme/DataDesignerPlugins.git",
    "ref": "data-designer-my-plugin/v0.2.0",
    "subdirectory": "plugins/data-designer-my-plugin"
  }
}
```

Consumers derive a PEP 508 direct reference from those fields, for example:

```text
data-designer-my-plugin @ git+https://github.com/acme/DataDesignerPlugins.git@data-designer-my-plugin/v0.2.0#subdirectory=plugins/data-designer-my-plugin
```

`path` sources are for local authoring catalogs and offline checks. They are not
release-safe catalog sources.

## Multi-plugin packages

One Python package may expose more than one Data Designer runtime plugin entry
point. Such packages release as a unit: one package version, one release tag,
one wheel/sdist, and one publish event cover all entry points in that package.
The catalog represents this as multiple `plugins` entries sharing the same
`package`, `source`, and `docs` objects.

## Release CI

Tag pushes trigger the publish workflow. It verifies that:

- the plugin directory exists;
- the tagged commit is reachable from `main`;
- the tag version matches plugin metadata;
- the tag pusher is authorized by the plugin `CODEOWNERS`;
- required PyPI metadata is present, including `description`, `license`,
  `readme`, and `authors`;
- the package declares `requires-python` and a direct versioned
  `data-designer` dependency;
- all declared `data_designer.plugins` entry points are represented in the
  schema v2 catalog for the releasing package;
- catalog package version, package path, description, compatibility metadata,
  docs URL, source metadata, and release ref match package metadata and tap
  config;
- release catalogs do not use `path` sources;
- plugin tests pass in an isolated virtual environment;
- the package builds successfully.

If all checks pass, CI publishes the package to PyPI using the repository
`PYPI_TOKEN` secret.
