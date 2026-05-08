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
`[tool.ddp.catalog].release-ref-template`, which defaults to
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

## Catalog discoverability

A released plugin becomes discoverable through a catalog when the generated
`catalog/plugins.json` on the published catalog site includes a package object
for the package and its runtime entry points. For the NVIDIA catalog, that means
the GitHub Pages catalog at:

```text
https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json
```

must include entry-point metadata, compatibility metadata, docs URL, and an
`install` object. The checked-in raw JSON catalog is the source reviewed before
deployment, but the Pages URL is the canonical installer surface because it is
deployed with the docs and static package index. Data Designer can discover the
package from the catalog before installation, install it from the package's
`install.requirement`, and finally discover the runtime plugin from the
installed package's `data_designer.plugins` entry point.

Catalog discovery is not runtime discovery. A catalog package makes plugins visible
as installable metadata; it does not make them available in a Python
environment until the package is installed.

## Install metadata

The NVIDIA catalog uses unpinned package-name requirements plus the static Simple API
package index:

```json
{
  "name": "data-designer-my-plugin",
  "install": {
    "requirement": "data-designer-my-plugin",
    "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
  }
}
```

Consumers pass `install.requirement` to their installer and add
`install.index_url` as an extra package index. For `uv`:

```bash
uv pip install \
  --default-index https://pypi.org/simple/ \
  --index https://nvidia-nemo.github.io/DataDesignerPlugins/simple/ \
  data-designer-my-plugin
```

External catalogs can use direct references when packages are hosted elsewhere:

```text
data-designer-my-plugin @ git+https://github.com/acme/DataDesignerPlugins.git@data-designer-my-plugin/v0.2.0#subdirectory=plugins/data-designer-my-plugin
```

```text
data-designer-my-plugin @ https://packages.example.test/data_designer_my_plugin-0.2.0-py3-none-any.whl
```

The generated NVIDIA catalog should use unpinned package-name requirements
pointing at the DataDesignerPlugins package index. Package versions stay in the
package artifacts and the Simple API index, and package managers resolve the
version to install.

## Multi-plugin packages

One Python package may expose more than one Data Designer runtime plugin entry
point. Such packages release as a unit: one package version, one release tag,
one wheel/sdist, and one publish event cover all entry points in that package.
The catalog represents this as one package entry with multiple runtime plugin
entries in that package's `plugins` array.

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
  catalog for the releasing package;
- catalog package description, compatibility metadata, docs URL, install
  metadata, and release ref match package metadata and catalog config;
- plugin tests pass in an isolated virtual environment;
- the package builds successfully;
- `twine check` accepts the wheel and source distribution;
- existing GitHub Release assets are not overwritten with different bytes;
- the static package index can be regenerated from `packages.json`.

If all checks pass, CI uploads the wheel and source distribution to the
`ddp-package-assets` GitHub Release, updates the mutable `packages.json`
metadata asset, rebuilds the `dumb-pypi` Simple API index, and deploys the
complete GitHub Pages site.
