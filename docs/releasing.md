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

## Release CI

Tag pushes trigger the publish workflow. It verifies that:

- the plugin directory exists;
- the tagged commit is reachable from `main`;
- the tag version matches plugin metadata;
- the tag pusher is authorized by the plugin `CODEOWNERS`;
- plugin tests pass in an isolated virtual environment;
- the package builds successfully.

If all checks pass, CI publishes the package to PyPI using the repository
`PYPI_TOKEN` secret.

