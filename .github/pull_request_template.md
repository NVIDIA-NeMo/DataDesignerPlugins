## Summary

<!-- Brief description of what this PR changes. -->

## Linked Issue

<!-- External contributors should link an open, triaged issue. Use "Fixes #NNN" or "Closes #NNN" below. See CONTRIBUTING.md. -->

## Checklist

- [ ] Linked to the open, triaged issue this PR addresses (`Fixes #NNN` or `Closes #NNN`), or I am a repository collaborator performing routine maintenance or already-planned work
- [ ] Follows the template structure (`config.py`, `impl.py`, `plugin.py`) if adding a plugin
- [ ] `assert_valid_plugin(plugin)` passes
- [ ] Unit tests included and passing (`make test-plugin PLUGIN=<name>`)
- [ ] Plugin installs standalone (`uv pip install -e plugins/<plugin-dir>`)
- [ ] Plugin docs regenerated if plugin docs, list, or metadata changed (`make plugin-docs`)
- [ ] Documentation builds if docs changed (`make docs`)
- [ ] `catalog/plugins.json` updated only if preparing a first release (`make catalog PLUGIN=<name>`)
- [ ] `.github/CODEOWNERS` regenerated if ownership changed (`make codeowners`)
- [ ] Per-plugin `CODEOWNERS` file included (auto-created by `ddp new`)
- [ ] NVIDIA SPDX headers on all files (`make check-license-headers`)
