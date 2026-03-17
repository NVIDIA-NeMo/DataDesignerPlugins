## New Plugin: [plugin-name]

**Column type**: `column-type-here`
**Description**: Brief description of what this plugin does.

### Checklist

- [ ] Plugin follows the template structure (`config.py`, `impl.py`, `plugin.py`)
- [ ] `assert_valid_plugin(plugin)` passes
- [ ] Unit tests included and passing (`uv run pytest plugins/<plugin-dir>/tests/ -v`)
- [ ] Plugin installs standalone (`uv pip install -e plugins/<plugin-dir>`)
- [ ] `docs/catalog.md` regenerated (`python tools/generate_catalog.py > docs/catalog.md`)
- [ ] Per-plugin `CODEOWNERS` file included (auto-created by `scaffold-plugin`)
- [ ] NVIDIA copyright headers on all files
