# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``ddp new`` scaffold generator."""

from __future__ import annotations

import importlib
import json
import sys
import textwrap
from pathlib import Path

import pytest
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.plugins.plugin import PluginType

from ddp import catalog, scaffold

EXPECTED_FILE_TREE = {
    "CODEOWNERS",
    "README.md",
    "docs/index.md",
    "pyproject.toml",
    "src/acme_dd_sample_plugin/__init__.py",
    "src/acme_dd_sample_plugin/config.py",
    "src/acme_dd_sample_plugin/impl.py",
    "src/acme_dd_sample_plugin/plugin.py",
    "tests/test_plugin.py",
}


def write_external_catalog_repo(root: Path) -> None:
    """Write a minimal external-style catalog repository skeleton.

    Args:
        root: Temporary repository root.
    """
    (root / "plugins").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "external-catalog-workspace"

            [tool.ddp.catalog]
            catalog-url = "https://catalog.example.test/plugins.json"
            repository-url = "https://git.example.test/acme/dd-plugins"
            repository-git-url = "https://git.example.test/acme/dd-plugins.git"
            docs-base-url = "https://docs.example.test/ddp/"
            package-prefix = "acme-dd-"
            package-index-url = "https://docs.example.test/ddp/simple/"
            package-assets-url = "https://git.example.test/acme/dd-plugins/releases/download/ddp-package-assets/"
            package-assets-release-tag = "ddp-package-assets"
            release-ref-template = "release/{package}/{version}"
            default-data-designer-requirement = "data-designer>=9.9"
            author-name = "ACME Labs"
            """
        ).lstrip(),
        encoding="utf-8",
    )


def scaffold_sample_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str] | None = None,
) -> Path:
    """Scaffold the sample plugin in an external catalog fixture."""
    write_external_catalog_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scaffold, "_discover_owner", lambda: "@acme/platform")

    scaffold.main(["sample-plugin", *(args or [])])
    return tmp_path / "plugins" / "acme-dd-sample-plugin"


def generated_file_tree(plugin_dir: Path) -> set[str]:
    """Return generated files relative to the plugin root."""
    return {path.relative_to(plugin_dir).as_posix() for path in plugin_dir.rglob("*") if path.is_file()}


def read_generated_text(plugin_dir: Path) -> dict[str, str]:
    """Read important generated files keyed by relative path."""
    return {
        "pyproject": (plugin_dir / "pyproject.toml").read_text(encoding="utf-8"),
        "readme": (plugin_dir / "README.md").read_text(encoding="utf-8"),
        "docs_index": (plugin_dir / "docs" / "index.md").read_text(encoding="utf-8"),
        "config": (plugin_dir / "src" / "acme_dd_sample_plugin" / "config.py").read_text(encoding="utf-8"),
        "impl": (plugin_dir / "src" / "acme_dd_sample_plugin" / "impl.py").read_text(encoding="utf-8"),
        "plugin": (plugin_dir / "src" / "acme_dd_sample_plugin" / "plugin.py").read_text(encoding="utf-8"),
        "test": (plugin_dir / "tests" / "test_plugin.py").read_text(encoding="utf-8"),
    }


def clear_sample_plugin_modules() -> None:
    """Remove generated sample plugin modules from the import cache."""
    for module_name in list(sys.modules):
        if module_name == "acme_dd_sample_plugin" or module_name.startswith("acme_dd_sample_plugin."):
            del sys.modules[module_name]


@pytest.mark.parametrize(
    "plugin_type",
    ["column-generator", "seed-reader", "processor"],
)
def test_scaffold_generates_expected_file_tree_for_all_plugin_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plugin_type: str,
) -> None:
    plugin_dir = scaffold_sample_plugin(tmp_path, monkeypatch, ["--type", plugin_type])

    assert generated_file_tree(plugin_dir) == EXPECTED_FILE_TREE


@pytest.mark.parametrize(
    ("plugin_type", "expected_plugin_type"),
    [
        ("column-generator", PluginType.COLUMN_GENERATOR),
        ("seed-reader", PluginType.SEED_READER),
        ("processor", PluginType.PROCESSOR),
    ],
)
def test_scaffolded_plugins_import_and_validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plugin_type: str,
    expected_plugin_type: PluginType,
) -> None:
    plugin_dir = scaffold_sample_plugin(tmp_path, monkeypatch, ["--type", plugin_type])
    monkeypatch.syspath_prepend(str(plugin_dir / "src"))
    clear_sample_plugin_modules()

    plugin_module = importlib.import_module("acme_dd_sample_plugin.plugin")

    assert_valid_plugin(plugin_module.plugin)
    assert plugin_module.plugin.plugin_type == expected_plugin_type
    if plugin_type == "seed-reader":
        assert issubclass(plugin_module.plugin.impl_cls, SeedReader)
    if plugin_type == "processor":
        assert issubclass(plugin_module.plugin.impl_cls, Processor)


def test_scaffold_uses_external_catalog_config_in_generated_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = scaffold_sample_plugin(tmp_path, monkeypatch)

    assert plugin_dir.is_dir()
    assert (plugin_dir / "src" / "acme_dd_sample_plugin").is_dir()

    pyproject = (plugin_dir / "pyproject.toml").read_text(encoding="utf-8")
    readme = (plugin_dir / "README.md").read_text(encoding="utf-8")
    docs_index = (plugin_dir / "docs" / "index.md").read_text(encoding="utf-8")
    test_file = (plugin_dir / "tests" / "test_plugin.py").read_text(encoding="utf-8")

    assert 'name = "acme-dd-sample-plugin"' in pyproject
    assert '"data-designer>=9.9"' in pyproject
    assert '{name = "ACME Labs"}' in pyproject
    assert 'Repository = "https://git.example.test/acme/dd-plugins"' in pyproject
    assert 'sample-plugin = "acme_dd_sample_plugin.plugin:plugin"' in pyproject
    assert "github.com/NVIDIA-NeMo/DataDesignerPlugins" not in pyproject
    assert "NVIDIA Corporation" not in pyproject

    assert "# acme-dd-sample-plugin" in readme
    assert "uv add data-designer acme-dd-sample-plugin" in readme
    assert "https://docs.example.test/ddp/authoring/" in readme
    assert "github.com/NVIDIA-NeMo/DataDesignerPlugins" not in readme

    assert "# acme-dd-sample-plugin" in docs_index
    assert "uv add data-designer acme-dd-sample-plugin" in docs_index
    assert "https://docs.example.test/ddp/authoring/" in docs_index
    assert "github.com/NVIDIA-NeMo/DataDesignerPlugins" not in docs_index

    assert "from acme_dd_sample_plugin.plugin import plugin" in test_file


def test_scaffold_does_not_register_catalog_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold_sample_plugin(tmp_path, monkeypatch)

    assert not (tmp_path / "catalog" / "plugins.json").exists()


def test_scaffolded_package_can_be_registered_for_first_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = scaffold_sample_plugin(tmp_path, monkeypatch)
    catalog_path = tmp_path / "catalog" / "plugins.json"

    catalog.register_catalog_package(plugin_dir, catalog_path=catalog_path)

    output = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert output["packages"][0]["name"] == "acme-dd-sample-plugin"
    assert output["packages"][0]["plugins"][0]["name"] == "sample-plugin"


def test_column_generator_scaffold_contents_are_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = scaffold_sample_plugin(tmp_path, monkeypatch)
    files = read_generated_text(plugin_dir)

    assert 'description = "Data Designer sample-plugin column generator plugin"' in files["pyproject"]
    assert "This package provides a Data Designer column generator plugin." in files["readme"]
    assert "Once installed, the `sample-plugin` column type is automatically discovered" in files["readme"]
    assert "## Column type" in files["docs_index"]
    assert "Use the `sample-plugin` column type" in files["docs_index"]

    assert "from data_designer.config.base import SingleColumnConfig" in files["config"]
    assert "class SamplePluginColumnConfig(SingleColumnConfig):" in files["config"]
    assert 'column_type: Literal["sample-plugin"] = "sample-plugin"' in files["config"]

    assert (
        "from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn" in files["impl"]
    )
    assert "class SamplePluginColumnGenerator(ColumnGeneratorFullColumn[SamplePluginColumnConfig]):" in files["impl"]
    assert "data[self.config.name] = None" in files["impl"]

    assert 'config_qualified_name="acme_dd_sample_plugin.config.SamplePluginColumnConfig"' in files["plugin"]
    assert 'impl_qualified_name="acme_dd_sample_plugin.impl.SamplePluginColumnGenerator"' in files["plugin"]
    assert "plugin_type=PluginType.COLUMN_GENERATOR" in files["plugin"]
    assert "assert_valid_plugin(plugin)" in files["test"]


def test_seed_reader_scaffold_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = scaffold_sample_plugin(tmp_path, monkeypatch, ["--type", "seed-reader"])
    files = read_generated_text(plugin_dir)

    assert 'description = "Data Designer sample-plugin seed reader plugin"' in files["pyproject"]
    assert "seed-reader entry point" in files["readme"]
    assert 'Configure it with `seed_type="sample-plugin"`' in files["readme"]
    assert "column type" not in files["readme"].lower()
    assert "## Plugin type" in files["docs_index"]
    assert "## Entry point" in files["docs_index"]
    assert "DuckDB-friendly manifest" in files["docs_index"]
    assert "column type" not in files["docs_index"].lower()

    assert "from data_designer.config.base import ConfigBase" in files["config"]
    assert "from data_designer.config.seed_source import FileSystemSeedSource" in files["config"]
    assert "class SamplePluginSeedSource(FileSystemSeedSource, ConfigBase):" in files["config"]
    assert 'seed_type: Literal["sample-plugin"] = "sample-plugin"' in files["config"]

    assert "from data_designer.engine.resources.seed_reader import FileSystemSeedReader" in files["impl"]
    assert "class SamplePluginSeedReader(FileSystemSeedReader[SamplePluginSeedSource]):" in files["impl"]
    assert "def build_manifest(" in files["impl"]
    assert "context: SeedReaderFileSystemContext" in files["impl"]
    assert "pd.DataFrame | list[dict[str, Any]]" in files["impl"]
    assert "DuckDB-friendly dict" in files["impl"]
    assert "return []" in files["impl"]

    assert 'config_qualified_name="acme_dd_sample_plugin.config.SamplePluginSeedSource"' in files["plugin"]
    assert 'impl_qualified_name="acme_dd_sample_plugin.impl.SamplePluginSeedReader"' in files["plugin"]
    assert "plugin_type=PluginType.SEED_READER" in files["plugin"]
    assert "assert_valid_plugin(plugin)" in files["test"]


def test_processor_scaffold_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = scaffold_sample_plugin(tmp_path, monkeypatch, ["--type", "processor"])
    files = read_generated_text(plugin_dir)

    assert 'description = "Data Designer sample-plugin processor plugin"' in files["pyproject"]
    assert "processor entry point" in files["readme"]
    assert 'Configure it with `processor_type="sample-plugin"`' in files["readme"]
    assert "column type" not in files["readme"].lower()
    assert "## Plugin type" in files["docs_index"]
    assert "## Entry point" in files["docs_index"]
    assert "process_after_batch" in files["docs_index"]
    assert "does not fully check processor implementation classes" in files["docs_index"]
    assert "column type" not in files["docs_index"].lower()

    assert "from data_designer.config.base import ProcessorConfig" in files["config"]
    assert "class SamplePluginProcessorConfig(ProcessorConfig):" in files["config"]
    assert 'processor_type: Literal["sample-plugin"] = "sample-plugin"' in files["config"]

    assert "from data_designer.engine.processing.processors.base import Processor" in files["impl"]
    assert "class SamplePluginProcessor(Processor[SamplePluginProcessorConfig]):" in files["impl"]
    assert "def process_after_batch(" in files["impl"]
    assert "current_batch_number: int | None" in files["impl"]
    assert "return data" in files["impl"]

    assert 'config_qualified_name="acme_dd_sample_plugin.config.SamplePluginProcessorConfig"' in files["plugin"]
    assert 'impl_qualified_name="acme_dd_sample_plugin.impl.SamplePluginProcessor"' in files["plugin"]
    assert "plugin_type=PluginType.PROCESSOR" in files["plugin"]
    assert "assert_valid_plugin(plugin)" in files["test"]
    assert "does not fully check processor implementation classes yet" in files["test"]
    assert "issubclass(plugin.impl_cls, Processor)" in files["test"]


def test_scaffold_missing_catalog_config_exits_with_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "plugins").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "missing-catalog"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        scaffold.main(["sample-plugin"])

    assert exc_info.value.code == 1
    assert "[tool.ddp.catalog]" in capsys.readouterr().err
