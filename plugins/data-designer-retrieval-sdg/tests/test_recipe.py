# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import typer
from click.testing import CliRunner
from data_designer.config.config_builder import DataDesignerConfigBuilder

from data_designer_retrieval_sdg.recipe import build_typer_app, load_config_builder
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


def test_load_config_builder_builds_retrieval_sdg_pipeline(tmp_path: Path) -> None:
    """The recipe entry point builds the retrieval SDG workflow from argv."""
    params = SimpleNamespace(
        argv=(
            "--input-dir",
            str(tmp_path),
            "--num-pairs",
            "2",
            "--start-index",
            "1",
            "--end-index",
            "4",
            "--file-extensions",
            ".txt",
        )
    )

    builder = load_config_builder(params)

    assert isinstance(builder, DataDesignerConfigBuilder)
    seed_config = builder.get_seed_config()
    assert seed_config is not None
    assert isinstance(seed_config.source, DocumentChunkerSeedSource)
    assert seed_config.source.path == str(tmp_path)
    assert seed_config.source.file_extensions == [".txt"]
    assert seed_config.selection_strategy is not None
    assert seed_config.selection_strategy.start == 1
    assert seed_config.selection_strategy.end == 4
    assert [column.name for column in builder.get_column_configs()] == [
        "document_artifacts",
        "qa_generation",
        "deduplicated_qa_pairs",
        "qa_evaluations",
    ]


def test_build_typer_app_exposes_recipe_help() -> None:
    """The recipe exposes Typer metadata for Data Designer inspection."""
    command = typer.main.get_command(build_typer_app())
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0
    assert "Build the retrieval SDG Data Designer workflow." in result.output
    assert "--input-dir" in result.output
    assert "--num-pairs" in result.output
