# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner

from data_designer_template.config import TextTransformColumnConfig
from data_designer_template.impl import TextTransformColumnGenerator
from data_designer_template.plugin import plugin


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)


def _make_generator(config: TextTransformColumnConfig) -> TextTransformColumnGenerator:
    """Create a generator instance without requiring a ResourceProvider."""
    generator = TextTransformColumnGenerator.__new__(TextTransformColumnGenerator)
    generator._config = config
    return generator


class TestTextTransformColumnConfig:
    def test_required_columns(self) -> None:
        config = TextTransformColumnConfig(name="out", source_column="my_col")
        assert config.required_columns == ["my_col"]

    def test_side_effect_columns(self) -> None:
        config = TextTransformColumnConfig(name="out", source_column="src")
        assert config.side_effect_columns == []

    def test_column_emoji(self) -> None:
        config = TextTransformColumnConfig(name="out", source_column="src")
        assert config.get_column_emoji() == "🔄"

    def test_default_transform_is_upper(self) -> None:
        config = TextTransformColumnConfig(name="out", source_column="src")
        assert config.transform == "upper"


class TestTextTransformColumnGenerator:
    @pytest.fixture()
    def source_df(self) -> pd.DataFrame:
        return pd.DataFrame({"input_text": ["hello world", "foo bar", "Test Case"]})

    def test_upper_transform(self, source_df: pd.DataFrame) -> None:
        generator = _make_generator(
            TextTransformColumnConfig(name="output", source_column="input_text", transform="upper"),
        )
        result = generator.generate(source_df)
        assert list(result["output"]) == ["HELLO WORLD", "FOO BAR", "TEST CASE"]

    def test_lower_transform(self, source_df: pd.DataFrame) -> None:
        generator = _make_generator(
            TextTransformColumnConfig(name="output", source_column="input_text", transform="lower"),
        )
        result = generator.generate(source_df)
        assert list(result["output"]) == ["hello world", "foo bar", "test case"]

    def test_title_transform(self, source_df: pd.DataFrame) -> None:
        generator = _make_generator(
            TextTransformColumnConfig(name="output", source_column="input_text", transform="title"),
        )
        result = generator.generate(source_df)
        assert list(result["output"]) == ["Hello World", "Foo Bar", "Test Case"]


class TestTextTransformPreviewIntegration:
    """Integration tests that run the plugin through the DataDesigner preview workflow.

    These tests verify end-to-end behavior: seed data flows through the config
    builder, the engine discovers and instantiates the plugin via its entry point,
    and the generated columns appear in the preview result.
    """

    def test_preview_applies_transform(self, tmp_path: pd.DataFrame) -> None:
        """Run a full preview and verify the plugin produces the expected output."""
        seed_df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})

        builder = DataDesignerConfigBuilder()
        builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
        builder.add_column(
            name="name_upper",
            column_type="text-transform",
            source_column="name",
            transform="upper",
        )

        result = DataDesigner(artifact_path=tmp_path / "artifacts").preview(builder, num_records=3)

        assert result.dataset is not None
        assert "name_upper" in result.dataset.columns
        assert list(result.dataset["name_upper"]) == ["ALICE", "BOB", "CHARLIE"]

    def test_preview_with_multiple_transforms(self, tmp_path: pd.DataFrame) -> None:
        """Chain multiple text-transform columns off the same seed column."""
        seed_df = pd.DataFrame({"greeting": ["hello world", "good morning"]})

        builder = DataDesignerConfigBuilder()
        builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
        builder.add_column(
            name="greeting_upper",
            column_type="text-transform",
            source_column="greeting",
            transform="upper",
        )
        builder.add_column(
            name="greeting_title",
            column_type="text-transform",
            source_column="greeting",
            transform="title",
        )

        result = DataDesigner(artifact_path=tmp_path / "artifacts").preview(builder, num_records=2)

        assert result.dataset is not None
        assert list(result.dataset["greeting_upper"]) == ["HELLO WORLD", "GOOD MORNING"]
        assert list(result.dataset["greeting_title"]) == ["Hello World", "Good Morning"]

    def test_preview_preserves_seed_columns(self, tmp_path: pd.DataFrame) -> None:
        """Verify that the original seed columns are retained in the preview output."""
        seed_df = pd.DataFrame({"text": ["one", "two", "three"], "id": [1, 2, 3]})

        builder = DataDesignerConfigBuilder()
        builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
        builder.add_column(
            name="text_lower",
            column_type="text-transform",
            source_column="text",
            transform="lower",
        )

        result = DataDesigner(artifact_path=tmp_path / "artifacts").preview(builder, num_records=3)

        assert result.dataset is not None
        assert "text" in result.dataset.columns
        assert "id" in result.dataset.columns
        assert "text_lower" in result.dataset.columns
