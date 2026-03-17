# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from data_designer.engine.testing.utils import assert_valid_plugin

from data_designer_template.config import TextTransformColumnConfig
from data_designer_template.impl import TextTransformColumnGenerator
from data_designer_template.plugin import plugin


def test_valid_plugin():
    assert_valid_plugin(plugin)


class TestTextTransformColumnGenerator:
    @pytest.fixture()
    def source_df(self):
        return pd.DataFrame({"input_text": ["hello world", "foo bar", "Test Case"]})

    @pytest.fixture()
    def make_config(self):
        def _make(name: str = "output", transform: str = "upper", source_column: str = "input_text"):
            return TextTransformColumnConfig(
                name=name,
                source_column=source_column,
                transform=transform,
            )

        return _make

    def test_upper_transform(self, source_df, make_config):
        config = make_config(transform="upper")
        generator = TextTransformColumnGenerator.__new__(TextTransformColumnGenerator)
        generator._config = config
        result = generator.generate(source_df)
        assert list(result["output"]) == ["HELLO WORLD", "FOO BAR", "TEST CASE"]

    def test_lower_transform(self, source_df, make_config):
        config = make_config(transform="lower")
        generator = TextTransformColumnGenerator.__new__(TextTransformColumnGenerator)
        generator._config = config
        result = generator.generate(source_df)
        assert list(result["output"]) == ["hello world", "foo bar", "test case"]

    def test_title_transform(self, source_df, make_config):
        config = make_config(transform="title")
        generator = TextTransformColumnGenerator.__new__(TextTransformColumnGenerator)
        generator._config = config
        result = generator.generate(source_df)
        assert list(result["output"]) == ["Hello World", "Foo Bar", "Test Case"]

    def test_required_columns(self, make_config):
        config = make_config(source_column="my_col")
        assert config.required_columns == ["my_col"]

    def test_side_effect_columns(self, make_config):
        config = make_config()
        assert config.side_effect_columns == []
