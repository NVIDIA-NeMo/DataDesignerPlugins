# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from data_designer.engine.testing.utils import assert_valid_plugin

from data_designer_list_explode.config import ListExplodeColumnConfig
from data_designer_list_explode.impl import ListExplodeColumnGenerator
from data_designer_list_explode.plugin import plugin


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)


class TestListExplodeColumnConfig:
    def test_required_columns(self) -> None:
        config = ListExplodeColumnConfig(name="tag", source_column="tags")
        assert config.required_columns == ["tags"]

    def test_side_effect_columns(self) -> None:
        config = ListExplodeColumnConfig(name="tag", source_column="tags")
        assert config.side_effect_columns == []

    def test_allow_resize_defaults_true(self) -> None:
        config = ListExplodeColumnConfig(name="tag", source_column="tags")
        assert config.allow_resize is True

    def test_drop_empty_defaults_false(self) -> None:
        config = ListExplodeColumnConfig(name="tag", source_column="tags")
        assert config.drop_empty is False

    def test_emoji(self) -> None:
        assert ListExplodeColumnConfig.get_column_emoji() == "💥"


def _make_generator(config: ListExplodeColumnConfig) -> ListExplodeColumnGenerator:
    """Create a generator without requiring a resource provider."""
    generator = ListExplodeColumnGenerator.__new__(ListExplodeColumnGenerator)
    generator._config = config
    return generator


class TestListExplodeColumnGenerator:
    @pytest.fixture()
    def basic_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [1, 2, 3],
                "tags": [["a", "b"], ["c"], ["d", "e", "f"]],
            }
        )

    def test_explode_in_place(self, basic_df: pd.DataFrame) -> None:
        config = ListExplodeColumnConfig(name="tags", source_column="tags")
        generator = _make_generator(config)

        result = generator.generate(basic_df)

        assert list(result["tags"]) == ["a", "b", "c", "d", "e", "f"]
        assert list(result["id"]) == [1, 1, 2, 3, 3, 3]

    def test_explode_to_new_column(self, basic_df: pd.DataFrame) -> None:
        config = ListExplodeColumnConfig(name="tag", source_column="tags")
        generator = _make_generator(config)

        result = generator.generate(basic_df)

        assert list(result["tag"]) == ["a", "b", "c", "d", "e", "f"]
        assert "tags" in result.columns
        assert len(result) == 6

    def test_single_element_lists(self) -> None:
        df = pd.DataFrame({"vals": [[10], [20], [30]]})
        config = ListExplodeColumnConfig(name="vals", source_column="vals")
        generator = _make_generator(config)

        result = generator.generate(df)

        assert list(result["vals"]) == [10, 20, 30]
        assert len(result) == 3

    def test_empty_list_preserved_by_default(self) -> None:
        df = pd.DataFrame({"vals": [["a"], [], ["b"]]})
        config = ListExplodeColumnConfig(name="vals", source_column="vals")
        generator = _make_generator(config)

        result = generator.generate(df)

        # pandas.explode turns [] into NaN
        assert len(result) == 3
        assert result["vals"].iloc[0] == "a"
        assert pd.isna(result["vals"].iloc[1])
        assert result["vals"].iloc[2] == "b"

    def test_drop_empty_removes_empty_lists(self) -> None:
        df = pd.DataFrame({"vals": [["a", "b"], [], ["c"]]})
        config = ListExplodeColumnConfig(name="vals", source_column="vals", drop_empty=True)
        generator = _make_generator(config)

        result = generator.generate(df)

        assert list(result["vals"]) == ["a", "b", "c"]
        assert len(result) == 3

    def test_drop_empty_removes_none_values(self) -> None:
        df = pd.DataFrame({"vals": [["a"], None, ["b"]]})
        config = ListExplodeColumnConfig(name="vals", source_column="vals", drop_empty=True)
        generator = _make_generator(config)

        result = generator.generate(df)

        assert list(result["vals"]) == ["a", "b"]
        assert len(result) == 2

    def test_numeric_lists(self) -> None:
        df = pd.DataFrame({"nums": [[1, 2], [3, 4, 5]]})
        config = ListExplodeColumnConfig(name="nums", source_column="nums")
        generator = _make_generator(config)

        result = generator.generate(df)

        assert list(result["nums"]) == [1, 2, 3, 4, 5]

    def test_nested_dicts_in_lists(self) -> None:
        df = pd.DataFrame(
            {
                "items": [
                    [{"name": "a", "val": 1}, {"name": "b", "val": 2}],
                    [{"name": "c", "val": 3}],
                ],
            }
        )
        config = ListExplodeColumnConfig(name="item", source_column="items")
        generator = _make_generator(config)

        result = generator.generate(df)

        assert len(result) == 3
        assert result["item"].iloc[0] == {"name": "a", "val": 1}
        assert result["item"].iloc[2] == {"name": "c", "val": 3}

    def test_preserves_other_columns(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["x", "y"],
                "score": [0.9, 0.8],
                "tags": [["a", "b"], ["c"]],
            }
        )
        config = ListExplodeColumnConfig(name="tag", source_column="tags")
        generator = _make_generator(config)

        result = generator.generate(df)

        assert list(result["id"]) == ["x", "x", "y"]
        assert list(result["score"]) == [0.9, 0.9, 0.8]

    def test_index_is_reset(self) -> None:
        df = pd.DataFrame({"vals": [["a", "b"], ["c"]]})
        config = ListExplodeColumnConfig(name="vals", source_column="vals")
        generator = _make_generator(config)

        result = generator.generate(df)

        assert list(result.index) == [0, 1, 2]
