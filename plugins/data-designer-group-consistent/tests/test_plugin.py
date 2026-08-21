# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner
from pydantic import ValidationError

from data_designer_group_consistent.config import GroupConsistentColumnConfig
from data_designer_group_consistent.impl import GroupConsistentColumnGenerator
from data_designer_group_consistent.plugin import plugin

PERSONAS = [
    {"first_name": "Amina", "last_name": "Diallo", "email": "amina@example.test"},
    {"first_name": "Carlos", "last_name": "Silva", "email": "carlos@example.test"},
    {"first_name": "Mei", "last_name": "Chen", "email": "mei@example.test"},
]


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)


def make_config(**overrides: object) -> GroupConsistentColumnConfig:
    """Create a valid test configuration with optional field overrides."""
    values = {
        "name": "synthetic_first_name",
        "group_by": ["patient_id"],
        "records": PERSONAS,
        "field_mapping": {
            "synthetic_first_name": "first_name",
            "synthetic_last_name": "last_name",
            "synthetic_email": "email",
        },
        "role": "patient",
        "seed": 7,
    }
    values.update(overrides)
    return GroupConsistentColumnConfig(**values)


def make_generator(config: GroupConsistentColumnConfig | None = None) -> GroupConsistentColumnGenerator:
    """Create a generator without requiring a resource provider."""
    generator = GroupConsistentColumnGenerator.__new__(GroupConsistentColumnGenerator)
    generator._config = config or make_config()
    return generator


class TestGroupConsistentColumnConfig:
    def test_declares_dependencies_and_side_effects(self) -> None:
        config = make_config(group_by=["household_id", "patient_id"])

        assert config.required_columns == ["household_id", "patient_id"]
        assert config.side_effect_columns == ["synthetic_last_name", "synthetic_email"]

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"group_by": ["patient_id", "patient_id"]}, "group_by columns must be unique"),
            ({"field_mapping": {"synthetic_last_name": "last_name"}}, "must include the primary output"),
            ({"role": "  "}, "role must not be blank"),
            (
                {"records": [{"first_name": "Amina", "last_name": "Diallo"}]},
                "missing mapped fields",
            ),
        ],
    )
    def test_rejects_invalid_generation_contract(self, overrides: dict[str, object], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            make_config(**overrides)


class TestGroupConsistentColumnGenerator:
    def test_reuses_correlated_record_for_noncontiguous_group_rows(self) -> None:
        data = pd.DataFrame({"patient_id": ["p1", "p2", "p1", "p3", "p2"]})

        result = make_generator().generate(data)

        for _, group in result.groupby("patient_id"):
            assert group["synthetic_first_name"].nunique() == 1
            assert group["synthetic_last_name"].nunique() == 1
            assert group["synthetic_email"].nunique() == 1
        selected_personas = {(record["first_name"], record["last_name"], record["email"]) for record in PERSONAS}
        assert (
            set(
                result[["synthetic_first_name", "synthetic_last_name", "synthetic_email"]].itertuples(
                    index=False, name=None
                )
            )
            <= selected_personas
        )

    def test_is_stable_across_batches_and_row_order(self) -> None:
        generator = make_generator()

        first_batch = generator.generate(pd.DataFrame({"patient_id": ["p1", "p2"]}))
        second_batch = generator.generate(pd.DataFrame({"patient_id": ["p3", "p1"]}))

        first_value = first_batch.loc[first_batch["patient_id"] == "p1", "synthetic_email"].item()
        second_value = second_batch.loc[second_batch["patient_id"] == "p1", "synthetic_email"].item()
        assert first_value == second_value

    def test_treats_missing_group_values_as_one_stable_group(self) -> None:
        data = pd.DataFrame({"patient_id": [None, "p1", float("nan"), None]})

        result = make_generator().generate(data)

        missing_group = result[result["patient_id"].isna()]
        assert missing_group["synthetic_email"].nunique() == 1


class TestGroupConsistentPreviewIntegration:
    def test_preview_generates_group_consistent_personas(self, tmp_path: Path) -> None:
        seed_df = pd.DataFrame({"patient_id": ["p1", "p2", "p1", "p3"]})
        builder = DataDesignerConfigBuilder()
        builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
        builder.add_column(
            name="synthetic_first_name",
            column_type="group-consistent",
            group_by=["patient_id"],
            records=PERSONAS,
            field_mapping={
                "synthetic_first_name": "first_name",
                "synthetic_last_name": "last_name",
                "synthetic_email": "email",
            },
            role="patient",
            seed=7,
        )

        result = DataDesigner(artifact_path=tmp_path / "artifacts").preview(builder, num_records=4)

        assert result.dataset is not None
        patient_rows = result.dataset[result.dataset["patient_id"] == "p1"]
        assert patient_rows["synthetic_first_name"].nunique() == 1
        assert patient_rows["synthetic_last_name"].nunique() == 1
        assert patient_rows["synthetic_email"].nunique() == 1
