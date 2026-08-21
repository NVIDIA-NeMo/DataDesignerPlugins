# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import Field, JsonValue, model_validator
from typing_extensions import Self


class GroupConsistentColumnConfig(SingleColumnConfig):
    """Selects one correlated record deterministically for each logical group."""

    column_type: Literal["group-consistent"] = "group-consistent"
    group_by: list[str] = Field(min_length=1)
    records: list[dict[str, JsonValue]] = Field(min_length=1)
    field_mapping: dict[str, str] = Field(min_length=1)
    role: str = Field(default="default", min_length=1)
    seed: int = 0

    @model_validator(mode="after")
    def validate_generation_contract(self) -> Self:
        """Validate group keys, output columns, and candidate record fields.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the configuration cannot generate every declared output.
        """
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("group_by columns must be unique")
        if self.name not in self.field_mapping:
            raise ValueError(f"field_mapping must include the primary output column {self.name!r}")
        if not self.role.strip():
            raise ValueError("role must not be blank")

        required_fields = set(self.field_mapping.values())
        for index, record in enumerate(self.records):
            missing_fields = required_fields - record.keys()
            if missing_fields:
                raise ValueError(f"records[{index}] is missing mapped fields: {sorted(missing_fields)}")
        return self

    @staticmethod
    def get_column_emoji() -> str:
        return "🔗"

    @property
    def required_columns(self) -> list[str]:
        return self.group_by

    @property
    def side_effect_columns(self) -> list[str]:
        return [column for column in self.field_mapping if column != self.name]
