# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

Difficulty = Literal["simple", "medium", "hard"]


class GeneralistAgentEnvColumnConfig(SingleColumnConfig):
    """Configuration for synthesizing Generalist agent environment tuples.

    The generator consumes a task category column and optional context columns,
    then writes one structured environment/task/verifier tuple per input row.
    """

    column_type: Literal["generalist-agent-env"] = "generalist-agent-env"

    task_category_column: str = Field(
        description="Input column containing the task category, such as 'travel itinerary planning'.",
    )
    context_columns: list[str] = Field(
        default_factory=list,
        description="Optional seed columns copied into the synthesized sandbox database context.",
    )
    difficulty: Difficulty = Field(
        default="hard",
        description="Final task difficulty to synthesize after the simple-to-hard iteration trace.",
    )
    database_size: int = Field(
        default=8,
        ge=3,
        le=30,
        description="Number of records to synthesize into the sandbox database for each row.",
    )
    required_tag: str | None = Field(
        default=None,
        description="Optional tag that every valid solution candidate must contain.",
    )
    max_cost: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum cost constraint for the final task; repaired upward if it makes the task unsat.",
    )
    min_score: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional minimum score constraint for the final task; repaired downward if it makes the task unsat.",
    )

    @staticmethod
    def get_column_emoji() -> str:
        return "🧰"

    @field_validator("task_category_column")
    @classmethod
    def validate_task_category_column(cls, value: str) -> str:
        """Validate the task category source column name.

        Args:
            value: Candidate column name.

        Returns:
            The stripped column name.

        Raises:
            ValueError: If the column name is empty.
        """
        value = value.strip()
        if not value:
            raise ValueError("task_category_column must not be empty")
        return value

    @field_validator("context_columns")
    @classmethod
    def validate_context_columns(cls, value: list[str]) -> list[str]:
        """Validate and de-duplicate context column names.

        Args:
            value: Candidate context column names.

        Returns:
            Context column names with duplicates removed while preserving order.

        Raises:
            ValueError: If any context column name is empty.
        """
        columns: list[str] = []
        for column in value:
            column = column.strip()
            if not column:
                raise ValueError("context_columns must not contain empty column names")
            if column not in columns:
                columns.append(column)
        return columns

    @field_validator("required_tag")
    @classmethod
    def validate_required_tag(cls, value: str | None) -> str | None:
        """Normalize the optional required tag.

        Args:
            value: Candidate tag value.

        Returns:
            A lower-cased tag, or ``None`` when unset.

        Raises:
            ValueError: If the tag contains only whitespace.
        """
        if value is None:
            return None
        value = value.strip().lower()
        if not value:
            raise ValueError("required_tag must not be empty when provided")
        return value

    @model_validator(mode="after")
    def validate_distinct_columns(self) -> Self:
        """Validate cross-field column references.

        Returns:
            This config instance.

        Raises:
            ValueError: If the category column is repeated as context.
        """
        if self.task_category_column in self.context_columns:
            raise ValueError("context_columns must not repeat task_category_column")
        return self

    @property
    def required_columns(self) -> list[str]:
        return [self.task_category_column, *self.context_columns]

    @property
    def side_effect_columns(self) -> list[str]:
        return []
