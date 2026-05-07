# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

Difficulty = Literal["simple", "medium", "hard"]


def normalize_column_name(value: str, field_name: str) -> str:
    """Normalize and validate one column name.

    Args:
        value: Candidate column name.
        field_name: Name used in validation messages.

    Returns:
        The stripped column name.

    Raises:
        ValueError: If the column name is empty.
    """
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def normalize_context_columns(value: list[str]) -> list[str]:
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
        column = normalize_column_name(column, "context_columns")
        if column not in columns:
            columns.append(column)
    return columns


def normalize_required_tag(value: str | None) -> str | None:
    """Normalize an optional required tag.

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


class GeneralistAgentEnvironmentColumnConfig(SingleColumnConfig):
    """Configuration for constructing generated Generalist sandbox environments.

    The generator consumes Data Designer generated topic, constraints, database
    schema, and database records, then emits a row-local environment with
    executable tool implementations over those generated records.
    """

    column_type: Literal["generalist-agent-environment"] = "generalist-agent-environment"

    task_topic_column: str = Field(
        description="Input column containing a generated task topic, such as 'trip planning'.",
    )
    task_constraints_column: str | None = Field(
        default=None,
        description="Optional input column containing generated constraints as text, JSON, or a structured object.",
    )
    database_schema_column: str = Field(
        description="Input column containing the generated database schema for this row.",
    )
    database_records_column: str = Field(
        description="Input column containing generated database records for this row.",
    )
    context_columns: list[str] = Field(
        default_factory=list,
        description="Optional seed columns copied into environment context.",
    )

    @staticmethod
    def get_column_emoji() -> str:
        return "🧱"

    @field_validator("task_topic_column")
    @classmethod
    def validate_task_topic_column(cls, value: str) -> str:
        """Validate the task topic source column name."""
        return normalize_column_name(value, "task_topic_column")

    @field_validator("task_constraints_column")
    @classmethod
    def validate_task_constraints_column(cls, value: str | None) -> str | None:
        """Validate the optional task constraints source column name."""
        if value is None:
            return None
        return normalize_column_name(value, "task_constraints_column")

    @field_validator("database_schema_column")
    @classmethod
    def validate_database_schema_column(cls, value: str) -> str:
        """Validate the generated database schema source column name."""
        return normalize_column_name(value, "database_schema_column")

    @field_validator("database_records_column")
    @classmethod
    def validate_database_records_column(cls, value: str) -> str:
        """Validate the generated database records source column name."""
        return normalize_column_name(value, "database_records_column")

    @field_validator("context_columns")
    @classmethod
    def validate_context_columns(cls, value: list[str]) -> list[str]:
        """Validate context column names."""
        return normalize_context_columns(value)

    @model_validator(mode="after")
    def validate_distinct_columns(self) -> Self:
        """Validate cross-field column references."""
        named_columns = [
            self.task_topic_column,
            self.database_schema_column,
            self.database_records_column,
            *self.context_columns,
        ]
        if self.task_constraints_column is not None:
            named_columns.append(self.task_constraints_column)
        if len(named_columns) != len(set(named_columns)):
            raise ValueError(
                "task_topic_column, task_constraints_column, database_schema_column, "
                "database_records_column, and context_columns must be distinct"
            )
        return self

    @property
    def required_columns(self) -> list[str]:
        columns = [self.task_topic_column]
        if self.task_constraints_column is not None:
            columns.append(self.task_constraints_column)
        columns.extend([self.database_schema_column, self.database_records_column])
        columns.extend(self.context_columns)
        return columns

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class GeneralistAgentTaskColumnConfig(SingleColumnConfig):
    """Configuration for synthesizing tasks from generated environments."""

    column_type: Literal["generalist-agent-task"] = "generalist-agent-task"

    environment_column: str = Field(
        description="Column containing a generalist-agent-environment artifact.",
    )
    difficulty: Difficulty = Field(
        default="hard",
        description="Final task difficulty to synthesize after the simple-to-hard iteration trace.",
    )
    required_tag: str | None = Field(
        default=None,
        description="Optional tag that every valid solution candidate must contain.",
    )
    max_cost: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum cost constraint. Unsatisfiable values are repaired upward.",
    )
    min_score: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional minimum score constraint. Unsatisfiable values are repaired downward.",
    )

    @staticmethod
    def get_column_emoji() -> str:
        return "🧪"

    @field_validator("environment_column")
    @classmethod
    def validate_environment_column(cls, value: str) -> str:
        """Validate the environment source column name."""
        return normalize_column_name(value, "environment_column")

    @field_validator("required_tag")
    @classmethod
    def validate_required_tag(cls, value: str | None) -> str | None:
        """Normalize the optional required tag."""
        return normalize_required_tag(value)

    @property
    def required_columns(self) -> list[str]:
        return [self.environment_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []
