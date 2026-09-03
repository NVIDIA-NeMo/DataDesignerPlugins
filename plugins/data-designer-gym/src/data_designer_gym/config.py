# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.base import ProcessorConfig
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class GymMessageMapping(BaseModel):
    """Map one Data Designer column to a Gym input message."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1)
    content_column: str = Field(min_length=1)

    @field_validator("role", "content_column")
    @classmethod
    def validate_value(cls, value: str) -> str:
        """Reject empty mapping values."""
        if not value.strip():
            raise ValueError("message mapping values must not be empty")
        return value


class GymTaskProcessorConfig(ProcessorConfig):
    """Configuration for exporting Data Designer rows as Gym tasks."""

    processor_type: Literal["gym"] = "gym"
    task_column: str | None = None
    messages: list[GymMessageMapping] = Field(default_factory=list)
    tools_column: str | None = None
    tool_fields: list[str] | None = None
    response_params: dict[str, JsonValue] = Field(default_factory=dict)
    task_columns: dict[str, str] = Field(default_factory=dict)
    task_values: dict[str, JsonValue] = Field(default_factory=dict)
    scenario_id_column: str | None = None
    include: str | None = None
    provenance_columns: list[str] = Field(default_factory=list)
    scenario_namespace: str = Field(default="gym", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

    @model_validator(mode="after")
    def validate_mapping(self) -> GymTaskProcessorConfig:
        """Validate task assembly at the configuration boundary."""
        if (self.task_column is None) == (not self.messages):
            raise ValueError("configure exactly one of task_column or messages")

        declarative_values = [
            self.tools_column,
            self.tool_fields,
            self.response_params,
            self.task_columns,
            self.task_values,
        ]
        if self.task_column is not None and any(declarative_values):
            raise ValueError("declarative task fields cannot be used with task_column")
        if self.tool_fields is not None and self.tools_column is None:
            raise ValueError("tool_fields requires tools_column")
        if "input" in self.response_params or "tools" in self.response_params:
            raise ValueError("response_params cannot override input or tools")

        source_paths = [*self.provenance_columns, *self.task_columns.values()]
        source_paths.extend(message.content_column for message in self.messages)
        source_paths.extend(
            value
            for value in (self.task_column, self.tools_column, self.scenario_id_column, self.include)
            if value is not None
        )
        if any(not self._is_source_path(value) for value in source_paths):
            raise ValueError("column paths must contain non-empty dot-separated names")

        reserved = {"_dd_provenance", "_dd_scenario_id", "_ng_rollout_index", "_ng_task_index"}
        task_fields = set(self.task_columns) | set(self.task_values)
        if any(not field.strip() for field in task_fields):
            raise ValueError("task field names must not be empty")
        if overlap := task_fields.intersection(reserved | {"responses_create_params"}):
            raise ValueError(f"task fields are plugin-managed: {', '.join(sorted(overlap))}")
        if overlap := set(self.task_columns).intersection(self.task_values):
            raise ValueError(f"task fields cannot have both a column and value: {', '.join(sorted(overlap))}")
        if len(set(self.provenance_columns)) != len(self.provenance_columns):
            raise ValueError("provenance_columns must not contain duplicates")
        if self.tool_fields is not None:
            if any(not field.strip() for field in self.tool_fields):
                raise ValueError("tool_fields must not contain empty names")
            if len(set(self.tool_fields)) != len(self.tool_fields):
                raise ValueError("tool_fields must not contain duplicates")
        return self

    @staticmethod
    def _is_source_path(value: str) -> bool:
        return all(part and part.strip() == part for part in value.split("."))
