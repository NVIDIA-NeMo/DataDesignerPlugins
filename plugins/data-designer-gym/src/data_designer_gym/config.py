# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from data_designer.config.base import ProcessorConfig, SingleColumnConfig
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


class GymRequestConfig(BaseModel):
    """Describe one JSON request to a configured Gym service."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str | None = None
    route: str
    method: Literal["POST", "DELETE"] = "POST"
    request_column: str | None = None
    request_columns: dict[str, str] = Field(default_factory=dict)
    request_values: dict[str, JsonValue] = Field(default_factory=dict)
    response_headers: dict[str, str] = Field(default_factory=dict)
    bearer_token_env: str | None = None

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str | None) -> str | None:
        """Reject credentials, queries, and non-HTTP service URLs."""
        if value is not None:
            url = urlsplit(value)
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username is not None
                or url.password is not None
                or url.query
                or url.fragment
                or any(char.isspace() for char in value)
            ):
                raise ValueError("endpoint must be an HTTP(S) URL without credentials, query, or fragment")
            url.port
            value = value.rstrip("/")
        return value

    @field_validator("route")
    @classmethod
    def validate_route(cls, value: str) -> str:
        """Require a local absolute route."""
        if not value.startswith("/") or value.startswith("//") or any(char in value for char in "?#"):
            raise ValueError("route must be an absolute URL path without query or fragment")
        if any(char.isspace() for char in value) or any(part in (".", "..") for part in value.split("/")):
            raise ValueError("route must not contain whitespace or traversal")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> GymRequestConfig:
        """Validate request assembly and token mappings before execution."""
        if self.request_column is not None and (self.request_columns or self.request_values):
            raise ValueError("request_column cannot be combined with request_columns or request_values")
        if set(self.request_columns) & set(self.request_values):
            raise ValueError("request fields cannot have both a column and value")
        paths = [*self.request_columns.values(), *self.response_headers.values()]
        if self.request_column is not None:
            paths.append(self.request_column)
        if any(not GymTaskProcessorConfig._is_source_path(path) for path in paths):
            raise ValueError("column paths must contain non-empty dot-separated names")
        if any(not key.strip() for key in (*self.request_columns, *self.request_values)):
            raise ValueError("request field names must not be empty")
        for header in self.response_headers:
            if not header or not header.isascii() or any(not (char.isalnum() or char in "-_") for char in header):
                raise ValueError("response header names must be HTTP header tokens")
            if header.lower() in ("authorization", "cookie", "host", "content-length"):
                raise ValueError("response_headers cannot override transport or authentication headers")
        if self.bearer_token_env is not None and not self.bearer_token_env.isidentifier():
            raise ValueError("bearer_token_env must be an environment variable name")
        return self

    @property
    def source_paths(self) -> list[str]:
        return [self.request_column] if self.request_column is not None else list(self.request_columns.values())


class GymSessionConfig(BaseModel):
    """Configure row-local initialization and cleanup for trusted Gym services."""

    model_config = ConfigDict(extra="forbid")

    initialize: list[GymRequestConfig] = Field(default_factory=list)
    cleanup: list[GymRequestConfig] = Field(default_factory=list)
    cleanup_timeout: float = Field(default=10, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> GymSessionConfig:
        """Require cleanup when initialization allocates a session."""
        if bool(self.initialize) != bool(self.cleanup):
            raise ValueError("session initialize and cleanup must both be configured")
        if any(request.response_headers for request in self.cleanup):
            raise ValueError("response_headers are supported only for initialization")
        return self


class GymColumnConfig(GymRequestConfig, SingleColumnConfig):
    """Shared request, identity, and execution limits for Gym columns."""

    endpoint: str
    row_id_column: str
    provenance_columns: list[str] = Field(default_factory=list)
    timeout: float = Field(default=60, gt=0, allow_inf_nan=False)
    queue_timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_parallel_requests: int = Field(default=4, ge=1)
    session: GymSessionConfig | None = None
    response_fields: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_column(self) -> GymColumnConfig:
        """Validate dependencies and reject output/input collisions."""
        if not self.name.strip():
            raise ValueError("name must not be empty")
        paths = [self.row_id_column, *self.provenance_columns]
        if any(not GymTaskProcessorConfig._is_source_path(path) for path in paths):
            raise ValueError("column paths must contain non-empty dot-separated names")
        if self.response_headers:
            raise ValueError("response_headers are supported only for session initialization")
        if self.response_fields is not None:
            if any(not key.strip() or key.strip() != key for key in self.response_fields):
                raise ValueError("response field names must not be empty or contain surrounding whitespace")
            if any(not GymTaskProcessorConfig._is_source_path(path) for path in self.response_fields.values()):
                raise ValueError("response field paths must contain non-empty dot-separated names")
        if set(self.required_columns) & {self.name, *self.side_effect_columns}:
            raise ValueError("Gym outputs must not overwrite required columns")
        return self

    @property
    def required_columns(self) -> list[str]:
        paths = [*self.source_paths, self.row_id_column, *self.provenance_columns]
        if self.session is not None:
            for request in (*self.session.initialize, *self.session.cleanup):
                paths.extend(request.source_paths)
        return list(dict.fromkeys(path.split(".")[0] for path in paths))

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class GymResourceColumnConfig(GymColumnConfig):
    """Call a resource endpoint and expose its structured execution evidence."""

    column_type: Literal["gym-resource"] = "gym-resource"
    response_protocol: Literal["json", "gym-episode"] = "json"


class GymHarnessColumnConfig(GymColumnConfig):
    """Ask a harness for a final assistant answer and retain native evidence."""

    column_type: Literal["gym-harness"] = "gym-harness"
    route: str = "/v1/responses"
    evidence_column: str

    @field_validator("evidence_column")
    @classmethod
    def validate_evidence_column(cls, value: str) -> str:
        """Require a distinct, plain column name."""
        if not value.strip() or value.strip() != value or "." in value:
            raise ValueError("evidence_column must be a non-empty column name without dots")
        return value

    @model_validator(mode="after")
    def validate_harness(self) -> GymHarnessColumnConfig:
        """Reject invalid static harness requests before execution."""
        if self.evidence_column == self.name:
            raise ValueError("evidence_column must differ from name")
        if self.method != "POST":
            raise ValueError("harness requests must use POST")
        if self.request_column is None:
            if "input" not in self.request_columns and "input" not in self.request_values:
                raise ValueError("harness requests require input")
            if self.request_values.get("stream") or self.request_values.get("background"):
                raise ValueError("harness requests must be foreground and non-streaming")
        return self

    @property
    def side_effect_columns(self) -> list[str]:
        return [self.evidence_column]
