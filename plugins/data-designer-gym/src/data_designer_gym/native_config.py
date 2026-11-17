# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route native DD structured/judge columns through a Gym model service."""

from typing import Any, Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import BaseModel, Field, field_validator, model_validator

from data_designer_gym.config import GymResourceColumnConfig


class GymNativeLLMColumnConfig(SingleColumnConfig):
    """Preserve the native DD generation contract and replace only its transport.

    The resource must be an operator-managed Gym Chat Completions model service.
    It must not override the request's source sampling/model/thinking settings.
    """

    column_type: Literal["gym-native-llm"] = "gym-native-llm"
    native_column: dict[str, Any]
    model_alias: str
    resource: GymResourceColumnConfig
    metadata_extra_body_keys: list[str] = Field(default_factory=list)

    @field_validator("native_column", mode="before")
    @classmethod
    def serialize_native_column(cls, value):
        """Accept native configs without importing DD column enums during discovery."""
        return value.model_dump(mode="json") if isinstance(value, BaseModel) else value

    @property
    def native(self):
        """Validate the original config after DD plugin discovery has completed."""
        from data_designer.config.column_configs import LLMJudgeColumnConfig, LLMStructuredColumnConfig

        kind = self.native_column.get("column_type")
        if kind not in ("llm-judge", "llm-structured"):
            raise ValueError("native column must be llm-judge or llm-structured")
        config = LLMJudgeColumnConfig if kind == "llm-judge" else LLMStructuredColumnConfig
        return config.model_validate(self.native_column)

    @model_validator(mode="after")
    def validate_native_contract(self):
        """Reject ambiguous or unsupported native/service combinations."""
        native = self.native
        if self.name != native.name or self.model_alias != native.model_alias:
            raise ValueError("native name and model alias must match the outer column")
        if self.skip != native.skip or self.propagate_skip != native.propagate_skip or self.drop != native.drop:
            raise ValueError("native skip, propagation and drop policy must match the outer column")
        if native.tool_alias or native.multi_modal_context:
            raise ValueError("native Gym transport currently supports text-only columns without tools")
        if self.resource.session or self.resource.response_fields or self.resource.response_protocol != "json":
            raise ValueError("native Gym transport requires a direct raw Chat Completions model route")
        if self.resource.request_column != "_gym_native_request":
            raise ValueError("resource request_column must be _gym_native_request")
        return self

    @property
    def required_columns(self):
        """Keep every native template dependency and the source row identity."""
        return list(dict.fromkeys([*self.native.required_columns, self.resource.row_id_column]))

    @property
    def side_effect_columns(self):
        """Keep native trace/reasoning fields and all model-call evidence."""
        return [*self.native.side_effect_columns, self.name + "__gym_calls"]
