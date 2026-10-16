# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GymTask(BaseModel):
    """One Gym-native task with arbitrary environment metadata."""

    model_config = ConfigDict(extra="allow")

    responses_create_params: dict[str, Any]
    agent_ref: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_managed_fields(cls, value: Any) -> Any:
        """Keep identity fields under plugin control."""
        if isinstance(value, dict):
            managed = {"_dd_provenance", "_dd_scenario_id", "_ng_rollout_index", "_ng_task_index"}
            present = managed.intersection(value)
            if present:
                raise ValueError(f"Gym task contains plugin-managed fields: {', '.join(sorted(present))}")
        return value


class GymScenario(BaseModel):
    """Canonical scenario bundle exported by the plugin."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    scenario_id: str = Field(min_length=1)
    task: GymTask
    provenance: dict[str, Any] = Field(default_factory=dict)
