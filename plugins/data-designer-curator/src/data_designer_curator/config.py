# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.base import ProcessorConfig, SingleColumnConfig
from pydantic import Field, HttpUrl, field_validator, model_validator
from typing_extensions import Self

KeepPolicy = Literal["first", "last", "highest_score", "lowest_score"]


class ExactDedupProcessorConfig(ProcessorConfig):
    """Configuration for exact duplicate row removal."""

    processor_type: Literal["exact-dedup"] = "exact-dedup"
    text_columns: list[str] = Field(min_length=1)
    keep: KeepPolicy = "first"
    score_column: str | None = None
    audit: bool = True

    @field_validator("text_columns")
    @classmethod
    def validate_text_columns(cls, value: list[str]) -> list[str]:
        """Validate deduplication columns."""
        if any(not column.strip() for column in value):
            raise ValueError("text_columns cannot contain empty values.")
        return value

    @model_validator(mode="after")
    def validate_keep_policy(self) -> Self:
        """Require score_column for score-based representative selection."""
        if self.keep in {"highest_score", "lowest_score"} and self.score_column is None:
            raise ValueError(f"score_column is required when keep={self.keep!r}.")
        return self


class ScoreFilterProcessorConfig(ProcessorConfig):
    """Configuration for filtering rows by an existing score column."""

    processor_type: Literal["score-filter"] = "score-filter"
    score_column: str
    min_score: float | None = None
    max_score: float | None = None
    keep_null_scores: bool = False
    audit: bool = True

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        """Require at least one threshold."""
        if self.min_score is None and self.max_score is None:
            raise ValueError("At least one of min_score or max_score is required.")
        if self.min_score is not None and self.max_score is not None and self.min_score > self.max_score:
            raise ValueError("min_score cannot be greater than max_score.")
        return self


class RemoteScoreColumnConfig(SingleColumnConfig):
    """Configuration for scoring rows through an external HTTP endpoint."""

    column_type: Literal["remote-score"] = "remote-score"
    endpoint_url: HttpUrl
    target_columns: list[str] = Field(min_length=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    headers: dict[str, str] | None = None
    score_path: str = "score"
    side_effect_output_column: str | None = None

    @property
    def required_columns(self) -> list[str]:
        """Columns sent to the remote scoring endpoint."""
        return self.target_columns

    @property
    def side_effect_columns(self) -> list[str]:
        """Optional metadata column populated with the full endpoint result."""
        return [self.side_effect_output_column] if self.side_effect_output_column else []

    @staticmethod
    def get_column_emoji() -> str:
        """Label displayed in Data Designer logs."""
        return "[remote-score]"

    @field_validator("target_columns")
    @classmethod
    def validate_target_columns(cls, value: list[str]) -> list[str]:
        """Validate request column names."""
        if any(not column.strip() for column in value):
            raise ValueError("target_columns cannot contain empty values.")
        return value

    @field_validator("score_path")
    @classmethod
    def validate_score_path(cls, value: str) -> str:
        """Validate dot-separated response path."""
        parts = value.split(".")
        if any(not part.strip() for part in parts):
            raise ValueError("score_path must be a non-empty dot-separated path.")
        return value
