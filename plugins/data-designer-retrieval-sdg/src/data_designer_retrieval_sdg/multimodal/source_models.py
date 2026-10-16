# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source and split contracts for multimodal retrieval data generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SplitName = Literal["train", "validation", "evaluation"]


class StrictModel(BaseModel):
    """Immutable model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RetrievalSource(StrictModel):
    """Convenience input for a single retrieval unit."""

    document_id: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    text: str = ""
    images: list[str] = Field(default_factory=list, max_length=1)
    language: str = Field(default="source", min_length=1)
    source_uri: str | None = None
    page_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_content(self) -> RetrievalSource:
        """Require text or image content."""
        if not self.text.strip() and not self.images:
            raise ValueError("a retrieval source must contain text or at least one image")
        if any(not image.strip() for image in self.images):
            raise ValueError("retrieval-source image paths must not be blank")
        return self


class SplitRatios(StrictModel):
    """Query-group-level train, validation, and evaluation ratios."""

    train: float = Field(default=0.8, ge=0.0, le=1.0)
    validation: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluation: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> SplitRatios:
        """Require ratios to sum to one."""
        if abs(self.train + self.validation + self.evaluation - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return self
