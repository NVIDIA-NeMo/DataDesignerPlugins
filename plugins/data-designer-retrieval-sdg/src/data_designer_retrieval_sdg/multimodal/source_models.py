# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source and split contracts for multimodal retrieval data generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SplitName = Literal["train", "validation", "evaluation"]
TEXT_EXTENSIONS = (".txt", ".md", ".markdown")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


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


class SourceDirectoryConfig(StrictModel):
    """Discover local text and image files as canonical retrieval units."""

    path: Path
    recursive: bool = True
    file_pattern: str = Field(default="*", min_length=1)
    file_extensions: list[str] = Field(default_factory=lambda: [*TEXT_EXTENSIONS, *IMAGE_EXTENSIONS], min_length=1)
    num_files: int | None = Field(default=None, ge=1)
    text_chunking: Literal["sentences", "none"] = "sentences"
    sentences_per_chunk: int | None = Field(default=None, ge=1)
    max_tokens_per_chunk: int | None = Field(default=1024, ge=1)
    tokenizer: str = Field(default="nvidia/Nemotron-3.5-Embed-1B-BF16", min_length=1)
    reserved_tokens: int = Field(default=0, ge=0)
    split_oversized_sentences: bool = False
    language: str = Field(default="source", min_length=1)

    @model_validator(mode="after")
    def validate_token_budget(self) -> SourceDirectoryConfig:
        """Require usable token limits and compatible preparation settings.

        Returns:
            Validated directory configuration.

        Raises:
            ValueError: Token-budget settings are incomplete or incompatible.
        """
        if self.max_tokens_per_chunk is None:
            if self.reserved_tokens or self.split_oversized_sentences:
                raise ValueError("reserved_tokens and split_oversized_sentences require max_tokens_per_chunk")
        elif self.reserved_tokens >= self.max_tokens_per_chunk:
            raise ValueError("reserved_tokens must be less than max_tokens_per_chunk")
        if self.split_oversized_sentences and self.text_chunking != "sentences":
            raise ValueError("split_oversized_sentences requires text_chunking: sentences")
        return self

    @field_validator("file_pattern")
    @classmethod
    def basename_pattern(cls, value: str) -> str:
        """Validate a basename glob, with recursion controlled independently.

        Args:
            value: Configured filename pattern.

        Returns:
            The unchanged pattern.

        Raises:
            ValueError: The pattern contains a path separator.
        """
        if "/" in value or "\\" in value:
            raise ValueError("file_pattern must match basenames; use recursive for subdirectories")
        return value

    @field_validator("file_extensions")
    @classmethod
    def supported_extensions(cls, values: list[str]) -> list[str]:
        """Normalize and validate the supported file-format filter.

        Args:
            values: Configured filename extensions.

        Returns:
            Unique lowercase extensions in their original order.

        Raises:
            ValueError: An extension is unsupported.
        """
        extensions = list(dict.fromkeys(value.lower() for value in values))
        unsupported = set(extensions) - set(TEXT_EXTENSIONS + IMAGE_EXTENSIONS)
        if unsupported:
            raise ValueError(f"Unsupported directory file extensions: {sorted(unsupported)}")
        return extensions


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
