# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from data_designer.config.base import ConfigBase, ProcessorConfig
from pydantic import Field, field_validator

CuratorExecutionMode = Literal["none", "local_ray", "existing_ray"]
CuratorModifierPrimitive = Literal[
    "boilerplate_string",
    "line_remover",
    "markdown_remover",
    "newline_normalizer",
    "quotation_remover",
    "slicer",
    "unicode_reformatter",
    "url_remover",
]
CuratorTextFilterPrimitive = Literal[
    "alpha",
    "boilerplate_string",
    "bullets",
    "common_english_words",
    "ellipsis",
    "general_comment_to_code",
    "histogram",
    "html_boilerplate",
    "long_word",
    "mean_word_length",
    "non_alpha_numeric",
    "number_of_lines_of_code",
    "numbers",
    "parentheses",
    "per_extension",
    "pornographic_urls",
    "punctuation",
    "python_comment_to_code",
    "repeated_lines",
    "repeated_lines_by_char",
    "repeated_paragraphs",
    "repeated_paragraphs_by_char",
    "repeating_duplicate_ngrams",
    "repeating_top_ngrams",
    "substring",
    "symbols_to_words",
    "token_count",
    "tokenizer_fertility",
    "urls",
    "whitespace",
    "word_count",
    "words_without_alphabets",
    "xml_header",
]


class CuratorExecutionConfig(ConfigBase):
    """Connection settings for Curator-backed processors."""

    mode: CuratorExecutionMode = "local_ray"
    ray_address: str | None = None
    num_cpus: int | None = Field(default=None, ge=1)
    num_gpus: int | None = Field(default=None, ge=0)
    object_store_memory: int | None = Field(default=None, ge=1)
    enable_object_spilling: bool = False
    include_dashboard: bool = True
    ray_temp_dir: str | None = None
    metrics_dir: str | None = None
    client_kwargs: dict[str, Any] = Field(default_factory=dict)


class ExactDedupProcessorConfig(ProcessorConfig):
    """Configuration for Curator-backed exact duplicate row removal."""

    processor_type: Literal["exact-dedup"] = "exact-dedup"
    text_columns: list[str] = Field(min_length=1)
    id_column: str | None = None
    hash_method: Literal["md5"] = "md5"
    cache_dir: str | None = None
    execution: CuratorExecutionConfig = Field(default_factory=CuratorExecutionConfig)
    audit: bool = True

    @field_validator("text_columns")
    @classmethod
    def validate_text_columns(cls, value: list[str]) -> list[str]:
        """Validate deduplication columns."""
        if any(not column.strip() for column in value):
            raise ValueError("text_columns cannot contain empty values.")
        return value


class CuratorModifierConfig(ConfigBase):
    """A Curator text modifier primitive and constructor parameters."""

    primitive: CuratorModifierPrimitive
    params: dict[str, Any] = Field(default_factory=dict)


class CuratorModifyProcessorConfig(ProcessorConfig):
    """Configuration for applying a chain of Curator text modifiers."""

    processor_type: Literal["curator-modify"] = "curator-modify"
    input_field: str
    modifiers: list[CuratorModifierConfig] = Field(min_length=1)
    output_field: str | None = None

    @field_validator("input_field", "output_field")
    @classmethod
    def validate_field(cls, value: str | None) -> str | None:
        """Validate column names."""
        if value is not None and not value.strip():
            raise ValueError("field names cannot be empty.")
        return value


class CuratorTextFilterConfig(ConfigBase):
    """A Curator document filter primitive and constructor parameters."""

    primitive: CuratorTextFilterPrimitive
    params: dict[str, Any] = Field(default_factory=dict)
    text_field: str | None = None
    score_field: str | None = None
    invert: bool = False

    @field_validator("text_field", "score_field")
    @classmethod
    def validate_field(cls, value: str | None) -> str | None:
        """Validate optional column names."""
        if value is not None and not value.strip():
            raise ValueError("field names cannot be empty.")
        return value


class CuratorTextFilterProcessorConfig(ProcessorConfig):
    """Configuration for applying Curator document filters."""

    processor_type: Literal["curator-text-filter"] = "curator-text-filter"
    text_field: str
    filters: list[CuratorTextFilterConfig] = Field(min_length=1)
    audit: bool = True

    @field_validator("text_field")
    @classmethod
    def validate_text_field(cls, value: str) -> str:
        """Validate the default text column."""
        if not value.strip():
            raise ValueError("text_field cannot be empty.")
        return value
