# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from data_designer.config.models import ModalityDataType
from data_designer.config.utils.constants import REASONING_CONTENT_COLUMN_POSTFIX, TRACE_COLUMN_POSTFIX
from data_designer.config.utils.image_helpers import ImageFormat
from data_designer.config.utils.misc import assert_valid_jinja2_template, extract_keywords_from_jinja2_template
from data_designer.config.utils.trace_type import TraceType
from pydantic import Field, model_validator
from typing_extensions import Self

VisualSearchToolName = Literal[
    "open_image",
    "get_image_info",
    "list_images",
    "crop_image",
    "transform_image",
    "edit_color",
]


class VisualSearchColumnConfig(SingleColumnConfig):
    """Configuration for image-grounded visual search with local image-operation tools.

    The column runs a vision-capable chat model with built-in image tools. Each tool
    returns an image ID, and subsequent calls may operate on any previous image ID,
    which lets the model branch from earlier points in the image history.
    """

    column_type: Literal["visual-search"] = "visual-search"

    image_column: str = Field(description="Column containing a local image path, URL, base64 string, or data URI.")
    prompt: str = Field(description="Jinja2 prompt template for the visual search task.")
    model_alias: str = Field(description="Alias of the vision-capable chat model to use.")
    system_prompt: str | None = Field(default=None, description="Optional Jinja2 system prompt template.")
    image_data_type: ModalityDataType | None = Field(
        default=None,
        description="Optional explicit format for values in image_column. Leave unset for auto-detection.",
    )
    image_format: ImageFormat | None = Field(
        default=None,
        description="Required when image_data_type is base64 and the image format cannot be auto-detected.",
    )
    image_placeholder: str | None = Field(
        default=None,
        description="Optional model-specific image token to include in text for endpoints that require it.",
    )
    max_tool_call_turns: int = Field(
        default=6,
        ge=1,
        description="Maximum tool-calling turns allowed for each row before the model must answer.",
    )
    allowed_tools: list[VisualSearchToolName] | None = Field(
        default=None,
        description="Optional allowlist of built-in visual tools. Defaults to all tools.",
    )
    attach_images_after_tool_calls: bool = Field(
        default=True,
        description="Attach resulting tool images back into the next model turn.",
    )
    include_image_history: bool = Field(
        default=True,
        description="Add a side-effect column with the tree of image operations and IDs.",
    )
    with_trace: TraceType = Field(default=TraceType.NONE, description="Optional chat trace capture mode.")
    extract_reasoning_content: bool = Field(
        default=False,
        description="If True, capture reasoning_content from the final assistant message.",
    )
    use_default_system_prompt: bool = Field(
        default=True,
        description="Prepend built-in instructions explaining image IDs and visual tools.",
    )

    @staticmethod
    def get_column_emoji() -> str:
        return "🔎"

    @property
    def required_columns(self) -> list[str]:
        required_cols = [self.image_column, *extract_keywords_from_jinja2_template(self.prompt)]
        if self.system_prompt:
            required_cols.extend(extract_keywords_from_jinja2_template(self.system_prompt))
        return list(dict.fromkeys(required_cols))

    @property
    def side_effect_columns(self) -> list[str]:
        return [
            *([f"{self.name}__image_history"] if self.include_image_history else []),
            *([f"{self.name}{TRACE_COLUMN_POSTFIX}"] if self.with_trace != TraceType.NONE else []),
            *([f"{self.name}{REASONING_CONTENT_COLUMN_POSTFIX}"] if self.extract_reasoning_content else []),
        ]

    @model_validator(mode="after")
    def validate_templates_and_image_format(self) -> Self:
        """Validate prompt templates and image modality settings."""
        assert_valid_jinja2_template(self.prompt)
        if self.system_prompt:
            assert_valid_jinja2_template(self.system_prompt)
        if self.image_data_type == ModalityDataType.BASE64 and self.image_format is None:
            raise ValueError("image_format is required when image_data_type is base64")
        return self
