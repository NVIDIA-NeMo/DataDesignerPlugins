# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.base import SingleColumnConfig


class TextTransformColumnConfig(SingleColumnConfig):
    """Applies a text transformation (upper, lower, title) to an existing column."""

    column_type: Literal["text-transform"] = "text-transform"

    source_column: str
    transform: Literal["upper", "lower", "title"] = "upper"

    @staticmethod
    def get_column_emoji() -> str:
        return "🔄"

    @property
    def required_columns(self) -> list[str]:
        return [self.source_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []
