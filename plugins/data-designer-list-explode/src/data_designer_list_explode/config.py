# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.base import SingleColumnConfig


class ListExplodeColumnConfig(SingleColumnConfig):
    """Explodes a column containing structured lists into one row per element.

    Given a source column where each cell holds a list of values, this
    generator expands the DataFrame so that every list element occupies its
    own row.  Non-list columns are duplicated to match.  The exploded scalar
    values are written to the output column specified by ``name``.

    Args:
        source_column: Name of the column containing lists to explode.
        drop_empty: When ``True``, rows where the source value is ``None``
            or an empty list are removed from the result.
        allow_resize: Always ``True`` for this plugin because exploding
            changes the number of rows.
    """

    column_type: Literal["list-explode"] = "list-explode"

    source_column: str
    drop_empty: bool = False
    allow_resize: bool = True

    @staticmethod
    def get_column_emoji() -> str:
        return "💥"

    @property
    def required_columns(self) -> list[str]:
        return [self.source_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []
