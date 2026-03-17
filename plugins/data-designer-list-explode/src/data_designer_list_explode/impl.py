# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from data_designer_list_explode.config import ListExplodeColumnConfig

if TYPE_CHECKING:
    pass


class ListExplodeColumnGenerator(ColumnGeneratorFullColumn[ListExplodeColumnConfig]):
    """Explodes a list-valued column so each element becomes its own row."""

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Expand list-valued cells in the source column into separate rows.

        Args:
            data: Input DataFrame containing a column of lists.

        Returns:
            DataFrame with one row per list element.  The exploded scalar
            values are stored in the column given by ``self.config.name``.
        """
        source_col = self.config.source_column
        output_col = self.config.name

        result = data.explode(source_col, ignore_index=True)

        if output_col != source_col:
            result[output_col] = result[source_col]

        if self.config.drop_empty:
            result = _drop_empty_rows(result, output_col)

        return result


def _drop_empty_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove rows where *column* is ``None``, ``NaN``, or an empty list.

    Args:
        df: DataFrame to filter.
        column: Column name to inspect.

    Returns:
        Filtered DataFrame with a reset integer index.
    """
    mask = df[column].apply(_is_non_empty)
    return df.loc[mask].reset_index(drop=True)


def _is_non_empty(value: object) -> bool:
    """Return ``True`` when *value* is a meaningful scalar (not null / empty).

    Args:
        value: A cell value to test.

    Returns:
        ``False`` for ``None``, ``NaN``, and empty lists; ``True`` otherwise.
    """
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    return True
