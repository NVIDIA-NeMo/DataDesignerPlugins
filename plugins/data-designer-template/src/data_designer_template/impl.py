# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from data_designer_template.config import TextTransformColumnConfig

if TYPE_CHECKING:
    import pandas as pd


class TextTransformColumnGenerator(ColumnGeneratorFullColumn[TextTransformColumnConfig]):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        source = data[self.config.source_column]
        transform = self.config.transform
        if transform == "upper":
            data[self.config.name] = source.str.upper()
        elif transform == "lower":
            data[self.config.name] = source.str.lower()
        elif transform == "title":
            data[self.config.name] = source.str.title()
        return data
