# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor

from data_designer_curator.adapters.curator_text import CuratorTextAdapter
from data_designer_curator.config import CuratorModifyProcessorConfig

if TYPE_CHECKING:
    import pandas as pd


class CuratorModifyProcessor(Processor[CuratorModifyProcessorConfig]):
    """Apply Curator text modifier primitives to a dataset column."""

    def process_after_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Modify the final generated dataset with Curator."""
        if self.config.input_field not in data.columns:
            raise ValueError(f"Missing modifier input column: {self.config.input_field!r}")

        return CuratorTextAdapter().modify(
            data=data,
            input_field=self.config.input_field,
            output_field=self.config.output_field,
            modifiers=[modifier.model_dump() for modifier in self.config.modifiers],
        )
