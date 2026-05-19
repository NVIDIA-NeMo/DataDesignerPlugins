# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor

from data_designer_curator.adapters.curator_text import CuratorTextAdapter
from data_designer_curator.audit import write_audit
from data_designer_curator.config import CuratorTextFilterProcessorConfig

if TYPE_CHECKING:
    import pandas as pd


class CuratorTextFilterProcessor(Processor[CuratorTextFilterProcessorConfig]):
    """Filter rows through Curator document filter primitives."""

    def process_after_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter the final generated dataset with Curator."""
        self._validate_columns(data)
        specs = [filter_config.model_dump() for filter_config in self.config.filters]
        output, mask, scored = CuratorTextAdapter().text_filter(
            data=data,
            default_text_field=self.config.text_field,
            filters=specs,
        )

        if self.config.audit:
            write_audit(
                self._build_audit(scored, mask),
                self.artifact_storage.processors_outputs_path / self.config.name,
            )

        return output

    def _validate_columns(self, data: pd.DataFrame) -> None:
        missing = {
            filter_config.text_field or self.config.text_field
            for filter_config in self.config.filters
            if (filter_config.text_field or self.config.text_field) not in data.columns
        }
        if missing:
            raise ValueError(f"Missing text filter columns: {sorted(missing)!r}")

    def _build_audit(self, data: pd.DataFrame, keep_mask: pd.Series) -> pd.DataFrame:
        score_columns = [
            filter_config.score_field
            for filter_config in self.config.filters
            if filter_config.score_field is not None and filter_config.score_field in data.columns
        ]
        audit = data[score_columns].copy()
        audit["_dd_original_index"] = data.index
        audit["_dd_processor_name"] = self.config.name
        audit["_dd_action"] = keep_mask.map({True: "kept", False: "dropped"}).to_numpy()
        audit["_dd_reason"] = "curator text filter"
        audit["_dd_group_id"] = None
        audit["_dd_representative_index"] = None
        return audit[
            [
                "_dd_original_index",
                "_dd_processor_name",
                "_dd_action",
                "_dd_reason",
                "_dd_group_id",
                "_dd_representative_index",
                *score_columns,
            ]
        ]
