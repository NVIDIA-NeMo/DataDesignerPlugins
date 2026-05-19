# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor

from data_designer_curator.audit import write_audit
from data_designer_curator.config import ScoreFilterProcessorConfig

if TYPE_CHECKING:
    import pandas as pd


class ScoreFilterProcessor(Processor[ScoreFilterProcessorConfig]):
    """Keep rows whose score column falls within configured thresholds."""

    def process_after_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter the final generated dataset."""
        if self.config.score_column not in data.columns:
            raise ValueError(f"Missing score column: {self.config.score_column!r}")

        scores = data[self.config.score_column]
        mask = scores.notna()
        if self.config.min_score is not None:
            mask &= scores >= self.config.min_score
        if self.config.max_score is not None:
            mask &= scores <= self.config.max_score
        if self.config.keep_null_scores:
            mask |= scores.isna()

        if self.config.audit:
            write_audit(
                self._build_audit(data, mask),
                self.artifact_storage.processors_outputs_path / self.config.name,
            )

        return data.loc[mask].reset_index(drop=True)

    def _build_audit(self, data: pd.DataFrame, keep_mask: pd.Series) -> pd.DataFrame:
        audit = data[[self.config.score_column]].copy()
        audit["_dd_original_index"] = data.index
        audit["_dd_processor_name"] = self.config.name
        audit["_dd_action"] = keep_mask.map({True: "kept", False: "dropped"}).to_numpy()
        audit["_dd_reason"] = "score threshold"
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
                self.config.score_column,
            ]
        ]
