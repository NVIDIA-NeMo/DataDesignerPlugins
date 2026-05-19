# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor

from data_designer_curator.adapters.curator_text import (
    CURATOR_ID_COLUMN,
    CURATOR_TEXT_COLUMN,
    ORIGINAL_INDEX_COLUMN,
    CuratorTextAdapter,
)
from data_designer_curator.audit import write_audit
from data_designer_curator.config import ExactDedupProcessorConfig

if TYPE_CHECKING:
    import pandas as pd


class ExactDedupProcessor(Processor[ExactDedupProcessorConfig]):
    """Drop exact duplicate rows through NeMo Curator."""

    def process_after_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate the final generated dataset."""
        self._validate_columns(data)

        working = data.copy()
        working[ORIGINAL_INDEX_COLUMN] = data.index
        output = CuratorTextAdapter().exact_dedup(
            data=working,
            text_columns=self.config.text_columns,
            id_column=self.config.id_column,
            hash_method=self.config.hash_method,
            cache_dir=self._cache_dir(),
            execution=self.config.execution,
        )
        if ORIGINAL_INDEX_COLUMN not in output.columns:
            raise ValueError(f"Curator output must preserve {ORIGINAL_INDEX_COLUMN!r}.")

        output = output.sort_values(ORIGINAL_INDEX_COLUMN, kind="stable")
        kept_indexes = set(output[ORIGINAL_INDEX_COLUMN].tolist())
        keep_mask = working[ORIGINAL_INDEX_COLUMN].isin(kept_indexes)

        if self.config.audit:
            write_audit(
                self._build_audit(working, keep_mask),
                self.artifact_storage.processors_outputs_path / self.config.name,
            )

        return output.drop(
            columns=[ORIGINAL_INDEX_COLUMN, CURATOR_ID_COLUMN, CURATOR_TEXT_COLUMN], errors="ignore"
        ).reset_index(drop=True)

    def _validate_columns(self, data: pd.DataFrame) -> None:
        columns = list(self.config.text_columns)
        if self.config.id_column is not None:
            columns.append(self.config.id_column)
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise ValueError(f"Missing dedup columns: {missing!r}")

    def _cache_dir(self) -> Path:
        if self.config.cache_dir is not None:
            return Path(self.config.cache_dir)
        return self.artifact_storage.processors_outputs_path / self.config.name / "cache"

    def _build_audit(
        self,
        data: pd.DataFrame,
        keep_mask: pd.Series,
    ) -> pd.DataFrame:
        audit = data[[ORIGINAL_INDEX_COLUMN]].copy()
        audit["_dd_processor_name"] = self.config.name
        audit["_dd_action"] = keep_mask.map({True: "kept", False: "duplicate"}).to_numpy()
        audit["_dd_reason"] = audit["_dd_action"].map(
            {
                "kept": "selected representative",
                "duplicate": "exact duplicate",
            }
        )
        audit["_dd_group_id"] = None
        audit["_dd_representative_index"] = None
        return audit[
            [
                ORIGINAL_INDEX_COLUMN,
                "_dd_processor_name",
                "_dd_action",
                "_dd_reason",
                "_dd_group_id",
                "_dd_representative_index",
            ]
        ]
