# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor

from data_designer_curator.audit import write_audit
from data_designer_curator.config import ExactDedupProcessorConfig

if TYPE_CHECKING:
    import pandas as pd


def fingerprint_values(values: tuple[object, ...]) -> str:
    """Return a stable exact-match fingerprint for row values."""
    payload = "\x1f".join("" if value is None else str(value) for value in values)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def fingerprint_rows(data: pd.DataFrame, columns: list[str]) -> list[str]:
    """Fingerprint selected columns for every row."""
    rows = data[columns].itertuples(index=False, name=None)
    return [fingerprint_values(row) for row in rows]


class ExactDedupProcessor(Processor[ExactDedupProcessorConfig]):
    """Drop exact duplicate rows based on one or more text columns."""

    def process_after_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate the final generated dataset."""
        self._validate_columns(data)

        working = data.copy()
        working["_dd_original_index"] = data.index
        working["_dd_group_id"] = fingerprint_rows(working, self.config.text_columns)

        ranked = self._sort_for_keep_policy(working)
        representatives = (
            ranked.drop_duplicates(subset=["_dd_group_id"], keep="first")
            .set_index("_dd_group_id")["_dd_original_index"]
            .to_dict()
        )
        keep_mask = working["_dd_original_index"].isin(representatives.values())

        if self.config.audit:
            write_audit(
                self._build_audit(working, keep_mask, representatives),
                self.artifact_storage.processors_outputs_path / self.config.name,
            )

        return data.loc[keep_mask.to_numpy()].reset_index(drop=True)

    def _validate_columns(self, data: pd.DataFrame) -> None:
        columns = list(self.config.text_columns)
        if self.config.score_column is not None:
            columns.append(self.config.score_column)
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise ValueError(f"Missing dedup columns: {missing!r}")

    def _sort_for_keep_policy(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.config.keep == "first":
            return data
        if self.config.keep == "last":
            return data.iloc[::-1]

        score_column = self.config.score_column
        if score_column is None:
            raise ValueError("score_column is required for score-based keep policies.")
        ascending = self.config.keep == "lowest_score"
        return data.sort_values(score_column, ascending=ascending, kind="stable")

    def _build_audit(
        self,
        data: pd.DataFrame,
        keep_mask: pd.Series,
        representatives: dict[str, object],
    ) -> pd.DataFrame:
        audit = data[["_dd_original_index", "_dd_group_id"]].copy()
        audit["_dd_processor_name"] = self.config.name
        audit["_dd_action"] = keep_mask.map({True: "kept", False: "duplicate"}).to_numpy()
        audit["_dd_reason"] = audit["_dd_action"].map(
            {
                "kept": "selected representative",
                "duplicate": "exact duplicate",
            }
        )
        audit["_dd_representative_index"] = data["_dd_group_id"].map(representatives)
        return audit[
            [
                "_dd_original_index",
                "_dd_processor_name",
                "_dd_action",
                "_dd_reason",
                "_dd_group_id",
                "_dd_representative_index",
            ]
        ]
