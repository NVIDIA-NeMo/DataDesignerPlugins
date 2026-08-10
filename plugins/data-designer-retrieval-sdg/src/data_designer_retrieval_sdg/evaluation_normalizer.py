# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-batch normalization for structured QA evaluation output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from data_designer.engine.processing.processors.base import Processor

from data_designer_retrieval_sdg.config import QAEvaluationNormalizerConfig
from data_designer_retrieval_sdg.models import QAPairEvaluations

if TYPE_CHECKING:
    import pandas as pd


def normalize_qa_evaluations(value: Any) -> Any:
    """Return a JSON-compatible evaluation value with stable numeric types."""
    if value is None:
        return None
    return QAPairEvaluations.model_validate(value).model_dump(mode="json")


class QAEvaluationNormalizer(Processor[QAEvaluationNormalizerConfig]):
    """Normalize structured QA evaluations before Parquet checkpointing."""

    def process_after_batch(self, data: pd.DataFrame, *, current_batch_number: int | None) -> pd.DataFrame:
        """Coerce overall scores to floats without changing row count or order."""
        column_name = self.config.column_name
        if column_name not in data.columns:
            raise ValueError(f"QA evaluation column {column_name!r} was not found in the generated batch")

        normalized = data.copy()
        normalized[column_name] = normalized[column_name].map(normalize_qa_evaluations)
        return normalized
