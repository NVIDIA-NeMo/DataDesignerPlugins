# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for QA evaluation score normalization."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pyarrow.dataset as ds

from data_designer_retrieval_sdg.config import QAEvaluationNormalizerConfig
from data_designer_retrieval_sdg.evaluation_normalizer import QAEvaluationNormalizer


def _qa_evaluations(overall_score: int | float) -> dict:
    criterion = {"score": 9, "justification": "good"}
    return {
        "evaluations": [
            {
                "relevance": criterion,
                "accuracy": criterion,
                "context_support": criterion,
                "clarity": criterion,
                "overall": {"score": overall_score, "assessment": "good"},
                "improvements": "none",
            }
        ]
    }


def test_integral_and_fractional_scores_share_a_parquet_schema(tmp_path: Path) -> None:
    processor = QAEvaluationNormalizer(
        config=QAEvaluationNormalizerConfig(name="normalize_scores"),
        resource_provider=MagicMock(),
    )

    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir()
    for batch_number, score in enumerate((9, 9.5)):
        batch = pd.DataFrame({"qa_evaluations": [_qa_evaluations(score)]})
        normalized = processor.process_after_batch(batch, current_batch_number=batch_number)
        normalized.to_parquet(parquet_dir / f"batch_{batch_number:05d}.parquet", index=False)

    records = ds.dataset(parquet_dir, format="parquet").to_table().to_pylist()
    scores = [record["qa_evaluations"]["evaluations"][0]["overall"]["score"] for record in records]
    assert scores == [9.0, 9.5]
    assert all(isinstance(score, float) for score in scores)
