# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest
from data_designer.engine.resources.resource_provider import ResourceProvider

from data_designer_curator.config import ExactDedupProcessorConfig
from data_designer_curator.processors.dedup import ExactDedupProcessor


def test_exact_dedup_keeps_first(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"text": ["same", "same", "different"], "value": [1, 2, 3]})
    config = ExactDedupProcessorConfig(name="dedup", text_columns=["text"], audit=False)

    output = ExactDedupProcessor(config, resource_provider).process_after_generation(data)

    assert output.to_dict(orient="records") == [
        {"text": "same", "value": 1},
        {"text": "different", "value": 3},
    ]


def test_exact_dedup_keeps_last(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"text": ["same", "same", "different"], "value": [1, 2, 3]})
    config = ExactDedupProcessorConfig(name="dedup", text_columns=["text"], keep="last", audit=False)

    output = ExactDedupProcessor(config, resource_provider).process_after_generation(data)

    assert output.to_dict(orient="records") == [
        {"text": "same", "value": 2},
        {"text": "different", "value": 3},
    ]


@pytest.mark.parametrize(
    ("keep", "expected_value"),
    [
        ("highest_score", 2),
        ("lowest_score", 1),
    ],
)
def test_exact_dedup_keeps_by_score(
    keep: str,
    expected_value: int,
    resource_provider: ResourceProvider,
) -> None:
    data = pd.DataFrame({"text": ["same", "same", "different"], "score": [0.1, 0.9, 0.5], "value": [1, 2, 3]})
    config = ExactDedupProcessorConfig(
        name="dedup",
        text_columns=["text"],
        keep=keep,
        score_column="score",
        audit=False,
    )

    output = ExactDedupProcessor(config, resource_provider).process_after_generation(data)

    assert output.loc[output["text"] == "same", "value"].item() == expected_value


def test_exact_dedup_uses_multiple_columns(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame(
        {
            "question": ["q1", "q1", "q1"],
            "answer": ["a1", "a1", "a2"],
        }
    )
    config = ExactDedupProcessorConfig(name="dedup", text_columns=["question", "answer"], audit=False)

    output = ExactDedupProcessor(config, resource_provider).process_after_generation(data)

    assert output.to_dict(orient="records") == [
        {"question": "q1", "answer": "a1"},
        {"question": "q1", "answer": "a2"},
    ]


def test_exact_dedup_writes_audit(resource_provider: ResourceProvider, tmp_path: Path) -> None:
    data = pd.DataFrame({"text": ["same", "same", "different"]})
    config = ExactDedupProcessorConfig(name="dedup", text_columns=["text"])

    ExactDedupProcessor(config, resource_provider).process_after_generation(data)

    audit_path = tmp_path / "dataset" / "processors-files" / "dedup" / "audit.parquet"
    audit = pd.read_parquet(audit_path)
    assert audit["_dd_action"].tolist() == ["kept", "duplicate", "kept"]
    assert audit["_dd_representative_index"].tolist() == [0, 0, 2]


def test_exact_dedup_raises_for_missing_column(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"other": ["same"]})
    config = ExactDedupProcessorConfig(name="dedup", text_columns=["text"], audit=False)

    with pytest.raises(ValueError, match="Missing dedup columns"):
        ExactDedupProcessor(config, resource_provider).process_after_generation(data)


def test_exact_dedup_requires_score_column_for_score_policy() -> None:
    with pytest.raises(ValueError, match="score_column is required"):
        ExactDedupProcessorConfig(name="dedup", text_columns=["text"], keep="highest_score")
