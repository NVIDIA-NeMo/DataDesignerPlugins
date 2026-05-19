# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest
from data_designer.engine.resources.resource_provider import ResourceProvider

from data_designer_curator.adapters import curator_text
from data_designer_curator.config import ScoreFilterProcessorConfig
from data_designer_curator.processors.filters import ScoreFilterProcessor


class FakeCuratorFilter:
    def __init__(self, *, filter_fn: object, filter_field: str) -> None:
        self.filter_fn = filter_fn
        self.filter_field = filter_field

    def compute_filter_mask(
        self,
        data: pd.DataFrame,
        filter_fn: object,
        filter_field: str,
        invert: bool,
    ) -> pd.Series:
        mask = data[filter_field].map(filter_fn)
        return ~mask if invert else mask


@pytest.fixture(autouse=True)
def fake_curator_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(curator_text, "_import_curator_filter", lambda: FakeCuratorFilter)


def test_score_filter_applies_min_score(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"score": [0.2, 0.8, 0.9], "text": ["low", "ok", "high"]})
    config = ScoreFilterProcessorConfig(name="filter", score_column="score", min_score=0.8, audit=False)

    output = ScoreFilterProcessor(config, resource_provider).process_after_generation(data)

    assert output["text"].tolist() == ["ok", "high"]


def test_score_filter_applies_max_score(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"score": [0.2, 0.8, 0.9], "text": ["low", "ok", "high"]})
    config = ScoreFilterProcessorConfig(name="filter", score_column="score", max_score=0.8, audit=False)

    output = ScoreFilterProcessor(config, resource_provider).process_after_generation(data)

    assert output["text"].tolist() == ["low", "ok"]


def test_score_filter_handles_null_scores(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"score": [0.2, None, 0.9], "text": ["low", "unknown", "high"]})
    config = ScoreFilterProcessorConfig(
        name="filter",
        score_column="score",
        min_score=0.8,
        keep_null_scores=True,
        audit=False,
    )

    output = ScoreFilterProcessor(config, resource_provider).process_after_generation(data)

    assert output["text"].tolist() == ["unknown", "high"]


def test_score_filter_writes_audit(resource_provider: ResourceProvider, tmp_path: Path) -> None:
    data = pd.DataFrame({"score": [0.2, 0.8, 0.9]})
    config = ScoreFilterProcessorConfig(name="filter", score_column="score", min_score=0.8)

    ScoreFilterProcessor(config, resource_provider).process_after_generation(data)

    audit_path = tmp_path / "dataset" / "processors-files" / "filter" / "audit.parquet"
    audit = pd.read_parquet(audit_path)
    assert audit["_dd_action"].tolist() == ["dropped", "kept", "kept"]
    assert audit["score"].tolist() == [0.2, 0.8, 0.9]


def test_score_filter_raises_for_missing_score_column(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"other": [0.2]})
    config = ScoreFilterProcessorConfig(name="filter", score_column="score", min_score=0.8, audit=False)

    with pytest.raises(ValueError, match="Missing score column"):
        ScoreFilterProcessor(config, resource_provider).process_after_generation(data)


def test_score_filter_requires_threshold() -> None:
    with pytest.raises(ValueError, match="At least one"):
        ScoreFilterProcessorConfig(name="filter", score_column="score")
