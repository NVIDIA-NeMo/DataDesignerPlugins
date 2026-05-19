# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest
from data_designer.engine.resources.resource_provider import ResourceProvider

from data_designer_curator.config import CuratorTextFilterConfig, CuratorTextFilterProcessorConfig
from data_designer_curator.processors.filters import CuratorTextFilterProcessor


class FakeCuratorTextAdapter:
    calls: list[dict[str, object]] = []

    def text_filter(self, **kwargs: object) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        self.calls.append(kwargs)
        data = kwargs["data"]
        assert isinstance(data, pd.DataFrame)
        scored = data.copy()
        scored["word_count"] = scored["text"].str.split().str.len()
        mask = scored["word_count"] >= 2
        return scored.loc[mask].reset_index(drop=True), mask, scored


def test_curator_text_filter_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
    resource_provider: ResourceProvider,
) -> None:
    FakeCuratorTextAdapter.calls = []
    monkeypatch.setattr("data_designer_curator.processors.filters.CuratorTextAdapter", FakeCuratorTextAdapter)
    data = pd.DataFrame({"text": ["short", "two words"]})
    config = CuratorTextFilterProcessorConfig(
        name="filter",
        text_field="text",
        filters=[
            CuratorTextFilterConfig(
                primitive="word_count",
                params={"min_words": 2},
                score_field="word_count",
            )
        ],
        audit=False,
    )

    output = CuratorTextFilterProcessor(config, resource_provider).process_after_generation(data)

    assert output["text"].tolist() == ["two words"]
    assert output["word_count"].tolist() == [2]
    assert FakeCuratorTextAdapter.calls[0]["default_text_field"] == "text"
    assert FakeCuratorTextAdapter.calls[0]["filters"] == [
        {
            "primitive": "word_count",
            "params": {"min_words": 2},
            "text_field": None,
            "score_field": "word_count",
            "invert": False,
        }
    ]


def test_curator_text_filter_writes_audit(
    monkeypatch: pytest.MonkeyPatch,
    resource_provider: ResourceProvider,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("data_designer_curator.processors.filters.CuratorTextAdapter", FakeCuratorTextAdapter)
    data = pd.DataFrame({"text": ["short", "two words"]})
    config = CuratorTextFilterProcessorConfig(
        name="filter",
        text_field="text",
        filters=[CuratorTextFilterConfig(primitive="word_count", score_field="word_count")],
    )

    CuratorTextFilterProcessor(config, resource_provider).process_after_generation(data)

    audit_path = tmp_path / "dataset" / "processors-files" / "filter" / "audit.parquet"
    audit = pd.read_parquet(audit_path)
    assert audit["_dd_action"].tolist() == ["dropped", "kept"]
    assert audit["word_count"].tolist() == [1, 2]


def test_curator_text_filter_raises_for_missing_text_column(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"other": ["short"]})
    config = CuratorTextFilterProcessorConfig(
        name="filter",
        text_field="text",
        filters=[CuratorTextFilterConfig(primitive="word_count")],
        audit=False,
    )

    with pytest.raises(ValueError, match="Missing text filter columns"):
        CuratorTextFilterProcessor(config, resource_provider).process_after_generation(data)
