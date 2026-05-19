# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest
from data_designer.engine.resources.resource_provider import ResourceProvider

from data_designer_curator.adapters.curator_text import ORIGINAL_INDEX_COLUMN
from data_designer_curator.config import CuratorDedupProcessorConfig
from data_designer_curator.errors import CuratorDependencyError
from data_designer_curator.processors.dedup import CuratorDedupProcessor


class FakeCuratorTextAdapter:
    calls: list[dict[str, object]] = []

    def dedup(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(kwargs)
        data = kwargs["data"]
        assert isinstance(data, pd.DataFrame)
        return data.drop_duplicates(subset=["text"], keep="first")


def test_curator_dedup_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
    resource_provider: ResourceProvider,
) -> None:
    FakeCuratorTextAdapter.calls = []
    monkeypatch.setattr("data_designer_curator.processors.dedup.CuratorTextAdapter", FakeCuratorTextAdapter)
    data = pd.DataFrame({"text": ["same", "same", "different"], "value": [1, 2, 3]})
    config = CuratorDedupProcessorConfig(name="dedup", dedup_type="fuzzy", text_columns=["text"], audit=False)

    output = CuratorDedupProcessor(config, resource_provider).process_after_generation(data)

    assert output.to_dict(orient="records") == [
        {"text": "same", "value": 1},
        {"text": "different", "value": 3},
    ]
    call = FakeCuratorTextAdapter.calls[0]
    assert call["dedup_type"] == "fuzzy"
    assert call["text_columns"] == ["text"]
    assert call["params"] == {}
    assert call["id_column"] is None
    assert ORIGINAL_INDEX_COLUMN in call["data"].columns


def test_curator_dedup_writes_audit(
    monkeypatch: pytest.MonkeyPatch,
    resource_provider: ResourceProvider,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("data_designer_curator.processors.dedup.CuratorTextAdapter", FakeCuratorTextAdapter)
    data = pd.DataFrame({"text": ["same", "same", "different"]})
    config = CuratorDedupProcessorConfig(name="dedup", dedup_type="semantic", text_columns=["text"])

    CuratorDedupProcessor(config, resource_provider).process_after_generation(data)

    audit_path = tmp_path / "dataset" / "processors-files" / "dedup" / "audit.parquet"
    audit = pd.read_parquet(audit_path)
    assert audit["_dd_action"].tolist() == ["kept", "duplicate", "kept"]
    assert audit["_dd_reason"].tolist() == ["selected representative", "semantic duplicate", "selected representative"]


def test_curator_dedup_raises_for_missing_column(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"other": ["same"]})
    config = CuratorDedupProcessorConfig(name="dedup", text_columns=["text"], audit=False)

    with pytest.raises(ValueError, match="Missing dedup columns"):
        CuratorDedupProcessor(config, resource_provider).process_after_generation(data)


def test_curator_dedup_raises_without_curator(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"text": ["same"]})
    config = CuratorDedupProcessorConfig(name="dedup", text_columns=["text"], audit=False)

    with pytest.raises(CuratorDependencyError, match="NeMo Curator"):
        CuratorDedupProcessor(config, resource_provider).process_after_generation(data)
