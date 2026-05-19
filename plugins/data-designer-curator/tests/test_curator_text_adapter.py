# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from data_designer_curator.adapters import curator_text
from data_designer_curator.adapters.curator_text import CuratorTextAdapter
from data_designer_curator.config import CuratorExecutionConfig


class FakeDocumentDataset:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @classmethod
    def from_pandas(cls, data: pd.DataFrame, npartitions: int) -> "FakeDocumentDataset":
        return cls(data)


class FakeExactDuplicates:
    def __init__(
        self,
        *,
        id_field: str,
        text_field: str,
        hash_method: str,
        perform_removal: bool,
        cache_dir: str,
    ) -> None:
        self.id_field = id_field
        self.text_field = text_field
        self.hash_method = hash_method
        self.perform_removal = perform_removal
        self.cache_dir = cache_dir

    def identify_duplicates(self, dataset: FakeDocumentDataset) -> FakeDocumentDataset:
        duplicate_mask = dataset.df.duplicated(subset=[self.text_field], keep="first")
        return FakeDocumentDataset(dataset.df.loc[duplicate_mask, [self.id_field]])

    def remove(self, dataset: FakeDocumentDataset, duplicates: FakeDocumentDataset) -> FakeDocumentDataset:
        duplicate_ids = set(duplicates.df[self.id_field].tolist())
        return FakeDocumentDataset(dataset.df.loc[~dataset.df[self.id_field].isin(duplicate_ids)])


class FakeFilter:
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


def test_exact_dedup_uses_curator_exact_duplicates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(curator_text, "_import_legacy_exact_dedup", lambda: (FakeExactDuplicates, FakeDocumentDataset))
    data = pd.DataFrame(
        {
            "_dd_original_index": [0, 1, 2],
            "text": ["same", "same", "different"],
            "value": [1, 2, 3],
        }
    )

    output = CuratorTextAdapter().exact_dedup(
        data=data,
        text_columns=["text"],
        id_column=None,
        hash_method="md5",
        cache_dir=tmp_path,
        execution=CuratorExecutionConfig(mode="none"),
    )

    assert output["value"].tolist() == [1, 3]


def test_score_filter_uses_curator_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(curator_text, "_import_curator_filter", lambda: FakeFilter)
    data = pd.DataFrame({"score": [0.2, None, 0.9], "text": ["low", "unknown", "high"]})

    output, mask = CuratorTextAdapter().score_filter(
        data=data,
        score_column="score",
        min_score=0.8,
        max_score=None,
        keep_null_scores=True,
    )

    assert mask.tolist() == [False, True, True]
    assert output["text"].tolist() == ["unknown", "high"]
