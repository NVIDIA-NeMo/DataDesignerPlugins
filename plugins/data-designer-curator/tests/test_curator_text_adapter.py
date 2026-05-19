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


class FakeDocumentBatch:
    def __init__(self, *, task_id: str, dataset_name: str, data: pd.DataFrame) -> None:
        self.task_id = task_id
        self.dataset_name = dataset_name
        self.data = data

    def to_pandas(self) -> pd.DataFrame:
        return self.data


class FakeModify:
    def __init__(self, *, modifier_fn: list[object], input_fields: str) -> None:
        self.modifier_fn = modifier_fn
        self.input_fields = input_fields

    def process(self, batch: FakeDocumentBatch) -> FakeDocumentBatch:
        data = batch.to_pandas()
        for modifier in self.modifier_fn:
            data[self.input_fields] = data[self.input_fields].apply(modifier.modify_document)
        return FakeDocumentBatch(task_id=batch.task_id, dataset_name=batch.dataset_name, data=data)


class FakeModifier:
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def modify_document(self, text: str) -> str:
        return f"{text}{self.suffix}"


class FakeScoreFilter:
    def __init__(
        self,
        *,
        filter_obj: object | list[object],
        text_field: str | list[str],
        score_field: str | list[str | None] | None,
        invert: bool | list[bool],
    ) -> None:
        self.filter_obj = filter_obj if isinstance(filter_obj, list) else [filter_obj]
        self.text_field = text_field if isinstance(text_field, list) else [text_field]
        self.score_field = score_field if isinstance(score_field, list) else [score_field]
        self.invert = invert if isinstance(invert, list) else [invert]

    def compute_filter_mask(
        self,
        data: pd.DataFrame,
        filter_obj: object,
        text_field: str,
        score_field: str | None,
        invert: bool,
    ) -> pd.Series:
        scores = data[text_field].map(filter_obj.score_document)
        if score_field is not None:
            data[score_field] = scores
        mask = scores.map(filter_obj.keep_document)
        return ~mask if invert else mask


class FakeMinWordsFilter:
    def __init__(self, min_words: int) -> None:
        self.min_words = min_words

    def score_document(self, text: str) -> int:
        return len(text.split())

    def keep_document(self, score: int) -> bool:
        return score >= self.min_words


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


def test_modify_uses_curator_modify(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(curator_text, "_import_curator_modify", lambda: (FakeModify, FakeDocumentBatch))
    monkeypatch.setattr(curator_text, "_build_modifier", lambda spec: FakeModifier(spec["params"]["suffix"]))
    data = pd.DataFrame({"text": ["a", "b"]})

    output = CuratorTextAdapter().modify(
        data=data,
        input_field="text",
        output_field="clean_text",
        modifiers=[
            {"primitive": "fake", "params": {"suffix": "!"}},
            {"primitive": "fake", "params": {"suffix": "?"}},
        ],
    )

    assert output["text"].tolist() == ["a", "b"]
    assert output["clean_text"].tolist() == ["a!?", "b!?"]


def test_text_filter_uses_curator_score_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(curator_text, "_import_curator_score_filter", lambda: FakeScoreFilter)
    monkeypatch.setattr(
        curator_text,
        "_build_text_filter",
        lambda spec: FakeMinWordsFilter(spec["params"]["min_words"]),
    )
    data = pd.DataFrame({"text": ["short", "two words", "three word row"]})

    output, mask, scored = CuratorTextAdapter().text_filter(
        data=data,
        default_text_field="text",
        filters=[{"primitive": "word_count", "params": {"min_words": 2}, "score_field": "word_count"}],
    )

    assert mask.tolist() == [False, True, True]
    assert output["text"].tolist() == ["two words", "three word row"]
    assert scored["word_count"].tolist() == [1, 2, 3]
