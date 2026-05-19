# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from data_designer_curator.adapters import curator_text
from data_designer_curator.adapters.curator_text import CuratorTextAdapter
from data_designer_curator.config import CuratorExecutionConfig


class FakeWorkflowResult:
    def __init__(self, metadata: dict[str, object] | None = None) -> None:
        self.metadata = metadata or {}

    def get_metadata(self, key: str) -> object | None:
        return self.metadata.get(key)


class FakeExactWorkflow:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def run(self) -> FakeWorkflowResult:
        input_path = Path(str(self.kwargs["input_path"]))
        output_path = Path(str(self.kwargs["output_path"])) / "ExactDuplicateIds"
        output_path.mkdir(parents=True, exist_ok=True)
        data = pd.read_parquet(input_path / "part.0.parquet")
        duplicate_ids = data.loc[data.duplicated(subset=[str(self.kwargs["text_field"])], keep="first")]
        duplicate_ids[[str(self.kwargs["id_field"])]].to_parquet(output_path / "part.0.parquet", index=False)
        return FakeWorkflowResult()


class FakeFuzzyWorkflow:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def run(self) -> FakeWorkflowResult:
        output_path = Path(str(self.kwargs["output_path"])) / "FuzzyDuplicateIds"
        output_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"_curator_dedup_id": [1]}).to_parquet(output_path / "part.0.parquet", index=False)
        return FakeWorkflowResult({"id_generator_path": str(Path(str(self.kwargs["output_path"])) / "ids.json")})


class FakeSemanticWorkflow:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def run(self) -> FakeWorkflowResult:
        input_path = Path(str(self.kwargs["input_path"]))
        output_path = Path(str(self.kwargs["output_path"])) / "deduplicated"
        output_path.mkdir(parents=True, exist_ok=True)
        data = pd.read_parquet(input_path / "part.0.parquet")
        data.drop_duplicates(subset=[str(self.kwargs["text_field"])], keep="first").to_parquet(
            output_path / "part.0.parquet",
            index=False,
        )
        return FakeWorkflowResult({"num_duplicates": 1})


class FakeRemovalWorkflow:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def run(self) -> FakeWorkflowResult:
        input_path = Path(str(self.kwargs["input_path"]))
        ids_path = Path(str(self.kwargs["ids_to_remove_path"]))
        output_path = Path(str(self.kwargs["output_path"]))
        output_path.mkdir(parents=True, exist_ok=True)
        data = pd.read_parquet(input_path / "part.0.parquet")
        ids = pd.read_parquet(ids_path / "part.0.parquet")
        if self.kwargs.get("id_generator_path") is not None:
            output = data.drop(index=ids.iloc[:, 0].astype(int).tolist())
        else:
            output = data.loc[~data[str(self.kwargs["id_field"])].isin(set(ids.iloc[:, 0].astype(str)))]
        output.to_parquet(output_path / "part.0.parquet", index=False)
        return FakeWorkflowResult({"num_duplicates_removed": len(data) - len(output)})


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


def test_dedup_exact_uses_curator_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeExactWorkflow.calls = []
    FakeRemovalWorkflow.calls = []
    monkeypatch.setattr(curator_text, "_import_exact_dedup_workflow", lambda: FakeExactWorkflow)
    monkeypatch.setattr(curator_text, "_import_duplicates_removal_workflow", lambda: FakeRemovalWorkflow)
    data = pd.DataFrame(
        {
            "_dd_original_index": [0, 1, 2],
            "text": ["same", "same", "different"],
            "value": [1, 2, 3],
        }
    )

    output = CuratorTextAdapter().dedup(
        data=data,
        dedup_type="exact",
        text_columns=["text"],
        id_column=None,
        params={"input_blocksize": "128MiB"},
        cache_dir=tmp_path,
        execution=CuratorExecutionConfig(mode="none"),
    )

    assert output["value"].tolist() == [1, 3]
    assert FakeExactWorkflow.calls[0]["assign_id"] is False
    assert FakeExactWorkflow.calls[0]["id_field"] == "_dd_curator_id"
    assert FakeExactWorkflow.calls[0]["input_blocksize"] == "128MiB"
    assert FakeRemovalWorkflow.calls[0]["duplicate_id_field"] == "_dd_curator_id"


def test_dedup_fuzzy_uses_curator_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeFuzzyWorkflow.calls = []
    FakeRemovalWorkflow.calls = []
    monkeypatch.setattr(curator_text, "_import_fuzzy_dedup_workflow", lambda: FakeFuzzyWorkflow)
    monkeypatch.setattr(curator_text, "_import_duplicates_removal_workflow", lambda: FakeRemovalWorkflow)
    data = pd.DataFrame(
        {
            "_dd_original_index": [0, 1, 2],
            "text": ["same", "same-ish", "different"],
            "value": [1, 2, 3],
        }
    )

    output = CuratorTextAdapter().dedup(
        data=data,
        dedup_type="fuzzy",
        text_columns=["text"],
        id_column=None,
        params={"char_ngrams": 24},
        cache_dir=tmp_path,
        execution=CuratorExecutionConfig(mode="none"),
    )

    assert output["value"].tolist() == [1, 3]
    assert FakeFuzzyWorkflow.calls[0]["char_ngrams"] == 24
    assert FakeRemovalWorkflow.calls[0]["id_generator_path"] == str(tmp_path / "fuzzy" / "ids.json")
    assert FakeRemovalWorkflow.calls[0]["duplicate_id_field"] == "_curator_dedup_id"


def test_dedup_semantic_uses_curator_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeSemanticWorkflow.calls = []
    monkeypatch.setattr(curator_text, "_import_semantic_dedup_workflow", lambda: FakeSemanticWorkflow)
    data = pd.DataFrame(
        {
            "_dd_original_index": [0, 1, 2],
            "text": ["same", "same", "different"],
            "value": [1, 2, 3],
        }
    )

    output = CuratorTextAdapter().dedup(
        data=data,
        dedup_type="semantic",
        text_columns=["text"],
        id_column=None,
        params={"n_clusters": 2},
        cache_dir=tmp_path,
        execution=CuratorExecutionConfig(mode="none"),
    )

    assert output["value"].tolist() == [1, 3]
    assert FakeSemanticWorkflow.calls[0]["perform_removal"] is True
    assert FakeSemanticWorkflow.calls[0]["id_field"] == "_dd_curator_id"
    assert FakeSemanticWorkflow.calls[0]["n_clusters"] == 2


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
