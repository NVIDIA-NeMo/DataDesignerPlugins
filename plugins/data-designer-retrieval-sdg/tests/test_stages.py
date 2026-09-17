# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for deterministic query selection and answer assembly."""

from __future__ import annotations

import copy
from pathlib import Path

import data_designer.config as dd
import pandas as pd
import pytest
from data_designer.interface import DataDesigner
from pydantic import ValidationError

from data_designer_retrieval_sdg.models import RetrievalAnswer, RetrievalQuery
from data_designer_retrieval_sdg.stages import assemble_retrieval_pairs, select_retrieval_queries


def query(text: str = "What storage temperature does Drug X require?") -> dict:
    return dict(
        question=text,
        query_surface="question",
        question_complexity=2,
        query_type="contextual",
        reasoning_type="factual",
    )


def judgement(index: int, passes: bool = True) -> dict:
    return dict(
        candidate_index=index,
        standalone_query=passes,
        plausible_information_need=True,
        retrieval_discriminative=True,
        query_surface_correct=True,
        source_language_preserved=True,
        reason="Specific standalone query" if passes else "Ambiguous",
    )


def answer(index: int) -> dict:
    return dict(
        candidate_index=index,
        answer="20-25 C",
        evidence="Drug X storage: 20-25 C.",
        positive_unit_ids=["unit-1"],
        segment_ids=[1],
        hop_count=1,
        hop_contexts=[],
    )


def seed() -> dict:
    return {
        "deduplicated_queries": [query("What does the chart show?"), query(), query("Find Drug X dosing guidance.")],
        "source_blind_evaluations": {"evaluations": [judgement(2), judgement(0), judgement(1, False)]},
    }


def test_selection_blocks_deterministic_and_model_failures() -> None:
    result = select_retrieval_queries(seed())["query_selection"]
    assert [item["original_index"] for item in result["selected"]] == [2]
    assert [item["rejection_reason"] for item in result["rejected"]] == [
        "source_relative_query",
        "query_quality_rejected: Ambiguous",
    ]


def test_selection_quarantines_mark_reading_even_when_query_judge_accepts() -> None:
    question = "In Acme's 2024 revenue chart, which division's bar extends further?"
    row = {"deduplicated_queries": [query(question)], "source_blind_evaluations": {"evaluations": [judgement(0)]}}
    result = select_retrieval_queries(row)["query_selection"]
    assert result["selected"] == []
    assert result["rejected"][0]["question"] == question
    assert result["rejected"][0]["rejection_reason"] == "visual_mark_reading_query"


@pytest.mark.parametrize("evaluations", [[], [judgement(0), judgement(0)]])
def test_missing_or_duplicate_query_judgements_fail_closed(evaluations: list) -> None:
    row = {"deduplicated_queries": [query()], "source_blind_evaluations": {"evaluations": evaluations}}
    result = select_retrieval_queries(row)["query_selection"]
    assert result["selected"] == []
    assert result["rejected"][0]["rejection_reason"] == "missing_or_duplicate_query_judgement"


def test_assembly_keeps_exact_query_and_reindexes_judgement() -> None:
    row = select_retrieval_queries(seed())
    row["answer_generation"] = {"answers": [answer(0)]}
    original = copy.deepcopy(row)
    result = assemble_retrieval_pairs(row)
    assert row == original
    assert result["deduplicated_qa_pairs"][0]["question"] == seed()["deduplicated_queries"][2]["question"]
    assert result["query_quality_evaluations"]["evaluations"][0]["candidate_index"] == 0
    assert len(result["generation_diagnostics"]) == 2


def test_assembly_handles_missing_and_reordered_answers() -> None:
    row = select_retrieval_queries(
        {
            "deduplicated_queries": [query(str(i)) for i in range(3)],
            "source_blind_evaluations": {"evaluations": [judgement(i) for i in range(3)]},
        }
    )
    row["answer_generation"] = {"answers": [answer(2), answer(0)]}
    result = assemble_retrieval_pairs(row)
    assert [item["question"] for item in result["deduplicated_qa_pairs"]] == ["0", "2"]
    assert [item["candidate_index"] for item in result["query_quality_evaluations"]["evaluations"]] == [0, 1]
    assert result["generation_diagnostics"][0]["candidate_index"] == 1


def test_duplicate_answer_is_quarantined_and_unknown_index_fails() -> None:
    row = select_retrieval_queries(seed())
    row["answer_generation"] = {"answers": [answer(0), answer(0)]}
    assert assemble_retrieval_pairs(row)["deduplicated_qa_pairs"] == []
    row["answer_generation"] = {"answers": [answer(1)]}
    with pytest.raises(ValueError, match="out of range"):
        assemble_retrieval_pairs(row)


def test_empty_queries_survive_skipped_model_columns() -> None:
    row = select_retrieval_queries({"deduplicated_queries": [], "source_blind_evaluations": None})
    row["answer_generation"] = None
    result = assemble_retrieval_pairs(row)
    assert result["deduplicated_qa_pairs"] == []
    assert result["query_quality_evaluations"] == {"evaluations": []}


def test_schemas_prevent_joint_generation_and_answer_time_rewriting() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery.model_validate({**query(), "answer": "20-25 C"})
    with pytest.raises(ValidationError):
        RetrievalAnswer.model_validate({**answer(0), "question": "A rewritten query"})


def test_native_custom_stages_execute_in_data_designer(tmp_path: Path) -> None:
    row = seed()
    row["answer_generation"] = {"answers": [answer(0)]}
    builder = dd.DataDesignerConfigBuilder()
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=pd.DataFrame([row])))
    builder.add_column(dd.CustomColumnConfig(name="query_selection", generator_function=select_retrieval_queries))
    builder.add_column(dd.CustomColumnConfig(name="deduplicated_qa_pairs", generator_function=assemble_retrieval_pairs))
    result = DataDesigner(artifact_path=tmp_path).preview(builder, num_records=1).dataset.iloc[0]
    assert len(result["deduplicated_qa_pairs"]) == 1
    assert result["query_quality_evaluations"]["evaluations"][0]["candidate_index"] == 0

    designer = DataDesigner(artifact_path=tmp_path / "resumable")
    designer.set_run_config(dd.RunConfig(otel_metrics_port=None))
    first = designer.create(builder, num_records=1, dataset_name="staged")
    resumed = designer.create(builder, num_records=1, dataset_name="staged", resume="always")
    pd.testing.assert_frame_equal(first.load_dataset(), resumed.load_dataset())


def test_native_skips_preserve_empty_rows_and_diagnostics(tmp_path: Path) -> None:
    builder = dd.DataDesignerConfigBuilder()
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=pd.DataFrame([{"deduplicated_queries": []}])))
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="source_blind_evaluations",
            expr="{{ 1 / 0 }}",
            skip=dd.SkipConfig(when="{{ deduplicated_queries | length == 0 }}"),
        )
    )
    builder.add_column(
        dd.CustomColumnConfig(name="query_selection", generator_function=select_retrieval_queries, propagate_skip=False)
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="answer_generation",
            expr="{{ 1 / 0 }}",
            skip=dd.SkipConfig(when="{{ query_selection.selected | length == 0 }}"),
        )
    )
    builder.add_column(
        dd.CustomColumnConfig(
            name="deduplicated_qa_pairs", generator_function=assemble_retrieval_pairs, propagate_skip=False
        )
    )
    result = DataDesigner(artifact_path=tmp_path).preview(builder, num_records=1).dataset.iloc[0]
    assert len(result["deduplicated_qa_pairs"]) == 0
    assert result["query_quality_evaluations"] == {"evaluations": []}
