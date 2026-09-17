# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-contract tests for the unified retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from jinja2 import Template
from pydantic import ValidationError

import data_designer_retrieval_sdg as retrieval_sdg
from data_designer_retrieval_sdg.models import QuestionAnswerPair
from data_designer_retrieval_sdg.pipeline import (
    build_qa_generation_pipeline,
    build_retrieval_pipeline,
)
from data_designer_retrieval_sdg.retrieval import GeneratedRetrievalRecord, RetrievalSource
from data_designer_retrieval_sdg.retrieval.quality import deterministic_rejection_reason, query_rejection_reason
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


def _mixed_seed() -> DataFrameSeedSource:
    rows = [
        RetrievalSource(
            document_id="text-doc",
            unit_id="text-unit",
            text="The recommended adult dose is 20 mg.",
            language="en",
        ).to_seed_record(),
        RetrievalSource(
            document_id="image-doc",
            unit_id="image-unit",
            text="",
            images=["https://example.invalid/page.png"],
            language="en",
            page_number=1,
        ).to_seed_record(),
    ]
    return DataFrameSeedSource(df=pd.DataFrame(rows))


def _unified_builder() -> object:
    return build_retrieval_pipeline(
        _mixed_seed(),
        artifact_model_alias="artifact",
        generator_model_alias="generator",
        judge_model_alias="judge",
        embedding_model_alias="embed",
    )


def test_text_and_image_rows_use_one_staged_pipeline() -> None:
    builder = _unified_builder()
    columns = builder.get_column_configs()

    assert [column.name for column in columns] == [
        "document_artifacts",
        "query_generation",
        "deduplicated_queries",
        "source_blind_evaluations",
        "query_selection",
        "answer_generation",
        "deduplicated_qa_pairs",
        "source_assessments",
        "qa_evaluations",
    ]
    assert columns[0].multi_modal_context[0].column_name == "images"
    assert columns[1].multi_modal_context[0].column_name == "images"
    assert columns[3].multi_modal_context is None
    assert columns[5].multi_modal_context[0].column_name == "images"
    assert columns[7].multi_modal_context[0].column_name == "images"
    assert columns[8].multi_modal_context is None
    assert columns[0].multi_modal_context[0].get_contexts({"images": []}) == []
    assert "deduplicated_qa_pairs" in columns[7].required_columns
    assert "source_assessments" in columns[8].required_columns
    assert columns[5].skip.when == "{{ query_selection.selected | length == 0 }}"
    assert columns[6].propagate_skip is False


def test_query_judge_is_source_blind() -> None:
    query_column = _unified_builder().get_column_config("source_blind_evaluations")

    assert "{{ text }}" not in query_column.prompt
    assert "candidate.answer" not in query_column.prompt
    assert "candidate.evidence" not in query_column.prompt
    assert query_column.multi_modal_context is None
    assert "standalone_query" in query_column.prompt
    assert "retrieval_discriminative" in query_column.prompt
    assert "source_language_preserved" in query_column.prompt


def test_grounding_judge_distinguishes_source_text_from_generated_evidence() -> None:
    column = _unified_builder().get_column_config("source_assessments")
    rendered = Template(column.prompt).render(
        retrieval_units=[{"unit_id": "image-unit", "segment_id": 1, "text": ""}],
        deduplicated_qa_pairs=[],
        query_quality_evaluations={"evaluations": []},
    )

    assert "supplied_text_present=false" in rendered
    assert "<supplied_source_text></supplied_source_text>" in rendered
    assert "never authoritative source text" in rendered
    assert "text_only is impossible" in rendered


def test_text_compatibility_builder_delegates_to_same_pipeline(tmp_path: Path) -> None:
    builder = build_qa_generation_pipeline(DocumentChunkerSeedSource(path=str(tmp_path)))

    assert [column.name for column in builder.get_column_configs()] == [
        column.name for column in _unified_builder().get_column_configs()
    ]


def test_generation_schema_requires_retrieval_evidence() -> None:
    with pytest.raises(ValidationError):
        QuestionAnswerPair.model_validate(
            {
                "question": "What dose is recommended for adults with condition X?",
                "answer": "20 mg",
                "question_complexity": 4,
                "query_type": "contextual",
                "reasoning_type": "factual",
                "segment_ids": [1],
                "hop_count": 1,
                "hop_contexts": [],
            }
        )


def test_generator_schema_rejects_judge_owned_modality() -> None:
    with pytest.raises(ValidationError):
        QuestionAnswerPair.model_validate(
            {
                "question": "Find dosage guidance for adults with condition X.",
                "answer": "20 mg",
                "evidence": "The dosage row specifies 20 mg.",
                "query_surface": "instruction",
                "positive_unit_ids": ["unit-1"],
                "question_complexity": 4,
                "query_type": "contextual",
                "reasoning_type": "factual",
                "segment_ids": [1],
                "hop_count": 1,
                "hop_contexts": [],
                "evidence_modality": "image_grounded",
            }
        )


def test_generated_record_rejects_duplicate_judge_indexes() -> None:
    row = {
        "source_id": "source-1",
        "retrieval_units": [{"unit_id": "unit-1", "document_id": "doc-1", "text": "Source text."}],
        "deduplicated_qa_pairs": [],
        "query_quality_evaluations": {
            "evaluations": [
                {
                    "candidate_index": 0,
                    "standalone_query": True,
                    "plausible_information_need": True,
                    "retrieval_discriminative": True,
                    "query_surface_correct": True,
                    "source_language_preserved": True,
                    "reason": "valid",
                },
                {
                    "candidate_index": 0,
                    "standalone_query": True,
                    "plausible_information_need": True,
                    "retrieval_discriminative": True,
                    "query_surface_correct": True,
                    "source_language_preserved": True,
                    "reason": "duplicate",
                },
            ]
        },
        "qa_evaluations": {"evaluations": []},
    }
    with pytest.raises(ValidationError, match="duplicate candidate indexes"):
        GeneratedRetrievalRecord.model_validate(row)


@pytest.mark.parametrize(
    ("question", "answer", "expected"),
    [
        ("What response rate is shown in the chart?", "14%", "source_relative_query"),
        ("What trend is shown in the data for product X?", "A decline.", "source_relative_query"),
        ("Which treatment arm reported a １４％ response rate?", "14%", "numeric_answer_repeated_in_query"),
        ("What does GDUFA require from applicants?", "GDUFA", None),
        (
            "Which projects received funding?",
            "The projects are not specified in the provided material.",
            "answer_does_not_resolve_query",
        ),
    ],
)
def test_deterministic_rejection_is_high_precision(
    question: str,
    answer: str,
    expected: str | None,
) -> None:
    assert deterministic_rejection_reason(question, answer) == expected


def test_deterministic_rejection_catches_surface_mismatch() -> None:
    assert (
        deterministic_rejection_reason(
            "What dose is recommended for adults?",
            "20 mg",
            "instruction",
        )
        == "query_surface_mismatch"
    )


@pytest.mark.parametrize(
    "question",
    [
        "In the May 2025 shipping totals bar chart, which region's bar extends further, North or South?",
        "Which bar is taller in Acme's 2024 sales graph?",
        "In Acme's revenue chart, whose bar is shorter?",
        "For Acme's sales bar graph, which bars extend farther?",
    ],
)
def test_explicit_chart_mark_reading_is_rejected(question: str) -> None:
    """A scoped chart-reading task still lacks an underlying information need."""
    assert query_rejection_reason(question) == "visual_mark_reading_query"
    assert deterministic_rejection_reason(question, "North") == "visual_mark_reading_query"


@pytest.mark.parametrize(
    "question",
    [
        "How did North and South shipping totals compare in May 2025?",
        "North South shipping totals May 2025 bar chart",
        "Find design guidance for comparing unequal-length bars in a bar chart.",
        "What makes unequal bar lengths misleading in revenue graphs?",
        "Which support bar is longer on Model Z's physical frame?",
        "Find photographs distinguishing Model Z's long and short support bars.",
        "Which chart type best compares quantities with different measurement units?",
    ],
)
def test_legitimate_chart_and_appearance_information_needs_remain_allowed(question: str) -> None:
    assert query_rejection_reason(question) is None


def test_generation_and_blind_judge_distinguish_mark_geometry_from_information_needs() -> None:
    builder = _unified_builder()
    for name in ("query_generation", "source_blind_evaluations"):
        prompt = builder.get_column_config(name).prompt
        assert "mark geometry" in prompt
        assert "visualization design" in prompt
        assert "physical objects" in prompt


def test_existing_source_relative_diagnostic_retains_precedence() -> None:
    assert query_rejection_reason("Which bar is longer in the chart?") == "source_relative_query"


def test_package_root_exposes_modality_neutral_api() -> None:
    assert retrieval_sdg.RetrievalSource.__name__ == "RetrievalSource"
    assert callable(retrieval_sdg.build_retrieval_pipeline)
    assert callable(retrieval_sdg.export_retrieval_data)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"generator_model_alias": ""}, "aliases must be non-empty"),
        ({"similarity_threshold": 1.1}, "similarity_threshold"),
        ({"start_index": 2, "end_index": 1}, "start_index"),
    ],
)
def test_pipeline_rejects_invalid_arguments(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "artifact_model_alias": "artifact",
        "generator_model_alias": "generator",
        "judge_model_alias": "judge",
        "embedding_model_alias": "embed",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_retrieval_pipeline(_mixed_seed(), **arguments)
