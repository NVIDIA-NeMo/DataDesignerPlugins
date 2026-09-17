# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public regression tests for semantic temporal and event-scope judging."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from data_designer.config import DataDesignerConfigBuilder, LLMStructuredColumnConfig
from data_designer.config.run_config import JinjaRenderingEngine
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.column_generators.utils.prompt_renderer import (
    PromptType,
    RecordBasedPromptRenderer,
    create_response_recipe,
)

from data_designer_retrieval_sdg.pipeline import build_retrieval_pipeline
from data_designer_retrieval_sdg.retrieval import export_retrieval_data
from data_designer_retrieval_sdg.stages import select_retrieval_queries


def _builder() -> DataDesignerConfigBuilder:
    """Build the public shared retrieval pipeline used in production."""
    return build_retrieval_pipeline(
        DataFrameSeedSource(df=pd.DataFrame([{"images": []}])),
        artifact_model_alias="artifact",
        generator_model_alias="generator",
        judge_model_alias="judge",
        embedding_model_alias="embed",
    )


def _render_row() -> dict:
    """Return a fictional row sufficient for every temporal-scope prompt."""
    question = "Find the recent Solace developer conference panel."
    unit = {
        "unit_id": "solace#segment-1",
        "document_id": "solace",
        "segment_id": 1,
        "text": "This event introduced the Solace caching feature.",
        "images": [],
    }
    return {
        "source_id": "solace",
        "language": "en",
        "images": [],
        "text": unit["text"],
        "retrieval_units": [unit],
        "sections_structured": ["Segment 1: This event introduced the Solace caching feature."],
        "document_artifacts": {"key_concepts": []},
        "deduplicated_queries": [{"question": question, "query_surface": "instruction"}],
        "query_selection": {
            "selected": [
                {
                    "query": {
                        "question": question,
                        "query_surface": "instruction",
                        "question_complexity": 2,
                        "query_type": "contextual",
                        "reasoning_type": "temporal",
                    }
                }
            ]
        },
        "deduplicated_qa_pairs": [
            {
                "question": question,
                "answer": "The event introduced caching.",
                "evidence": "The source says this event introduced caching.",
                "query_surface": "instruction",
                "positive_unit_ids": [unit["unit_id"]],
                "question_complexity": 2,
                "query_type": "contextual",
                "reasoning_type": "temporal",
                "segment_ids": [1],
                "hop_count": 1,
                "hop_contexts": [],
            }
        ],
        "source_assessments": {
            "assessments": [
                {
                    "candidate_index": 0,
                    "positive_unit_ids": [unit["unit_id"]],
                    "answerable": False,
                    "independent_answer": None,
                    "supporting_evidence": None,
                    "uncertainty_reasons": ["The event and date are unidentified."],
                    "positive_source_relevant": False,
                    "evidence_modality": "text_only",
                    "source_language_preserved": True,
                    "native_text_evidence": [],
                }
            ]
        },
    }


def _render(column: LLMStructuredColumnConfig, row: dict) -> str:
    """Render the actual secure system and user requests."""
    renderer = RecordBasedPromptRenderer(
        create_response_recipe(column),
        jinja_rendering_engine=JinjaRenderingEngine.SECURE,
    )
    system = renderer.render(prompt_template=column.system_prompt, record=row, prompt_type=PromptType.SYSTEM_PROMPT)
    user = renderer.render(prompt_template=column.prompt, record=row, prompt_type=PromptType.USER_PROMPT)
    return (system or "") + "\n" + (user or "")


@pytest.mark.parametrize(
    "name",
    ["query_generation", "source_blind_evaluations", "answer_generation", "source_assessments"],
)
def test_rendered_native_requests_share_temporal_event_scope_contract(name: str) -> None:
    """Require the common contract through the public native prompt renderer."""
    rendered = _render(_builder().get_column_config(name), _render_row())
    for obligation in (
        '"recent", "upcoming", "currently", "latest"',
        '"past 50 years" does not identify a period',
        "as-of date, edition",
        "named product, organization or recurring conference alone is not a time anchor",
        "never assume the current date",
        "Do not demand dates for timeless definitions",
        "trigger condition, not a calendar date",
        '"The crisis", "this event"',
        "an original entity date and a later transfer",
        "date-to-event relationship from co-occurrence",
    ):
        assert obligation in rendered


def test_rendered_judges_require_their_specific_fail_closed_fields() -> None:
    """Keep query, source, and final resolution failures assigned to existing fields."""
    builder = _builder()
    row = _render_row()
    blind = " ".join(_render(builder.get_column_config("source_blind_evaluations"), row).split())
    source = " ".join(_render(builder.get_column_config("source_assessments"), row).split())
    final = " ".join(_render(builder.get_column_config("qa_evaluations"), row).split())
    assert "retrieval_discriminative=false for unresolved temporal or event scope" in blind
    assert "named product do not override that failure" in blind
    assert "answerable=false and independent_answer=null" in source
    assert "uncertainty_reasons" in source
    assert "generated artifacts or world knowledge" in source
    assert "Inspect within-cell line" in source
    assert "Missing date semantics cannot be supplied by the query" in source
    assert "answer_resolves_query" in final
    assert 'unidentified "crisis/event" is not a resolved explanation' in final
    assert "even if both readings repeat the same ambiguous wording" in final


@pytest.mark.parametrize(
    "name",
    ["query_generation", "source_blind_evaluations", "answer_generation", "source_assessments"],
)
def test_rendered_requests_do_not_invent_event_or_jurisdiction(name: str) -> None:
    """Verify contract wiring, not hosted-model compliance with the contract."""
    rendered = " ".join(_render(_builder().get_column_config(name), _render_row()).split())
    assert '"Northstar date"' in rendered
    assert "Do not silently choose one" in rendered
    assert "language does not identify its country" in rendered
    assert "requires no redundant country or event label" in rendered
    assert 'a date plus "four yes values"' in rendered
    assert "never infer missing column semantics" in rendered
    assert "technique itself is the requested information" in rendered
    assert "positives do not establish completeness" in rendered
    assert "study will supply its period after retrieval" in rendered


def test_rendered_judges_require_relationship_support_and_category_agreement() -> None:
    """Require semantic support instructions beyond deterministic quote membership."""
    builder = _builder()
    row = _render_row()
    source = " ".join(_render(builder.get_column_config("source_assessments"), row).split())
    final = " ".join(_render(builder.get_column_config("qa_evaluations"), row).split())
    assert "Mentally hide the image" in source
    assert "Do not assume reading order preserves layout" in source
    assert "one reading excludes an entire category and the other only a subtype" in final
    assert "Reject rather than choosing which reading is probably right" in final


def _query(text: str, surface: str = "question", reasoning: str = "temporal") -> dict:
    """Build one fictional retrieval-query fixture."""
    return {
        "question": text,
        "query_surface": surface,
        "question_complexity": 2,
        "query_type": "contextual",
        "reasoning_type": reasoning,
    }


def _judgement(index: int, *, discriminative: bool = True) -> dict:
    """Build an explicit semantic query judgement fixture."""
    return {
        "candidate_index": index,
        "standalone_query": True,
        "plausible_information_need": True,
        "retrieval_discriminative": discriminative,
        "query_surface_correct": True,
        "source_language_preserved": True,
        "reason": "Anchored or evergreen." if discriminative else "Temporal or event scope is unresolved.",
    }


@pytest.mark.parametrize(
    ("question", "surface"),
    [
        ("Find the recent Solace developer conference panel.", "instruction"),
        ("upcoming Atlas acceleration feature", "keyword"),
        ("Harbor bird decline over the past 50 years", "keyword"),
        ("Why did this crisis accelerate Northwind automation?", "question"),
        ("Northstar date", "keyword"),
        ("date de Northstar", "keyword"),
        ("Find the regulated household electricity price increase in 2024.", "instruction"),
    ],
)
def test_query_gate_rejects_explicit_scope_failure_without_rewriting(question: str, surface: str) -> None:
    """Exercise existing semantic gate fields while preserving the original query."""
    row = {
        "deduplicated_queries": [_query(question, surface)],
        "source_blind_evaluations": {"evaluations": [_judgement(0, discriminative=False)]},
    }
    before = copy.deepcopy(row)
    result = select_retrieval_queries(row)["query_selection"]
    assert row == before
    assert result["selected"] == []
    assert result["rejected"][0]["question"] == question
    assert result["rejected"][0]["rejection_reason"].startswith("query_quality_rejected:")


@pytest.mark.parametrize(
    ("question", "surface", "reasoning"),
    [
        ("Find the Solace 2024 developer conference panel on storage.", "instruction", "temporal"),
        ("Atlas acceleration feature announced at its 2024 launch", "keyword", "temporal"),
        ("Harbor bird decline from 1970 to 2020", "keyword", "temporal"),
        ("When does Atlas issue a lane-change notification?", "question", "procedural"),
        ("What storage temperature does Atlas coolant require?", "question", "factual"),
        ("Northstar incorporation date", "keyword", "factual"),
        ("date de constitution de Northstar", "keyword", "factual"),
        ("How do regulated electricity prices work?", "question", "factual"),
    ],
)
def test_query_gate_preserves_supported_anchored_and_evergreen_queries(
    question: str, surface: str, reasoning: str
) -> None:
    """Show that existing gates accept positive semantic judgements without broad date regexes."""
    row = {
        "deduplicated_queries": [_query(question, surface, reasoning)],
        "source_blind_evaluations": {"evaluations": [_judgement(0)]},
    }
    result = select_retrieval_queries(row)["query_selection"]
    assert result["rejected"] == []
    assert result["selected"][0]["query"]["question"] == question


def _export_record(source_id: str, question: str) -> dict:
    """Build a complete fictional accepted record for public export-gate tests."""
    unit_id = f"{source_id}#segment-1"
    quote = "The Solace 2024 launch introduced cache acceleration."
    criterion = {"score": 9, "justification": "The scoped reading agrees."}
    return {
        "source_id": source_id,
        "language": "en",
        "retrieval_units": [
            {"unit_id": unit_id, "document_id": source_id, "text": quote, "images": [], "segment_id": 1}
        ],
        "deduplicated_qa_pairs": [
            {
                **_query(question),
                "answer": "Cache acceleration.",
                "evidence": quote,
                "positive_unit_ids": [unit_id],
                "segment_ids": [1],
                "hop_count": 1,
                "hop_contexts": [],
            }
        ],
        "query_quality_evaluations": {"evaluations": [_judgement(0)]},
        "source_assessments": {
            "assessments": [
                {
                    "candidate_index": 0,
                    "positive_unit_ids": [unit_id],
                    "answerable": True,
                    "independent_answer": "Cache acceleration.",
                    "supporting_evidence": quote,
                    "uncertainty_reasons": [],
                    "positive_source_relevant": True,
                    "evidence_modality": "text_only",
                    "source_language_preserved": True,
                    "native_text_evidence": [{"unit_id": unit_id, "quote": quote}],
                }
            ]
        },
        "qa_evaluations": {
            "evaluations": [
                {
                    "candidate_index": 0,
                    "relevance": criterion,
                    "accuracy": criterion,
                    "context_support": criterion,
                    "clarity": criterion,
                    "overall": {"score": 9, "assessment": "Scoped and supported."},
                    "improvements": "None.",
                    "answer_grounded": True,
                    "answer_resolves_query": True,
                    "unsupported_claims": [],
                    "positive_source_relevant": True,
                    "evidence_modality": "text_only",
                    "answer_not_revealed_by_query": True,
                    "source_language_preserved": True,
                    "native_text_evidence": [{"unit_id": unit_id, "quote": quote}],
                }
            ]
        },
        "generation_diagnostics": [],
    }


@pytest.mark.parametrize("failed_stage", ["source", "final"])
def test_export_gate_rejects_supplied_temporal_failure_and_preserves_query(tmp_path: Path, failed_stage: str) -> None:
    """Verify existing source/final booleans remain authoritative after prompt changes."""
    question = "Why did this event accelerate Solace automation?"
    failing = _export_record("ambiguous", question)
    if failed_stage == "source":
        assessment = failing["source_assessments"]["assessments"][0]
        assessment.update(
            answerable=False,
            independent_answer=None,
            supporting_evidence=None,
            uncertainty_reasons=["The event is not identified."],
            positive_source_relevant=False,
            native_text_evidence=[],
        )
    else:
        failing["qa_evaluations"]["evaluations"][0]["answer_resolves_query"] = False
    control = _export_record("control", "What did the Solace 2024 launch introduce?")
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(failing) + "\n" + json.dumps(control) + "\n", encoding="utf-8")

    summary = export_retrieval_data(input_path, tmp_path / "bundle", dataset_id="fictional")
    diagnostics = [
        json.loads(line) for line in (tmp_path / "bundle/candidate_diagnostics.jsonl").read_text().splitlines()
    ]
    rejected = next(item for item in diagnostics if item["source_id"] == "ambiguous")
    assert summary.accepted_candidate_count == summary.rejected_candidate_count == 1
    assert rejected["question"] == question
    expected = "source_assessment_rejected" if failed_stage == "source" else "grounding_rejected:"
    assert rejected["rejection_reason"].startswith(expected)
