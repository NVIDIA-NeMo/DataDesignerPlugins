# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IndependentSourceAssessment obligations through public pipeline/export APIs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from data_designer.config import LLMStructuredColumnConfig
from data_designer.config.run_config import JinjaRenderingEngine
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.column_generators.utils.prompt_renderer import (
    PromptType,
    RecordBasedPromptRenderer,
    create_response_recipe,
)
from jinja2 import Template
from pydantic import ValidationError

from data_designer_retrieval_sdg.pipeline import build_retrieval_pipeline
from data_designer_retrieval_sdg.retrieval import GeneratedRetrievalRecord, export_retrieval_data
from data_designer_retrieval_sdg.stages import select_retrieval_queries


def _builder() -> object:
    return build_retrieval_pipeline(
        DataFrameSeedSource(df=pd.DataFrame([{"images": []}])),
        artifact_model_alias="artifact",
        generator_model_alias="generator",
        judge_model_alias="judge",
        embedding_model_alias="embed",
    )


def _record(source_id: str = "sample") -> dict:
    """Supply separately authored source reading and candidate verification fixtures."""
    criterion = {"score": 9, "justification": "Agrees with the independent reading."}
    return {
        "source_id": source_id,
        "language": "en",
        "retrieval_units": [
            {
                "unit_id": source_id,
                "document_id": source_id,
                "text": "Aster shipped 88 crates in November.",
                "images": [],
            }
        ],
        "deduplicated_qa_pairs": [
            {
                "question": f"How many crates did Aster ship in November for {source_id}?",
                "query_surface": "question",
                "answer": "88 crates",
                "evidence": "November shipments total 88 crates.",
                "positive_unit_ids": [source_id],
                "question_complexity": 3,
                "query_type": "contextual",
                "reasoning_type": "factual",
                "segment_ids": [1],
                "hop_count": 1,
                "hop_contexts": [],
            }
        ],
        "query_quality_evaluations": {
            "evaluations": [
                {
                    "candidate_index": 0,
                    "standalone_query": True,
                    "plausible_information_need": True,
                    "retrieval_discriminative": True,
                    "query_surface_correct": True,
                    "source_language_preserved": True,
                    "reason": "Scoped shipment lookup.",
                }
            ]
        },
        "source_assessments": {
            "assessments": [
                {
                    "candidate_index": 0,
                    "positive_unit_ids": [source_id],
                    "answerable": True,
                    "independent_answer": "The November shipment count is eighty-eight crates.",
                    "supporting_evidence": "The source states Aster's shipment count and month together.",
                    "uncertainty_reasons": [],
                    "positive_source_relevant": True,
                    "evidence_modality": "text_only",
                    "source_language_preserved": True,
                    "native_text_evidence": [{"unit_id": source_id, "quote": "Aster shipped 88 crates in November."}],
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
                    "overall": {"score": 9, "assessment": "Same count, entity and month."},
                    "improvements": "None.",
                    "answer_grounded": True,
                    "answer_resolves_query": True,
                    "unsupported_claims": [],
                    "positive_source_relevant": True,
                    "evidence_modality": "text_only",
                    "answer_not_revealed_by_query": True,
                    "source_language_preserved": True,
                    "native_text_evidence": [{"unit_id": source_id, "quote": "Aster shipped 88 crates in November."}],
                }
            ]
        },
    }


def _render_context() -> dict:
    row = _record()
    row["retrieval_units"][0]["text"] = "RAW_SOURCE_SENTINEL"
    row["images"] = ["https://example.invalid/IMAGE_SENTINEL.png"]
    row["document_artifacts"] = {"key_concepts": ["ARTIFACT_SENTINEL"]}
    row["text"] = "RAW_SOURCE_SENTINEL"
    row["sections_structured"] = [{"segment_id": 1, "text": "RAW_SOURCE_SENTINEL"}]
    row["retrieval_units"][0]["segment_id"] = 1
    pair = row["deduplicated_qa_pairs"][0]
    pair.update(
        answer="ANSWER_SENTINEL",
        evidence="EVIDENCE_SENTINEL",
        hop_contexts=[{"summary": "HOP_SENTINEL"}],
        reasoning_type="REASONING_SENTINEL",
    )
    row["deduplicated_queries"] = [{"question": pair["question"], "query_surface": pair["query_surface"]}]
    row["query_selection"] = {"selected": [{"query": row["deduplicated_queries"][0]}]}
    return row


def _native_request(column: LLMStructuredColumnConfig, row: dict) -> str:
    """Render actual model messages using Data Designer's secure response recipe."""
    renderer = RecordBasedPromptRenderer(
        create_response_recipe(column),
        jinja_rendering_engine=JinjaRenderingEngine.SECURE,
    )
    system = renderer.render(prompt_template=column.system_prompt, record=row, prompt_type=PromptType.SYSTEM_PROMPT)
    user = renderer.render(prompt_template=column.prompt, record=row, prompt_type=PromptType.USER_PROMPT)
    return (system or "") + "\n" + (user or "")


@pytest.mark.parametrize(
    ("column_name", "instruction"),
    [
        ("source_assessments", "Encode numeric answers as strings"),
        ("source_assessments", "Do not reproduce decorative dot leaders"),
        ("source_blind_evaluations", "Counterparty names, even several"),
        ("source_blind_evaluations", "False if ANY material subject, event, period, jurisdiction"),
        ("source_blind_evaluations", "Before deciding, identify the requested fact"),
        ("source_assessments", "Matching labels alone do not establish layout"),
    ],
)
def test_native_judge_schema_carries_precise_field_instructions(column_name: str, instruction: str) -> None:
    """Keep observed formatting and subject-scope pitfalls visible in the native schema."""
    rendered = _native_request(_builder().get_column_config(column_name), _render_context())
    assert instruction in rendered


def test_numeric_source_answers_are_not_silently_coerced() -> None:
    """A formatting failure must not be repaired into an approved source assessment."""
    row = _record()
    row["source_assessments"]["assessments"][0]["independent_answer"] = 88
    with pytest.raises(ValidationError, match="independent_answer"):
        GeneratedRetrievalRecord.model_validate(row)


@pytest.mark.parametrize(
    "name",
    [column.name for column in _builder().get_column_configs() if isinstance(column, LLMStructuredColumnConfig)],
)
def test_every_shared_llm_column_renders_with_native_secure_policy(name: str) -> None:
    """Exercise the host's filter allowlist and structured response instructions."""
    assert _native_request(_builder().get_column_config(name), _render_context())


def test_source_assessment_request_is_answer_blind() -> None:
    column = _builder().get_column_config("source_assessments")
    row = _render_context()
    rendered = _native_request(column, row)
    assert row["deduplicated_qa_pairs"][0]["question"] in rendered
    assert "RAW_SOURCE_SENTINEL" in rendered
    assert column.multi_modal_context[0].column_name == "images"
    assert set(column.required_columns) == {"deduplicated_qa_pairs", "retrieval_units", "language", "images"}
    for sentinel in ("ANSWER_SENTINEL", "EVIDENCE_SENTINEL", "HOP_SENTINEL", "ARTIFACT_SENTINEL", "REASONING_SENTINEL"):
        assert sentinel not in rendered


@pytest.mark.parametrize("name", ["answer_generation", "source_assessments"])
@pytest.mark.parametrize("image_count", [0, 1, 3])
def test_native_source_requests_identify_actual_image_attachments(name: str, image_count: int) -> None:
    """Bind attachment positions to units without rendering paths or encoded pixels as text."""
    row = _render_context()
    row["images"] = [f"https://example.invalid/PRIVATE_IMAGE_{index}.png" for index in range(image_count)]
    row["retrieval_units"][0]["images"] = row["images"][:1]
    row["retrieval_units"].extend(
        {"unit_id": f"other_{index}", "segment_id": index + 1, "text": "Other source.", "images": [image]}
        for index, image in reversed(list(enumerate(row["images"][1:], start=1)))
    )
    rendered = " ".join(_native_request(_builder().get_column_config(name), row).split())
    assert f"Attached image count: {image_count}" in rendered
    for index in range(image_count):
        unit_id = "sample" if index == 0 else f"other_{index}"
        assert f"Attachment {index + 1} source_unit_ids: {unit_id}" in rendered
    assert "PRIVATE_IMAGE" not in rendered
    assert "Text presence does not mean images are absent" in rendered
    if name == "source_assessments":
        assert "ANSWER_SENTINEL" not in rendered and "EVIDENCE_SENTINEL" not in rendered


@pytest.mark.parametrize("name", ["query_generation", "source_blind_evaluations"])
@pytest.mark.parametrize("language", ["en", "es"])
def test_native_queries_require_particular_subject_identity(name: str, language: str) -> None:
    """Deliver the missing-subject contract without adding source context to the blind judge."""
    row = _render_context()
    row["language"] = language
    rendered = _native_request(_builder().get_column_config(name), row)
    assert "particular company, study or transaction" in rendered
    assert "Generic roles" in rendered
    assert "Topic words and years alone" in rendered
    assert "Do not require a named entity for a genuinely general conceptual question" in rendered
    if name == "source_blind_evaluations":
        assert "retrieval_discriminative=false" in rendered
        assert "RAW_SOURCE_SENTINEL" not in rendered and "ANSWER_SENTINEL" not in rendered
    else:
        assert "return fewer/no queries" in rendered


def test_source_judge_cannot_supply_missing_query_identity_from_the_page() -> None:
    """Require independent checking of implicit subject scope, not just finding matching numbers."""
    rendered = _native_request(_builder().get_column_config("source_assessments"), _render_context())
    assert "Do not use the positive source to fill an identity missing from the query" in rendered
    assert "positive_source_relevant=false" in rendered


def test_final_comparison_request_cannot_reinterpret_source() -> None:
    column = _builder().get_column_config("qa_evaluations")
    rendered = _native_request(column, _render_context())
    assert "ANSWER_SENTINEL" in rendered and "EVIDENCE_SENTINEL" in rendered
    assert "eighty-eight crates" in rendered
    assert column.multi_modal_context is None
    assert set(column.required_columns) == {"deduplicated_qa_pairs", "source_assessments"}
    for sentinel in ("RAW_SOURCE_SENTINEL", "IMAGE_SENTINEL", "ARTIFACT_SENTINEL", "HOP_SENTINEL"):
        assert sentinel not in rendered


@pytest.mark.parametrize("language", ["en", "es"])
@pytest.mark.parametrize("images", [[], ["https://example.invalid/IMAGE_SENTINEL.png"]])
def test_native_query_requests_distinguish_recurring_scope_from_evergreen_needs(
    language: str, images: list[str]
) -> None:
    """Deliver the temporal contract through the real secure renderer without source leakage."""
    builder = _builder()
    row = _render_context()
    row.update(language=language, images=images)
    query = "Meridian ranking scores" if language == "en" else "puntuaciones de la clasificación Meridian"
    row["deduplicated_queries"][0]["question"] = query
    for name in ("query_generation", "source_blind_evaluations"):
        column = builder.get_column_config(name)
        rendered = _native_request(column, row)
        for obligation in ("edition, date or anchored window", "evergreen", "billing", "source language"):
            assert obligation in rendered
        assert language in rendered
        if name == "query_generation":
            assert "never invent a date" in rendered
            assert "RAW_SOURCE_SENTINEL" in rendered
        else:
            assert query in rendered
            assert "retrieval_discriminative=false" in rendered
            assert column.multi_modal_context is None
            assert set(column.required_columns) == {"language", "deduplicated_queries"}
            for sentinel in ("RAW_SOURCE_SENTINEL", "IMAGE_SENTINEL", "ANSWER_SENTINEL", "EVIDENCE_SENTINEL"):
                assert sentinel not in rendered


@pytest.mark.parametrize("language", ["en", "es"])
def test_native_answer_and_independent_reading_preserve_query_focus_and_categories(language: str) -> None:
    """Render multilingual invented requests while keeping the independent reading answer-blind."""
    builder = _builder()
    row = _render_context()
    row["language"] = language
    question = (
        "Which asset classes does Meridian hold?" if language == "en" else "¿Qué clases de activos tiene Meridian?"
    )
    row["deduplicated_queries"][0]["question"] = question
    row["deduplicated_qa_pairs"][0]["question"] = question
    for name in ("answer_generation", "source_assessments"):
        rendered = _native_request(builder.get_column_config(name), row)
        assert question in rendered
        assert "concise but complete" in rendered
        assert "geographic coverage is not an asset class" in rendered
        assert "adjacent" in rendered
        assert "every requested list member" in rendered
        assert "source language" in rendered
        assert "RAW_SOURCE_SENTINEL" in rendered
        assert "ANSWER_SENTINEL" not in rendered and "EVIDENCE_SENTINEL" not in rendered
    comparison = _native_request(builder.get_column_config("qa_evaluations"), row)
    assert "geographic coverage is not an asset class" in comparison
    assert "RAW_SOURCE_SENTINEL" not in comparison


@pytest.mark.parametrize("name", ["source_blind_evaluations", "source_assessments", "qa_evaluations"])
def test_native_judges_request_the_host_response_format(name: str) -> None:
    """Exercise the real secure renderer, not an alternative JSON parser."""
    rendered = _native_request(_builder().get_column_config(name), _render_context())
    assert (
        "Return exactly one JSON object matching the supplied response schema inside a single Markdown "
        "code block opened with ```json and closed with ```. Do not return bare JSON or any text outside that code block."
    ) in rendered


@pytest.mark.parametrize(
    ("language", "questions"),
    [
        (
            "en",
            [
                "Cedar service score definition arithmetic mean of uptime and responsiveness",
                "Cedar service score definition uptime responsiveness",
                "Why does Cedar use an arithmetic rather than geometric mean?",
                "Compare Cedar's arithmetic-mean score with its former weighted score.",
                "Find the original standard defining Cedar's score as the arithmetic mean of uptime and responsiveness.",
            ],
        ),
        (
            "es",
            [
                "puntuación de servicio Cedar definición media aritmética de disponibilidad y capacidad de respuesta",
                "puntuación de servicio Cedar definición disponibilidad capacidad de respuesta",
                "¿Por qué Cedar usa una media aritmética en vez de geométrica?",
                "Compara la puntuación de media aritmética de Cedar con su anterior puntuación ponderada.",
                "Busca la norma original que define Cedar como la media aritmética de disponibilidad y capacidad de respuesta.",
            ],
        ),
    ],
)
@pytest.mark.parametrize("images", [[], ["https://example.invalid/IMAGE_SENTINEL.png"]])
def test_native_keyword_contrast_preserves_explicit_unknowns_and_isolation(
    language: str, questions: list[str], images: list[str]
) -> None:
    """Deliver the contrast for multilingual inputs, without claiming model classifications."""
    builder = _builder()
    surfaces = ("keyword", "keyword", "question", "instruction", "instruction")
    for question, surface in zip(questions, surfaces):
        row = _render_context()
        row.update(language=language, images=images)
        row["deduplicated_queries"][0].update(question=question, query_surface=surface)
        row["deduplicated_qa_pairs"][0].update(question=question, query_surface=surface)
        before = copy.deepcopy(row)
        for name in ("query_generation", "source_blind_evaluations", "qa_evaluations"):
            rendered = _native_request(builder.get_column_config(name), row)
            assert '"Cedar service score definition arithmetic mean of uptime and responsiveness"' in rendered
            assert '"Cedar service score definition uptime responsiveness"' in rendered
            assert "Repeating the supplied relationship adds no requested information" in rendered
            assert "an explicitly requested new explanation, comparison or reference may remain unresolved" in rendered
            assert "Do not invent that request" in rendered
            if name != "query_generation":
                assert question in rendered
                assert "RAW_SOURCE_SENTINEL" not in rendered and "IMAGE_SENTINEL" not in rendered
            if name == "source_blind_evaluations":
                assert "ANSWER_SENTINEL" not in rendered and "EVIDENCE_SENTINEL" not in rendered
                assert "plausible_information_need=false" in rendered
            if name == "qa_evaluations":
                assert "form alone never establishes a valid unresolved need" in rendered
                assert "answer_not_revealed_by_query=false and answer_resolves_query=false" in rendered
        assert row == before


@pytest.mark.parametrize(
    ("language", "surface", "completed", "unresolved"),
    [
        (
            "en",
            "question",
            "What is Meridian's formula, defined as M=(A+B)/2?",
            "How is the Meridian index calculated from A and B?",
        ),
        (
            "en",
            "instruction",
            "Find Meridian's formula M=(A+B)/2.",
            "Find the formula for the Meridian index using A and B.",
        ),
        ("en", "keyword", "Meridian index formula M=(A+B)/2", "Meridian index calculation formula inputs A B"),
        (
            "es",
            "question",
            "¿Cuál es la fórmula de Meridian, definida como M=(A+B)/2?",
            "¿Cómo se calcula el índice Meridian a partir de A y B?",
        ),
        (
            "es",
            "instruction",
            "Busca la fórmula de Meridian M=(A+B)/2.",
            "Busca la fórmula del índice Meridian a partir de A y B.",
        ),
        ("es", "keyword", "índice Meridian fórmula M=(A+B)/2", "índice Meridian fórmula cálculo entradas A B"),
    ],
)
@pytest.mark.parametrize("images", [[], ["https://example.invalid/IMAGE_SENTINEL.png"]])
def test_native_requests_deliver_unresolved_need_contract_without_changing_queries(
    language: str, surface: str, completed: str, unresolved: str, images: list[str]
) -> None:
    """Check multilingual instruction delivery/isolation, not model classifications."""
    builder = _builder()
    for question in (completed, unresolved):
        row = _render_context()
        row.update(language=language, images=images)
        row["deduplicated_queries"][0].update(question=question, query_surface=surface)
        row["deduplicated_qa_pairs"][0].update(question=question, query_surface=surface)
        before = copy.deepcopy(row)
        generation = _native_request(builder.get_column_config("query_generation"), row)
        assert "omit the requested result, not its scope" in generation
        assert "keyword string can reveal a complete answer" in generation
        blind = _native_request(builder.get_column_config("source_blind_evaluations"), row)
        assert question in blind and surface in blind
        assert "plausible_information_need=false" in blind
        assert "already completed by facts or relationships it supplies" in blind
        assert "genuinely new explanation, comparison" in blind
        for sentinel in ("RAW_SOURCE_SENTINEL", "IMAGE_SENTINEL", "ANSWER_SENTINEL", "EVIDENCE_SENTINEL"):
            assert sentinel not in blind
        comparison = _native_request(builder.get_column_config("qa_evaluations"), row)
        assert question in comparison and "ANSWER_SENTINEL" in comparison
        assert "answer_not_revealed_by_query=false and answer_resolves_query=false" in comparison
        assert "Completing grammar, expanding abbreviations, translating" in comparison
        assert "unstated request" in comparison
        for sentinel in ("RAW_SOURCE_SENTINEL", "IMAGE_SENTINEL", "ARTIFACT_SENTINEL"):
            assert sentinel not in comparison
        assert row == before


@pytest.mark.parametrize(
    "criterion", ["plausible_information_need", "answer_not_revealed_by_query", "answer_resolves_query"]
)
def test_perfect_scores_cannot_override_unresolved_need_failures(tmp_path: Path, criterion: str) -> None:
    """Use supplied decisions to test gates, without claiming automatic leakage detection."""
    row = _record()
    pair = row["deduplicated_qa_pairs"][0]
    pair.update(question="Aster November shipment count 88 crates", query_surface="keyword")
    final = row["qa_evaluations"]["evaluations"][0]
    for key in ("relevance", "accuracy", "context_support", "clarity", "overall"):
        final[key]["score"] = 10
    if criterion == "plausible_information_need":
        row["query_quality_evaluations"]["evaluations"][0][criterion] = False
        query = {
            key: pair[key]
            for key in ("question", "query_surface", "question_complexity", "query_type", "reasoning_type")
        }
        selection = select_retrieval_queries(
            {"deduplicated_queries": [query], "source_blind_evaluations": row["query_quality_evaluations"]}
        )
        assert selection["query_selection"]["selected"] == []
        assert selection["query_selection"]["rejected"][0]["question"] == pair["question"]
    else:
        final[criterion] = False
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(_record("control")) + "\n")
    summary = export_retrieval_data(path, tmp_path / "bundle", dataset_id="invented")
    assert summary.accepted_candidate_count == summary.rejected_candidate_count == 1
    diagnostics = [
        json.loads(line) for line in (tmp_path / "bundle/candidate_diagnostics.jsonl").read_text().splitlines()
    ]
    rejected = next(item for item in diagnostics if item["source_id"] == "sample")
    reason = "query_quality_rejected:" if criterion == "plausible_information_need" else "grounding_rejected:"
    assert rejected["rejection_reason"].startswith(reason)


def test_empty_candidates_skip_both_quality_stages_without_inventing_assessments() -> None:
    builder = _builder()
    for name in ("source_assessments", "qa_evaluations"):
        column = builder.get_column_config(name)
        assert column.skip.when == "{{ deduplicated_qa_pairs | length == 0 }}"
        assert Template(column.skip.when).render(deduplicated_qa_pairs=[]) == "True"
    row = _record()
    row.update(
        deduplicated_qa_pairs=[],
        query_quality_evaluations={"evaluations": []},
        qa_evaluations={"evaluations": []},
        source_assessments=None,
    )
    assert GeneratedRetrievalRecord.model_validate(row).source_assessments is None


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("answerable", False, "source_assessment_rejected"),
        ("uncertainty_reasons", ["The material count is unreadable."], "source_assessment_rejected"),
        ("positive_source_relevant", False, "source_assessment_rejected"),
        ("source_language_preserved", False, "source_assessment_rejected"),
        ("independent_answer", " ", "source_assessment_rejected"),
        ("supporting_evidence", None, "source_assessment_rejected"),
        ("positive_unit_ids", ["different-positive"], "source_assessment_positive_units_mismatch"),
        (
            "native_text_evidence",
            [{"unit_id": "sample", "quote": "Aster shipped 98 crates in November."}],
            "native_text_evidence_not_in_supplied_text",
        ),
        (
            "native_text_evidence",
            [{"unit_id": "other", "quote": "Aster shipped 88 crates in November."}],
            "native_text_evidence_not_from_positive_unit",
        ),
        ("native_text_evidence", [], "source_assessment_judgement_mismatch"),
        ("evidence_modality", "image_grounded", "source_assessment_judgement_mismatch"),
    ],
)
def test_final_approval_cannot_rescue_bad_source_assessment(
    tmp_path: Path, field: str, value: object, reason: str
) -> None:
    row = _record()
    row["source_assessments"]["assessments"][0][field] = value
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(_record("control")) + "\n")
    summary = export_retrieval_data(path, tmp_path / "bundle", dataset_id="invented")
    assert summary.accepted_candidate_count == 1
    assert summary.rejected_candidate_count == 1
    diagnostics = [
        json.loads(line) for line in (tmp_path / "bundle/candidate_diagnostics.jsonl").read_text().splitlines()
    ]
    assert next(item for item in diagnostics if item["source_id"] == "sample")["rejection_reason"].startswith(reason)


@pytest.mark.parametrize("missing", [None, {"assessments": []}])
def test_historical_record_reads_but_cannot_receive_verified_export(tmp_path: Path, missing: object) -> None:
    row = _record()
    row["source_assessments"] = missing
    GeneratedRetrievalRecord.model_validate(row)
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(_record("control")) + "\n")
    summary = export_retrieval_data(path, tmp_path / "bundle", dataset_id="invented")
    assert summary.accepted_candidate_count == 1
    diagnostics = [
        json.loads(line) for line in (tmp_path / "bundle/candidate_diagnostics.jsonl").read_text().splitlines()
    ]
    assert (
        next(item for item in diagnostics if item["source_id"] == "sample")["rejection_reason"]
        == "missing_source_assessment"
    )


@pytest.mark.parametrize("indexes", [[0, 0], [1]])
def test_source_assessment_indexes_cannot_alias_other_candidates(indexes: list[int]) -> None:
    row = _record()
    assessment = row["source_assessments"]["assessments"][0]
    row["source_assessments"]["assessments"] = [{**assessment, "candidate_index": index} for index in indexes]
    with pytest.raises(ValidationError, match="source assessment"):
        GeneratedRetrievalRecord.model_validate(row)


def test_assessments_survive_portable_export_without_rewriting_candidates(tmp_path: Path) -> None:
    row = _record()
    snapshot = copy.deepcopy(row)
    row["source_assessments"] = json.dumps(row["source_assessments"])
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    summary = export_retrieval_data(
        path, tmp_path / "bundle", dataset_id="invented", generator_model="generator", judge_model="judge"
    )
    assert summary.accepted_candidate_count == 1
    audit = json.loads((tmp_path / "bundle/source_records.jsonl").read_text())
    assert audit["source_assessments"] == snapshot["source_assessments"]
    assert audit["deduplicated_qa_pairs"] == snapshot["deduplicated_qa_pairs"]
    manifest = json.loads(Path(summary.run_manifest_path).read_text())
    assert manifest["quality_contract"] == "independent_source_assessment_v1"
    assert manifest["models"]["judge"] == "judge"


def test_all_historical_candidates_rejected_report_missing_assessment_reason(tmp_path: Path) -> None:
    row = _record()
    del row["source_assessments"]
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="missing_source_assessment"):
        export_retrieval_data(path, tmp_path / "bundle", dataset_id="invented")
    assert not (tmp_path / "bundle").exists()


def test_shuffled_assessments_are_bound_by_index_not_list_order(tmp_path: Path) -> None:
    row = _record()
    second = copy.deepcopy(row["deduplicated_qa_pairs"][0])
    second["question"] = "Find the Aster November shipment count for the regional report."
    row["deduplicated_qa_pairs"].append(second)
    row["query_quality_evaluations"]["evaluations"].append(
        {
            **row["query_quality_evaluations"]["evaluations"][0],
            "candidate_index": 1,
        }
    )
    row["qa_evaluations"]["evaluations"].append(
        {
            **row["qa_evaluations"]["evaluations"][0],
            "candidate_index": 1,
        }
    )
    assessments = row["source_assessments"]["assessments"]
    assessments.insert(0, {**assessments[0], "candidate_index": 1, "answerable": False})
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    summary = export_retrieval_data(path, tmp_path / "bundle", dataset_id="invented")
    assert summary.accepted_candidate_count == 1
    diagnostics = [
        json.loads(line) for line in (tmp_path / "bundle/candidate_diagnostics.jsonl").read_text().splitlines()
    ]
    assert [item["accepted"] for item in diagnostics] == [True, False]
    assert diagnostics[1]["rejection_reason"] == "source_assessment_rejected"
