# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the unified retrieval recipe exporter."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from data_designer_retrieval_sdg.prompts import SOURCE_ASSESSMENT_USER_PROMPT
from data_designer_retrieval_sdg.retrieval import (
    SplitRatios,
    assign_document_splits,
    export_retrieval_data,
)

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_export_preserves_pre_grounding_rejections_and_source_profiles(tmp_path: Path) -> None:
    unit = _unit("unit-1", "doc-1", text="Drug X storage is 20-25 C.")
    record = _record(
        "source-1",
        [unit],
        _candidate("What storage temperature does Drug X require?", "20-25 C", "question", ["unit-1"]),
        independent_answer="Drug X's storage range is 20 to 25 C.",
    )
    record["generation_diagnostics"] = [
        {
            "candidate_index": 1,
            "question": "What does the chart show?",
            "query_surface": "question",
            "rejection_reason": "source_relative_query",
        }
    ]
    record["document_artifacts"] = {
        "key_concepts": None,
        "relationships": None,
        "retrieval_profile": {
            "text_retrieval_useful": True,
            "visual_retrieval_useful": False,
            "visual_content_types": [],
            "suggested_query_operations": ["storage lookup"],
            "reason": "Named drug storage guidance",
        },
    }
    empty = {
        **record,
        "source_id": "source-2",
        "retrieval_units": [_unit("unit-2", "doc-2", text="Agenda")],
        "deduplicated_qa_pairs": [],
        "query_quality_evaluations": {"evaluations": []},
        "qa_evaluations": None,
        "source_assessments": None,
        "generation_diagnostics": [],
        "document_artifacts": None,
    }
    source = tmp_path / "generated.jsonl"
    source.write_text(json.dumps(record) + "\n" + json.dumps(empty) + "\n")
    summary = export_retrieval_data(source, tmp_path / "bundle", dataset_id="test")
    manifest = json.loads(Path(summary.run_manifest_path).read_text())
    assert summary.accepted_candidate_count == 1
    assert summary.rejected_candidate_count == 1
    assert manifest["pre_grounding_rejected_candidate_count"] == 1
    assert manifest["rejection_reason_counts"] == {"source_relative_query": 1}
    assert manifest["source_suitability_counts"]["text_useful"] == 1
    assert manifest["source_suitability_counts"]["unknown"] == 1
    assert manifest["schema_version"] == 2
    diagnostics = [
        json.loads(line) for line in (tmp_path / "bundle/candidate_diagnostics.jsonl").read_text().splitlines()
    ]
    assert diagnostics[-1]["stage"] == "pre_grounding"
    assert "answer" not in diagnostics[-1]
    records = _read_jsonl(tmp_path / "bundle/source_records.jsonl")
    assert records[0]["document_artifacts"]["key_concepts"] == []
    assert records[0]["document_artifacts"]["relationships"] == []


def test_export_does_not_default_required_profile_fields(tmp_path: Path) -> None:
    record = _record(
        "source-1",
        [_unit("unit-1", "doc-1", text="Drug X storage is 20-25 C.")],
        _candidate("What storage temperature does Drug X require?", "20-25 C", "question", ["unit-1"]),
        independent_answer="Store Drug X between 20 and 25 C.",
    )
    record["document_artifacts"] = {
        "retrieval_profile": {
            "text_retrieval_useful": True,
            "visual_retrieval_useful": False,
            "visual_content_types": None,
            "suggested_query_operations": [],
            "reason": "Text storage guidance",
        }
    }
    source = tmp_path / "generated.jsonl"
    _write_rows(source, [record])
    with pytest.raises(ValidationError, match="visual_content_types"):
        export_retrieval_data(source, tmp_path / "bundle", dataset_id="test")


def _write_image(path: Path) -> None:
    path.write_bytes(_TINY_PNG)


def _candidate(
    question: str,
    answer: str,
    surface: str,
    unit_ids: list[str],
) -> dict[str, object]:
    return {
        "question": question,
        "answer": answer,
        "evidence": "Evidence from the declared positive unit.",
        "query_surface": surface,
        "positive_unit_ids": unit_ids,
        "question_complexity": 4,
        "query_type": "contextual",
        "reasoning_type": "factual",
        "segment_ids": [1],
        "hop_count": 1,
        "hop_contexts": [],
    }


def _query_evaluation(index: int) -> dict[str, object]:
    return {
        "candidate_index": index,
        "standalone_query": True,
        "plausible_information_need": True,
        "retrieval_discriminative": True,
        "query_surface_correct": True,
        "source_language_preserved": True,
        "reason": "Valid standalone retrieval need.",
    }


def _grounding_evaluation(index: int, evidence_modality: str) -> dict[str, object]:
    criterion = {"score": 9, "justification": "Supported."}
    return {
        "candidate_index": index,
        "relevance": criterion,
        "accuracy": criterion,
        "context_support": criterion,
        "clarity": criterion,
        "overall": {"score": 9, "assessment": "Grounded."},
        "improvements": "None.",
        "answer_grounded": True,
        "answer_resolves_query": True,
        "unsupported_claims": [],
        "positive_source_relevant": True,
        "evidence_modality": evidence_modality,
        "answer_not_revealed_by_query": True,
        "source_language_preserved": True,
    }


def _unit(
    unit_id: str,
    document_id: str,
    *,
    text: str = "",
    image: Path | None = None,
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "document_id": document_id,
        "text": text,
        "images": [] if image is None else [str(image)],
        "source_uri": f"{document_id}.pdf",
        "page_number": 1 if image is not None else None,
        "segment_id": 1,
    }


def _record(
    source_id: str,
    units: list[dict[str, object]],
    candidate: dict[str, object],
    *,
    independent_answer: str,
    evidence_modality: str = "text_only",
) -> dict[str, object]:
    """Join explicitly authored mock source readings, never copy candidate answers."""
    return {
        "source_id": source_id,
        "retrieval_units": units,
        "language": "en",
        "deduplicated_qa_pairs": [candidate],
        "query_quality_evaluations": {"evaluations": [_query_evaluation(0)]},
        "qa_evaluations": {"evaluations": [_grounding_evaluation(0, evidence_modality)]},
        "source_assessments": {
            "assessments": [
                {
                    "candidate_index": 0,
                    "positive_unit_ids": candidate["positive_unit_ids"],
                    "answerable": True,
                    "independent_answer": independent_answer,
                    "supporting_evidence": "The mock source reader independently reports: " + independent_answer,
                    "uncertainty_reasons": [],
                    "positive_source_relevant": True,
                    "evidence_modality": evidence_modality,
                    "source_language_preserved": True,
                    "native_text_evidence": [],
                }
            ]
        },
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.parametrize(
    ("quotes", "reason"),
    [
        ([], "text_only_modality_requires_native_text_evidence_in_every_positive"),
        (
            [{"unit_id": "visual-unit", "quote": "The total combines two equally weighted scores."}],
            "native_text_evidence_not_in_supplied_text",
        ),
        ([{"unit_id": "other-unit", "quote": "Composite Score"}], "native_text_evidence_not_from_positive_unit"),
        ([{"unit_id": "visual-unit", "quote": "   \n"}], "native_text_evidence_not_in_supplied_text"),
    ],
)
def test_export_quarantines_unverified_native_text_support(tmp_path: Path, quotes: list, reason: str) -> None:
    """A native title cannot authenticate generated claims or pixel-only text."""
    image = tmp_path / "visual.png"
    _write_image(image)
    unsafe = _record(
        "visual-source",
        [_unit("visual-unit", "visual-doc", text="Composite Score", image=image)],
        _candidate(
            "How is the Aster composite score calculated?",
            "The total combines two equally weighted scores.",
            "question",
            ["visual-unit"],
        ),
        independent_answer="Two component scores receive equal weight in the composite.",
    )
    unsafe["qa_evaluations"]["evaluations"][0]["native_text_evidence"] = quotes
    safe = _record(
        "text-source",
        [_unit("text-unit", "text-doc", text="Aster storage is 5 C.")],
        _candidate("What temperature is recommended for Aster storage?", "5 C", "question", ["text-unit"]),
        independent_answer="Aster's storage temperature is five degrees Celsius.",
    )
    source = tmp_path / "generated.jsonl"
    _write_rows(source, [unsafe, safe])
    summary = export_retrieval_data(source, tmp_path / "bundle", dataset_id="test")
    assert summary.accepted_candidate_count == 1
    assert summary.rejected_candidate_count == 1
    diagnostics = _read_jsonl(tmp_path / "bundle/candidate_diagnostics.jsonl")
    rejected = next(item for item in diagnostics if item["source_id"] == "visual-source")
    assert rejected["rejection_reason"] == reason
    assert rejected["evidence_modality"] == "text_only"


@pytest.mark.parametrize(
    ("modality", "include_image"), [("text_only", True), ("image_grounded", True), ("text_only", False)]
)
@pytest.mark.parametrize("quote", ["Dose is 88 mg.", "dose is 98 mg."])
def test_export_checks_every_supplied_quote_without_case_or_numeric_forgiveness(
    tmp_path: Path,
    modality: str,
    include_image: bool,
    quote: str,
) -> None:
    image = tmp_path / "dose.png"
    _write_image(image)
    record = _record(
        "dose-source",
        [_unit("dose-unit", "dose-doc", text="Dose is 98 mg.", image=image if include_image else None)],
        _candidate("What is the specified dose for Aster?", "98 mg", "question", ["dose-unit"]),
        independent_answer="The specified Aster dose is ninety-eight milligrams.",
        evidence_modality=modality,
    )
    record["qa_evaluations"]["evaluations"][0]["native_text_evidence"] = [{"unit_id": "dose-unit", "quote": quote}]
    source = tmp_path / "generated.jsonl"
    _write_rows(source, [record])
    with pytest.raises(ValueError, match="no candidates passed"):
        export_retrieval_data(source, tmp_path / "bundle", dataset_id="test")


def test_export_preserves_matching_multilingual_quotes_and_requires_all_positives(tmp_path: Path) -> None:
    image = tmp_path / "storage.png"
    _write_image(image)
    units = [
        _unit("one", "doc-one", text="Aster\nse conserva\ta 5 °C.", image=image),
        _unit("two", "doc-two", text="Aster se conserva a 5 °C."),
    ]
    record = _record(
        "source",
        units,
        _candidate("¿A qué temperatura se conserva Aster?", "5 °C", "question", ["one", "two"]),
        independent_answer="Aster debe conservarse a cinco grados Celsius.",
    )
    quotes = [{"unit_id": "one", "quote": "Aster se conserva a 5 °C."}]
    record["qa_evaluations"]["evaluations"][0]["native_text_evidence"] = quotes
    source = tmp_path / "partial.jsonl"
    _write_rows(source, [record])
    with pytest.raises(ValueError, match="no candidates passed"):
        export_retrieval_data(source, tmp_path / "partial-bundle", dataset_id="test")
    quotes.append({"unit_id": "two", "quote": "Aster se conserva a 5 °C."})
    record["source_assessments"]["assessments"][0]["native_text_evidence"] = [
        {"unit_id": "one", "quote": "Aster se conserva a 5 °C."},
        {"unit_id": "two", "quote": "Aster se conserva a 5 °C."},
    ]
    complete = tmp_path / "complete.jsonl"
    _write_rows(complete, [record])
    summary = export_retrieval_data(complete, tmp_path / "complete-bundle", dataset_id="test")
    assert summary.accepted_candidate_count == 1
    audit = _read_jsonl(tmp_path / "complete-bundle/source_records.jsonl")
    assert audit[0]["qa_evaluations"]["evaluations"][0]["native_text_evidence"] == quotes


def test_grounding_prompt_requires_native_quotes_without_pixel_substitution() -> None:
    assert "native_text_evidence" in SOURCE_ASSESSMENT_USER_PROMPT
    assert "every material answer claim" in SOURCE_ASSESSMENT_USER_PROMPT
    assert "without looking at the images" in SOURCE_ASSESSMENT_USER_PROMPT


def test_grounding_prompt_requires_independent_numeric_reading_and_uncertainty_rejection() -> None:
    """Check the instruction contract, not probabilistic model accuracy."""
    assert "Independently read every needed number, date and table cell" in SOURCE_ASSESSMENT_USER_PROMPT
    assert "row/column alignment, labels and units" in SOURCE_ASSESSMENT_USER_PROMPT
    assert "uncertain\nor unreadable" in SOURCE_ASSESSMENT_USER_PROMPT
    assert "answerable=false" in SOURCE_ASSESSMENT_USER_PROMPT


def test_document_split_is_deterministic() -> None:
    ratios = SplitRatios(train=0.5, validation=0.25, evaluation=0.25)
    first = assign_document_splits(["a", "b", "c", "d"], ratios=ratios, seed=7)
    second = assign_document_splits(reversed(["a", "b", "c", "d"]), ratios=ratios, seed=7)

    assert first == second
    assert set(first.values()) == {"train", "validation", "evaluation"}


def test_export_mixed_inputs_writes_only_truthful_views(tmp_path: Path) -> None:
    image_only = tmp_path / "image-only.png"
    combined = tmp_path / "combined.png"
    _write_image(image_only)
    _write_image(combined)
    rows = [
        _record(
            "source-text",
            [_unit("unit-text", "doc-text", text="The adult dose is 20 mg.")],
            _candidate(
                "What dose is recommended for adults with condition X?",
                "20 mg",
                "question",
                ["unit-text"],
            ),
            independent_answer="Adults with condition X receive twenty milligrams.",
        ),
        _record(
            "source-image",
            [_unit("unit-image", "doc-image", image=image_only)],
            _candidate(
                "Find the annual product Y trend between 2022 and 2024.",
                "The value decreases.",
                "instruction",
                ["unit-image"],
            ),
            evidence_modality="image_grounded",
            independent_answer="Annual product Y values decline from 2022 to 2024.",
        ),
        _record(
            "source-combined",
            [_unit("unit-combined", "doc-combined", text="Phase 3 results.", image=combined)],
            _candidate(
                "condition Z phase 3 subgroup response comparison",
                "The active arm was higher.",
                "keyword",
                ["unit-combined"],
            ),
            evidence_modality="image_grounded",
            independent_answer="The active arm has the higher phase 3 response.",
        ),
    ]
    generated = tmp_path / "generated.jsonl"
    _write_rows(generated, rows)
    output = tmp_path / "bundle"

    summary = export_retrieval_data(
        generated,
        output,
        dataset_id="fixture/domain",
        generator_model="nvidia/qwen/qwen3.8-27b",
        judge_model="nvidia/nemotron-3-ultra-550b-a55b",
    )

    assert summary.source_record_count == 3
    assert summary.retrieval_unit_count == 3
    assert summary.accepted_candidate_count == 3
    assert summary.query_surface_counts == {"question": 1, "instruction": 1, "keyword": 1}
    assert summary.evidence_modality_counts == {"text_only": 1, "image_grounded": 2}
    assert summary.warnings == []

    units = _read_jsonl(output / "retrieval_units.jsonl")
    for row in units:
        for image_path in row["images"]:
            assert (output / str(image_path)).is_file()
    audit_records = _read_jsonl(output / "source_records.jsonl")
    assert audit_records[0]["deduplicated_qa_pairs"]
    assert audit_records[0]["query_quality_evaluations"]["evaluations"]
    assert audit_records[0]["qa_evaluations"]["evaluations"]

    expected_schemas = {
        "text": ["id", "text"],
        "image": ["image_filename", "image"],
        "image_and_text": ["docid", "text", "image"],
    }
    expected_query_counts = {"text": 1, "image": 2, "image_and_text": 2}
    for view, names in expected_schemas.items():
        corpus_files = list((output / "views" / view / "corpus").glob("*/part-00000.parquet"))
        assert all(pq.read_schema(path).names == names for path in corpus_files)
        training_count = 0
        for split in ("train", "validation"):
            payload = json.loads((output / "views" / view / f"{split}.json").read_text(encoding="utf-8"))
            training_count += len(payload["data"])
        eval_queries = _read_jsonl(output / "synthetic_eval" / view / "queries.jsonl")
        assert training_count + len(eval_queries) == expected_query_counts[view]
        eval_corpus = {str(row["_id"]) for row in _read_jsonl(output / "synthetic_eval" / view / "corpus.jsonl")}
        with (output / "synthetic_eval" / view / "qrels" / "test.tsv").open(
            encoding="utf-8",
            newline="",
        ) as file:
            qrels = list(csv.DictReader(file, delimiter="\t"))
        assert all(row["corpus-id"] in eval_corpus for row in qrels)

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = output / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_text_only_input_does_not_trigger_visual_gate(tmp_path: Path) -> None:
    row = _record(
        "source-text",
        [_unit("unit-text", "doc-text", text="The adult dose is 20 mg.")],
        _candidate(
            "What dose is recommended for adults with condition X?",
            "20 mg",
            "question",
            ["unit-text"],
        ),
        independent_answer="The adult dose is twenty milligrams.",
    )
    generated = tmp_path / "generated.jsonl"
    _write_rows(generated, [row])

    summary = export_retrieval_data(
        generated,
        tmp_path / "bundle",
        dataset_id="fixture/text",
        strict_visual=True,
        generator_model="generator",
        judge_model="judge",
        answer_model="answers",
    )
    assert not any(warning.startswith("NO_VISUAL_CANDIDATES") for warning in summary.warnings)
    assert json.loads(Path(summary.run_manifest_path).read_text())["models"]["answer"] == "answers"


def test_image_input_without_visual_candidate_warns_or_fails(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    _write_image(image)
    row = _record(
        "source-image",
        [_unit("unit-image", "doc-image", text="The adult dose is 20 mg.", image=image)],
        _candidate(
            "What dose is recommended for adults with condition X?",
            "20 mg",
            "question",
            ["unit-image"],
        ),
        independent_answer="Adults receive twenty milligrams.",
    )
    row["qa_evaluations"]["evaluations"][0]["native_text_evidence"] = [
        {"unit_id": "unit-image", "quote": "The adult dose is 20 mg."}
    ]
    row["source_assessments"]["assessments"][0]["native_text_evidence"] = [
        {"unit_id": "unit-image", "quote": "The adult dose is 20 mg."}
    ]
    generated = tmp_path / "generated.jsonl"
    _write_rows(generated, [row])

    summary = export_retrieval_data(generated, tmp_path / "warning", dataset_id="fixture/image")
    assert any(warning.startswith("NO_VISUAL_CANDIDATES") for warning in summary.warnings)
    with pytest.raises(ValueError, match="strict visual coverage"):
        export_retrieval_data(
            generated,
            tmp_path / "strict",
            dataset_id="fixture/image",
            strict_visual=True,
        )


def test_export_rejects_modality_inconsistent_with_positive_content(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    _write_image(image)
    rows = [
        _record(
            "source-image",
            [_unit("unit-image", "doc-image", image=image)],
            _candidate(
                "Find annual product X dosage guidance for adults.",
                "20 mg.",
                "instruction",
                ["unit-image"],
            ),
            independent_answer="Product X adult dosage is twenty milligrams.",
        ),
        _record(
            "source-text",
            [_unit("unit-text", "doc-text", text="Adults receive 10 mg.")],
            _candidate(
                "What dose is recommended for adults receiving product Y?",
                "10 mg.",
                "question",
                ["unit-text"],
            ),
            evidence_modality="image_grounded",
            independent_answer="Adults receive ten milligrams of product Y.",
        ),
        _record(
            "source-valid",
            [_unit("unit-valid", "doc-valid", text="Adults receive 5 mg.")],
            _candidate(
                "What dose is recommended for adults receiving product Z?",
                "5 mg.",
                "question",
                ["unit-valid"],
            ),
            independent_answer="Product Z's adult dose is five milligrams.",
        ),
    ]
    generated = tmp_path / "generated.jsonl"
    _write_rows(generated, rows)
    output = tmp_path / "bundle"

    summary = export_retrieval_data(generated, output, dataset_id="fixture/domain")

    assert summary.accepted_candidate_count == 1
    diagnostics = _read_jsonl(output / "candidate_diagnostics.jsonl")
    reasons = {str(row["source_id"]): row["rejection_reason"] for row in diagnostics}
    assert reasons["source-image"] == "text_only_modality_requires_text_in_every_positive"
    assert reasons["source-text"] == "image_grounded_modality_requires_images_in_every_positive"


def test_unsupported_claim_list_cannot_silently_pass(tmp_path: Path) -> None:
    rejected = _record(
        "source-rejected",
        [_unit("unit-rejected", "doc-rejected", text="Adults receive 10 mg.")],
        _candidate(
            "What dose is recommended for adults receiving product X?",
            "10 mg improves survival.",
            "question",
            ["unit-rejected"],
        ),
        independent_answer="The source supplies ten milligrams as the dose, not a survival claim.",
    )
    rejected["qa_evaluations"]["evaluations"][0]["unsupported_claims"] = ["10 mg improves survival"]
    accepted = _record(
        "source-accepted",
        [_unit("unit-accepted", "doc-accepted", text="Adults receive 5 mg.")],
        _candidate(
            "What dose is recommended for adults receiving product Y?",
            "5 mg.",
            "question",
            ["unit-accepted"],
        ),
        independent_answer="Adults receive five milligrams of product Y.",
    )
    generated = tmp_path / "generated.jsonl"
    _write_rows(generated, [rejected, accepted])
    output = tmp_path / "bundle"

    summary = export_retrieval_data(generated, output, dataset_id="fixture/domain")

    assert summary.accepted_candidate_count == 1
    diagnostics = _read_jsonl(output / "candidate_diagnostics.jsonl")
    rejected_diagnostic = next(row for row in diagnostics if row["source_id"] == "source-rejected")
    assert str(rejected_diagnostic["rejection_reason"]).startswith("grounding_rejected:")


def test_cross_unit_query_collision_is_quarantined(tmp_path: Path) -> None:
    query = "What dose is recommended for adults with condition X?"
    rows = [
        _record(
            "source-a",
            [_unit("unit-a", "doc-a", text="20 mg.")],
            _candidate(query, "20 mg", "question", ["unit-a"]),
            independent_answer="The dosage is twenty milligrams.",
        ),
        _record(
            "source-b",
            [_unit("unit-b", "doc-b", text="20 mg.")],
            _candidate(query, "20 mg", "question", ["unit-b"]),
            independent_answer="Twenty milligrams is the dosage.",
        ),
        _record(
            "source-c",
            [_unit("unit-c", "doc-c", text="A different source.")],
            _candidate(
                "Find comparative guidance for products C and D.",
                "Product C was higher.",
                "instruction",
                ["unit-c"],
            ),
            independent_answer="Product C exceeds product D.",
        ),
    ]
    generated = tmp_path / "generated.jsonl"
    _write_rows(generated, rows)

    summary = export_retrieval_data(generated, tmp_path / "bundle", dataset_id="fixture/domain")

    assert summary.accepted_candidate_count == 1
    diagnostics = _read_jsonl(tmp_path / "bundle" / "candidate_diagnostics.jsonl")
    assert sum(row["rejection_reason"] == "cross_unit_query_collision" for row in diagnostics) == 2


def test_linked_documents_cannot_cross_splits(tmp_path: Path) -> None:
    row = _record(
        "source-bundle",
        [
            _unit("unit-a", "doc-a", text="Evidence A."),
            _unit("unit-b", "doc-b", text="Evidence B."),
        ],
        _candidate(
            "How do policies A and B differ for adult applicants?",
            "They differ in eligibility.",
            "question",
            ["unit-a", "unit-b"],
        ),
        independent_answer="Eligibility is the distinction between policies A and B.",
    )
    other = _record(
        "source-other",
        [_unit("unit-c", "doc-c", text="Evidence C.")],
        _candidate(
            "Find policy C eligibility guidance for adult applicants.",
            "Adults qualify.",
            "instruction",
            ["unit-c"],
        ),
        independent_answer="Policy C permits adult applicants.",
    )
    generated = tmp_path / "generated.jsonl"
    _write_rows(generated, [row, other])
    output = tmp_path / "bundle"

    export_retrieval_data(
        generated,
        output,
        dataset_id="fixture/domain",
        ratios=SplitRatios(train=0.5, validation=0.0, evaluation=0.5),
    )
    split_manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    assert split_manifest["assignments"]["doc-a"] == split_manifest["assignments"]["doc-b"]
