# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Section planning uses generic sources and preserves the existing portable contracts."""

import json
import random

import pytest
from test_multimodal_sdg import ScriptedInference, fixture_config, scripted_response

from data_designer_retrieval_sdg.multimodal.combinations import (
    cluster_combinations,
    sample_members,
    semantic_combinations,
)
from data_designer_retrieval_sdg.multimodal.models import (
    Description,
    GenerationContext,
    Localization,
    QueryBatch,
    SummaryJudgment,
    VisualDescription,
)
from data_designer_retrieval_sdg.multimodal.planning import select_summaries
from data_designer_retrieval_sdg.multimodal.sampling import sample_instructions
from data_designer_retrieval_sdg.multimodal.summary_stages import (
    bounded_enriched_contexts,
    enriched_text,
    section_contexts,
)
from data_designer_retrieval_sdg.multimodal.workflow import (
    generate_candidates,
    generation_context_rows,
    load_contexts,
    normalize_localization,
    run_multimodal_sdg,
)
from data_designer_retrieval_sdg.retrieval.source_file import load_retrieval_sources


class SectionInference(ScriptedInference):
    """Record all planning stages with distinguishable generated visual enrichment."""

    def generate(self, requests):
        self.requests.extend(requests)
        responses = []
        for request in requests:
            if request.schema is VisualDescription:
                responses.append(VisualDescription(description="Generated chart description", has_visual_content=True))
            elif request.schema is Description:
                responses.append(Description(description="Equipment documentation"))
            else:
                responses.append(scripted_response(request))
        return responses


def test_sections_export_keeps_original_generic_sources(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"contexts_file": None, "context_strategy": "sections"})
    before = config.sources_file.read_bytes()
    inference = SectionInference()
    monkeypatch.setattr(
        "data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", lambda *args: inference
    )
    handoff = run_multimodal_sdg(config)
    assert config.sources_file.read_bytes() == before
    assert handoff.is_file()
    outcomes = json.loads((config.output_dir / "context_outcomes.json").read_text())
    assert [len(row["context"]["unit_ids"]) for row in outcomes] == [5, 5, 2, 1]
    assert all(set(row["summary_judgment"]) == set(SummaryJudgment.model_fields) for row in outcomes)
    assert all("page_number" not in request.text for request in inference.requests)
    # Generated enrichment is planning-only; query and localization evidence remain original.
    grounded = [request for request in inference.requests if issubclass(request.schema, (QueryBatch, Localization))]
    assert grounded and all("Generated chart description" not in request.text for request in grounded)
    assert (config.output_dir / "planning/visual_descriptions.json").is_file()
    assert not (config.output_dir / "planning/corpus_descriptions.json").exists()
    assert not any("Describe the corpus" in request.text for request in inference.requests)


def test_explicit_contexts_are_preserved_and_semantic_pairs_bounded(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"context_strategy": "sections"})
    sources = load_retrieval_sources(config.sources_file)
    monkeypatch.setattr(
        "data_designer_retrieval_sdg.multimodal.summary_stages.semantic_combinations", lambda rows, config: [(0, 1)]
    )
    candidates, outcomes = generate_candidates(config, sources, load_contexts(config, sources), SectionInference())
    assert outcomes[0]["context"]["context_id"] == "context-0"
    combined = [row for row in outcomes if row.get("combined")]
    assert len(combined) == 1
    assert combined[0]["context"]["unit_ids"] == ["unit-0", "unit-11", "unit-1"]
    assert all(set(c.source_unit_ids) <= {s.unit_id for s in sources} for c in candidates)


def test_enrichment_is_included_in_character_bounds(tmp_path):
    config = fixture_config(tmp_path).model_copy(update={"max_context_chars": 900})
    sources = load_retrieval_sources(config.sources_file)
    visual = {s.unit_id: {"description": "x" * 300, "has_visual_content": True} for s in sources}
    contexts = section_contexts(sources, config)
    chunks = bounded_enriched_contexts(contexts, sources, visual, config)
    assert len(chunks) > len(contexts)
    by_id = {s.unit_id: s for s in sources}
    assert all(len(enriched_text(c, by_id, visual)) <= config.max_context_chars for c in chunks)
    assert {key for c in chunks for key in c.unit_ids} == set(by_id)
    visual[sources[0].unit_id]["description"] = "x" * 1000
    with pytest.raises(ValueError, match="exceeds max_context_chars"):
        bounded_enriched_contexts(contexts, sources, visual, config)


def test_near_dedup_before_actual_generation_bounding(tmp_path):
    config = fixture_config(tmp_path).model_copy(update={"summary_near_duplicate_threshold": 0.95})
    sources = load_retrieval_sources(config.sources_file)
    text = "Stable operating specifications describe valve service without changing the rated pressure."
    rows = [
        {
            "context": GenerationContext(
                context_id=str(size), unit_ids=[f"unit-{i}" for i in range(size)]
            ).model_dump(),
            "summary": {"summary": text, "visual_evidence": ""},
            "summary_judgment": None,
            "document_ids": ["shared-manual"],
        }
        for size in (10, 9)
    ]
    selected = select_summaries(rows, config)
    assert len(selected) == 1
    bounded = generation_context_rows(selected, rows, sources, config)
    assert [len(row["context"]["unit_ids"]) for row in bounded] == [8, 2]
    assert {key for row in bounded for key in row["context"]["unit_ids"]} == {f"unit-{i}" for i in range(10)}
    assert rows[1]["selection_reason"] == "duplicate_summary"


@pytest.mark.parametrize("criterion", list(SummaryJudgment.model_fields))
def test_each_summary_grade_is_required(tmp_path, criterion):
    config = fixture_config(tmp_path)
    grades = {key: {"grade": 5, "explanation": "Good"} for key in SummaryJudgment.model_fields}
    grades[criterion]["grade"] = 3
    row = {"context": {"context_id": "a", "unit_ids": ["unit-0"], "language": "source"}, "summary_judgment": grades}
    assert select_summaries([row], config) == []
    assert row["selection_reason"] == "summary_quality"


def test_semantic_small_languages_skip_model_loading(tmp_path):
    config = fixture_config(tmp_path).model_copy(update={"combination_iterations": 20})
    rows = [{"context": {"language": language}} for language in ("en", "de") for _ in range(6)]
    assert semantic_combinations(rows, config) == []
    assert cluster_combinations([], [], 20) == []
    assert sample_members({"doc": [0]}, 2, random.Random(0)) is None
    assert sample_members({"doc": [0, 1, 2]}, 3, random.Random(0)) == (0, 1, 2)


def test_weighted_sampler_is_local_and_uses_fixed_metadata_vocabulary():
    state = random.getstate()
    first = sample_instructions(42, "arbitrary-unit/α")
    assert first == sample_instructions(42, "arbitrary-unit/α")
    assert random.getstate() == state
    assert [item.name for item in first] == ["text", "figure", "table"]
    assert all(item.query_type and item.format and item.answerability for item in first)
    assert any(first != sample_instructions(42, str(i)) for i in range(10))


def test_localization_filters_unknown_and_keeps_highest_grade():
    context = GenerationContext(context_id="opaque", unit_ids=["arbitrary/α"])
    support = {
        "unit_id": "arbitrary/α",
        "grade": 1,
        "modality": "text",
        "quote": "source",
        "visual_evidence": "",
        "contribution": "Relevant",
    }
    result = normalize_localization(
        Localization(supports=[support, {**support, "grade": 2}, {**support, "unit_id": "unknown"}]), context
    )
    assert len(result.supports) == 1 and result.supports[0].grade == 2


def test_nonvisual_description_is_not_used_for_planning(tmp_path):
    config = fixture_config(tmp_path)
    sources = load_retrieval_sources(config.sources_file)
    context = GenerationContext(context_id="opaque", unit_ids=["unit-0", "unit-1"])
    visual = {
        "unit-0": {"has_visual_content": False, "description": "Ordinary prose only"},
        "unit-1": {"has_visual_content": True, "description": "A substantive diagram"},
    }
    text = enriched_text(context, {s.unit_id: s for s in sources}, visual)
    assert "Ordinary prose only" not in text
    assert "A substantive diagram" in text
    assert "Valve 0 opens at 20 bar." in text


@pytest.mark.parametrize("query,modality", [("", "none"), ("   ", "text"), ("A query", "none"), (None, "text")])
def test_invalid_slot_fails_response_validation(query, modality):
    from pydantic import ValidationError

    from data_designer_retrieval_sdg.multimodal.models import QueryBatch

    with pytest.raises(ValidationError):
        QueryBatch.model_validate({"queries": [{"slot": 0, "query": query, "evidence_modality": modality}]})


def test_image_histogram_counts_only_bounded_generation_including_abstentions(tmp_path):
    from data_designer_retrieval_sdg.multimodal.reporting import generation_report

    config = fixture_config(tmp_path)
    sources = load_retrieval_sources(config.sources_file)
    rows = [
        {
            "context": GenerationContext(context_id="large", unit_ids=[f"unit-{i}" for i in range(10)]).model_dump(),
            "summary": {"summary": "Specifications", "visual_evidence": ""},
            "slots": {"queries": []},
        }
    ]
    bounded = generation_context_rows(rows.copy(), rows, sources, config)
    for row in bounded:
        row["slots"] = {"queries": [{"slot": 0, "query": None, "evidence_modality": "none"}]}
    report = generation_report(sources, [], rows)
    assert report["images_per_context"] == {"8": 1, "2": 1}
    assert report["coverage"]["generated"]["units"] == 0
    assert report["coverage"]["planned"]["units"] == 10
