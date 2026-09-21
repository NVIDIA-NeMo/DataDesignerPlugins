# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic EA workflow contracts on invented collections, with no benchmark fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.testing import make_stub_completion_response
from PIL import Image
from pydantic import ValidationError

from data_designer_retrieval_sdg.multimodal import ModelSettings, MultimodalSDGConfig, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.export import export_multimodal_bundle
from data_designer_retrieval_sdg.multimodal.inference import DataDesignerInference, Request
from data_designer_retrieval_sdg.multimodal.models import (
    ContextSummary,
    Localization,
    QueryBatch,
    QueryMetadata,
    QuerySlot,
    RelevanceJudgment,
    SelfSufficiencyJudgment,
    SummaryJudgment,
    Support,
)
from data_designer_retrieval_sdg.multimodal.storage import fingerprint, write_json
from data_designer_retrieval_sdg.multimodal.workflow import generate_candidates, load_contexts, localization_reasons
from data_designer_retrieval_sdg.retrieval.source_file import load_retrieval_sources


def fixture_config(tmp_path: Path) -> MultimodalSDGConfig:
    """Use arbitrary nonnumeric page IDs and two unrelated fictional collections."""
    Image.new("RGB", (24, 24), "white").save(tmp_path / "page.png")
    sources = [
        {
            "unit_id": f"unit-{i}",
            "document_id": "shared-manual",
            "text": f"Valve {i} opens at 20 bar.",
            "images": ["page.png"],
            "language": "en",
        }
        for i in range(12)
    ]
    sources.append(
        {
            "unit_id": "unselected",
            "document_id": "different-manual",
            "text": "An unrelated appendix.",
            "images": ["page.png"],
        }
    )
    source_file = tmp_path / "sources.jsonl"
    source_file.write_text("".join(json.dumps(row) + "\n" for row in sources))
    context_file = tmp_path / "contexts.jsonl"
    context_file.write_text(
        "".join(
            json.dumps({"context_id": f"context-{i}", "unit_ids": [f"unit-{i}", "unit-11"]}) + "\n" for i in range(10)
        )
    )
    model = ModelSettings(model="operator/vlm", endpoint="https://example.invalid/v1", credential_env="TEST_SDG_KEY")
    return MultimodalSDGConfig(
        sources_file=source_file,
        contexts_file=context_file,
        output_dir=tmp_path / "run",
        dataset_id="service-manuals",
        generator=model,
        judge=model,
    )


class ScriptedInference:
    """Deterministic stage outputs; never used to claim hosted model quality."""

    def __init__(self, root=None, config=None):
        self.requests = []

    def generate(self, requests):
        self.requests.extend(requests)
        return [scripted_response(request) for request in requests]


def scripted_response(request):
    if request.schema is ContextSummary:
        return ContextSummary(summary="Valve operating specifications", visual_evidence="")
    if request.schema is SummaryJudgment:
        return SummaryJudgment(
            **{
                key: {"grade": 5, "explanation": "Substantive"}
                for key in (
                    "information_richness",
                    "persona_relevance",
                    "query_generation_potential",
                    "conceptual_clarity",
                )
            }
        )
    if request.schema is QueryBatch:
        payload = json.loads(request.text.splitlines()[-1])
        subject = payload["sources"][0]["unit_id"]
        return QueryBatch(
            queries=[
                QuerySlot(slot=0, query=f"What pressure opens {subject}?", evidence_modality="text"),
                QuerySlot(slot=1, query=None, evidence_modality="none"),
                QuerySlot(slot=2, query=None, evidence_modality="none"),
            ]
        )
    if request.schema is SelfSufficiencyJudgment:
        return SelfSufficiencyJudgment(self_sufficiency=5, reasoning="Standalone")
    if request.schema is QueryMetadata:
        return QueryMetadata(
            has_answer=False,
            actual_query_type="numerical",
            actual_query_format="question",
            classification_reasoning="Standalone",
        )
    if request.schema is RelevanceJudgment:
        return RelevanceJudgment(relevance=5, reasoning="Source states a pressure")
    payload = json.loads(request.text.splitlines()[-1])
    return Localization(
        supports=[
            Support(
                unit_id=s["unit_id"],
                grade=1,
                modality="text",
                quote=s["text"],
                visual_evidence="",
                contribution="Partial specification",
            )
            for s in payload["sources"]
        ]
    )


def generated_fixture(tmp_path):
    config = fixture_config(tmp_path)
    sources = load_retrieval_sources(config.sources_file)
    contexts = load_contexts(config, sources)
    inference = ScriptedInference()
    candidates, outcomes = generate_candidates(config, sources, contexts, inference)
    return config, sources, candidates, outcomes, inference


def test_complete_generic_workflow_and_verified_resume(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
    handoff = run_multimodal_sdg(config)
    assert handoff.is_file()
    report = json.loads((handoff.parent / "report.json").read_text())
    assert report["corpus_units"] == 13
    assert report["accepted_query_ids"] == 10
    assert report["positive_annotations"] == 20
    assert report["query_groups"] == 10  # One shared document/context unit does not join queries.
    before = handoff.read_bytes()
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    assert handoff.read_bytes() == before
    with pytest.raises(FileExistsError):
        run_multimodal_sdg(config)
    with pytest.raises(ValueError, match="Artifact changed"):
        run_multimodal_sdg(config.model_copy(update={"resume": True, "seed": 3}))
    (handoff.parent / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))


def test_judges_use_reference_source_visibility(tmp_path):
    _, _, candidates, outcomes, inference = generated_fixture(tmp_path)
    assert len(outcomes) == 10 and len(outcomes[0]["slots"]["queries"]) == 3
    for request in inference.requests:
        if request.schema is QueryMetadata:
            assert not request.images and "opens at 20 bar" not in request.text
        elif request.schema is SelfSufficiencyJudgment:
            assert not request.images and "opens at 20 bar" in request.text
            assert "Valve operating specifications" not in request.text
        elif request.schema is SummaryJudgment:
            assert not request.images and "Valve operating specifications" in request.text
        else:
            assert len(request.images) == 2
            assert "image_unit_ids" in request.text
    assert all(c.accepted and c.query_judgment.observed_type == "numerical" for c in candidates)


@pytest.mark.parametrize("changes", [{"self_sufficiency": 3}, {"has_answer": True}])
def test_recorded_query_rejections_cannot_be_exported_as_passes(tmp_path, changes):
    config, sources, candidates, outcomes, _ = generated_fixture(tmp_path)
    candidate = candidates[0]
    candidates[0] = candidate.model_copy(update={"query_judgment": candidate.query_judgment.model_copy(update=changes)})
    with pytest.raises(ValueError, match="Recorded acceptance"):
        export_multimodal_bundle(tmp_path / "bundle", sources, candidates, outcomes, config)


@pytest.mark.parametrize(
    "support_changes,reason",
    [
        ({"unit_id": "not-in-context"}, "invalid_localized_identity"),
        ({"quote": ""}, "missing_text_evidence"),
        ({"modality": "image", "visual_evidence": ""}, "missing_visual_evidence"),
    ],
)
def test_localization_rejects_invalid_evidence(tmp_path, support_changes, reason):
    config, sources, candidates, _, _ = generated_fixture(tmp_path)
    context = load_contexts(config, sources)[0]
    support = candidates[0].localization.supports[0].model_copy(update=support_changes)
    assert localization_reasons(Localization(supports=[support]), context, {s.unit_id: s for s in sources}) == [reason]


def test_image_evidence_needs_no_fabricated_text_quote(tmp_path):
    config, sources, _, _, _ = generated_fixture(tmp_path)
    support = Support(
        unit_id="unit-0",
        grade=2,
        modality="image",
        quote="",
        visual_evidence="A visible pressure curve",
        contribution="Complete answer",
    )
    assert not localization_reasons(
        Localization(supports=[support]), load_contexts(config, sources)[0], {s.unit_id: s for s in sources}
    )


def test_all_positives_grades_full_corpus_and_query_groups_survive_export(tmp_path):
    config, sources, candidates, outcomes, _ = generated_fixture(tmp_path)
    candidates[1] = candidates[1].model_copy(update={"query": candidates[0].query.upper()})
    root = tmp_path / "bundle"
    export_multimodal_bundle(root, sources, candidates, outcomes, config)
    splits = json.loads((root / "split_manifest.json").read_text())["query_assignments"]
    assert splits["image"] == {}  # White images do not contain the localized text evidence.
    for view in ("text", "image_and_text"):
        assignments = splits[view]
        assert assignments[candidates[0].query_id] == assignments[candidates[1].query_id]
        train = json.loads((root / "views" / view / "train.json").read_text())["data"]
        assert train and all(len(row["pos_doc"]) == 2 for row in train)
        assert all(p["score"] == 1 for row in train for p in row["pos_doc"])
        assert all(not row["neg_doc"] for row in train)
        corpus = [
            json.loads(line) for line in (root / "synthetic_eval" / view / "corpus.jsonl").read_text().splitlines()
        ]
        assert len(corpus) == 13 and any(row["_id"] == "unselected" for row in corpus)
    with pytest.raises(FileExistsError):
        export_multimodal_bundle(root, sources, candidates, outcomes, config)


@pytest.mark.parametrize("modality", ["text", "image"])
def test_full_nemotron_consumer_contract(tmp_path, modality):
    consumer = pytest.importorskip("nemotron.recipes.embed.sdg_manifest")
    config, sources, candidates, outcomes, _ = generated_fixture(tmp_path)
    if modality == "image":
        for candidate in candidates:
            for support in candidate.localization.supports:
                support.modality = "image"
                support.quote = ""
                support.visual_evidence = "White page"
    handoff = export_multimodal_bundle(tmp_path / "bundle", sources, candidates, outcomes, config)
    for view in (modality, "image_and_text"):
        assert consumer.resolve_portable_training_input(handoff, view).is_file()
        assert consumer.resolve_portable_evaluation_input(handoff, view)[0].is_dir()


def test_request_cache_binds_schema_model_pixels_and_detects_edits(tmp_path):
    config = fixture_config(tmp_path)
    inference = DataDesignerInference(tmp_path / "inference", config)
    request = Request("Read evidence", ContextSummary, "generator", (tmp_path / "page.png",))
    first = inference.request_key(request)
    Image.new("RGB", (24, 24), "black").save(tmp_path / "page.png")
    assert inference.request_key(request) != first
    assert inference.request_key(
        Request("Changed", ContextSummary, "generator", request.images)
    ) != inference.request_key(request)
    key = inference.request_key(request)
    write_json(
        inference.root / "responses" / f"{key}.json",
        {"request_id": key, "response": {"summary": "x", "visual_evidence": ""}, "response_sha256": "bad"},
    )
    with pytest.raises(ValueError, match="integrity"):
        inference.cached(request)


@pytest.mark.parametrize("with_images", [False, True])
def test_native_dd_batch_accepts_bare_json_without_core_patch(tmp_path, monkeypatch, with_images):
    config = fixture_config(tmp_path)
    monkeypatch.setenv("TEST_SDG_KEY", "test-placeholder")
    completion = AsyncMock(
        return_value=make_stub_completion_response(
            content='{"summary":"Actual structured response","visual_evidence":""}'
        )
    )
    monkeypatch.setattr(ModelFacade, "acompletion", completion)
    inference = DataDesignerInference(tmp_path / "inference", config)
    request = Request("Summarize", ContextSummary, "generator", (tmp_path / "page.png",) if with_images else ())
    result = inference.generate([request, request])
    assert result[0].summary == "Actual structured response"
    assert result[0] == result[1] and completion.await_count == 1
    assert inference.generate([request])[0] == result[0]
    assert completion.await_count == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"concurrency": 0},
        {"batch_size": 0},
        {"max_units_per_context": 0},
        {"relevance_threshold": 6},
        {"unknown": "ignored?"},
    ],
)
def test_invalid_configuration_fails_before_model_calls(tmp_path, changes):
    payload = fixture_config(tmp_path).model_dump()
    with pytest.raises(ValidationError):
        MultimodalSDGConfig.model_validate({**payload, **changes})


def test_context_bounds_partition_without_truncation(tmp_path):
    config = fixture_config(tmp_path)
    contexts = load_contexts(
        config.model_copy(update={"max_units_per_context": 1}), load_retrieval_sources(config.sources_file)
    )
    assert len(contexts) == 20 and all(len(c.unit_ids) == 1 for c in contexts)
    assert len({c.context_id for c in contexts}) == 20
    assert sum(c.unit_ids == ["unit-11"] for c in contexts) == 10


def test_fingerprints_preserve_unicode_and_mapping_order():
    assert fingerprint({"b": "日本語", "a": 2}) == fingerprint({"a": 2, "b": "日本語"})


class RejectingInference(ScriptedInference):
    def generate(self, requests):
        values = super().generate(requests)
        return [
            value.model_copy(update={"has_answer": True}) if isinstance(value, QueryMetadata) else value
            for value in values
        ]


def test_failed_gates_preserve_rejections_and_never_localize(tmp_path):
    config = fixture_config(tmp_path)
    sources = load_retrieval_sources(config.sources_file)
    inference = RejectingInference()
    candidates, outcomes = generate_candidates(config, sources, load_contexts(config, sources), inference)
    assert len(candidates) == 10 and len(outcomes) == 10
    assert all(
        not candidate.accepted and candidate.rejection_reasons == ["query_contains_answer"] for candidate in candidates
    )
    assert not any(request.schema is Localization for request in inference.requests)
    with pytest.raises(ValueError, match="No candidates"):
        export_multimodal_bundle(tmp_path / "bundle", sources, candidates, outcomes, config)
    assert not (tmp_path / "bundle").exists()


class MissingInference(DataDesignerInference):
    def run_batch(self, items):
        self.calls = getattr(self, "calls", 0) + 1


def test_missing_response_has_finite_attempts_without_fabrication(tmp_path):
    inference = MissingInference(tmp_path / "inference", fixture_config(tmp_path))
    with pytest.raises(RuntimeError, match="Missing structured"):
        inference.generate([Request("one", ContextSummary, "judge")])
    assert inference.calls == 3


def test_run_lock_prevents_duplicate_workers(tmp_path):
    from filelock import FileLock, Timeout

    config = fixture_config(tmp_path)
    with FileLock(str(config.output_dir) + ".lock"):
        with pytest.raises(Timeout):
            run_multimodal_sdg(config)
    assert not config.output_dir.exists()


def test_partial_bundle_is_preserved_not_overwritten(tmp_path):
    config, sources, candidates, outcomes, _ = generated_fixture(tmp_path)
    root = tmp_path / "partial"
    root.mkdir()
    sentinel = root / "partial-evidence.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        export_multimodal_bundle(root, sources, candidates, outcomes, config)
    assert sentinel.read_text() == "preserve"


def test_source_content_drift_blocks_resume_before_inference(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
    run_multimodal_sdg(config)
    config.sources_file.write_text(config.sources_file.read_text().replace("20 bar", "25 bar"))
    with pytest.raises(ValueError, match="Artifact changed"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))


def test_image_only_sources_and_mixed_views_do_not_drop_positives(tmp_path):
    config, sources, candidates, outcomes, _ = generated_fixture(tmp_path)
    sources[0] = sources[0].model_copy(update={"text": ""})
    support = (
        candidates[0]
        .localization.supports[0]
        .model_copy(update={"modality": "image", "quote": "", "visual_evidence": "Visible chart"})
    )
    candidates[0] = candidates[0].model_copy(
        update={"localization": Localization(supports=[support, candidates[0].localization.supports[1]])}
    )
    handoff = export_multimodal_bundle(tmp_path / "bundle", sources, candidates, outcomes, config)
    report = json.loads((handoff.parent / "report.json").read_text())
    assert report["views"]["image"]["accepted_queries"] == 0
    assert report["views"]["text"]["accepted_queries"] == 9
    assert report["views"]["image_and_text"]["accepted_queries"] == 10
    assert report["positive_annotations"] == 20


@pytest.mark.parametrize("modality", ["text", "image", "text_and_image"])
def test_single_modality_views_require_all_localized_evidence(tmp_path, modality):
    config, sources, candidates, outcomes, _ = generated_fixture(tmp_path)
    for candidate in candidates:
        for support in candidate.localization.supports:
            support.modality = modality
            support.visual_evidence = "White page" if modality != "text" else ""
            if modality == "image":
                support.quote = ""
    handoff = export_multimodal_bundle(tmp_path / "bundle", sources, candidates, outcomes, config)
    report = json.loads((handoff.parent / "report.json").read_text())
    assignments = json.loads((handoff.parent / "split_manifest.json").read_text())["query_assignments"]
    for view in ("text", "image", "image_and_text"):
        expected = len(candidates) if view in (modality, "image_and_text") else 0
        assert report["views"][view]["accepted_queries"] == expected
        assert len(assignments[view]) == expected
        rows = json.loads((handoff.parent / "views" / view / "train.json").read_text())["data"]
        assert bool(rows) == bool(expected)
        assert all(len(row["pos_doc"]) == 2 for row in rows)


def test_inconsistent_slot_is_retried_without_rewriting_response(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    monkeypatch.setenv("TEST_SDG_KEY", "test-placeholder")
    completion = AsyncMock(
        side_effect=[
            make_stub_completion_response(content='{"queries":[{"slot":0,"query":"","evidence_modality":"none"}]}'),
            make_stub_completion_response(content='{"queries":[{"slot":0,"query":null,"evidence_modality":"none"}]}'),
        ]
    )
    monkeypatch.setattr(ModelFacade, "acompletion", completion)
    inference = DataDesignerInference(tmp_path / "inference", config)
    result = inference.generate([Request("Generate or abstain explicitly", QueryBatch, "generator")])
    assert result[0].queries[0].query is None
    assert completion.await_count == 2
    assert inference.generate([Request("Generate or abstain explicitly", QueryBatch, "generator")]) == result
    assert completion.await_count == 2
