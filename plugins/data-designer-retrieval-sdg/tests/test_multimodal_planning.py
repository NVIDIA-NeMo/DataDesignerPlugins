# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feature parity checks on invented sources, without benchmark-specific repairs."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from test_multimodal_sdg import ScriptedInference, fixture_config, generated_fixture, scripted_response

from data_designer_retrieval_sdg.multimodal import run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.inference import DataDesignerInference, Request
from data_designer_retrieval_sdg.multimodal.models import (
    ContextSummary,
    GenerationContext,
    MultimodalSDGConfig,
    QueryBatch,
    QueryInstruction,
    QuerySlot,
)
from data_designer_retrieval_sdg.multimodal.planning import (
    bounded_contexts,
    context_chars,
    context_instructions,
    duplicate_summary,
    related_contexts,
    select_summaries,
)
from data_designer_retrieval_sdg.multimodal.reporting import generation_report
from data_designer_retrieval_sdg.multimodal.storage import fingerprint, write_json
from data_designer_retrieval_sdg.multimodal.workflow import generate_candidates, load_contexts, localization_reasons
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource
from data_designer_retrieval_sdg.retrieval.source_file import load_retrieval_sources


def summary_row(key, units, quality=5, text="A substantive valve specification with useful source evidence"):
    return {
        "context": GenerationContext(context_id=key, unit_ids=units, language="en").model_dump(),
        "summary": {"summary": text, "visual_evidence": ""},
        "summary_judgment": {"fidelity": quality, "usefulness": quality, "reasoning": "Recorded"},
        "slots": {"queries": []},
    }


def test_document_sections_cover_every_unit_in_order(tmp_path):
    config = fixture_config(tmp_path).model_copy(
        update={"contexts_file": None, "context_strategy": "document", "max_units_per_context": 3}
    )
    sources = load_retrieval_sources(config.sources_file)
    contexts = load_contexts(config, sources)
    assert len(contexts) == 5
    assert [key for c in contexts for key in c.unit_ids] == [s.unit_id for s in sources]
    assert all(len({s.document_id for s in sources if s.unit_id in c.unit_ids}) == 1 for c in contexts)


def test_cross_document_bounds_interleave_without_losing_evidence(tmp_path):
    config = fixture_config(tmp_path).model_copy(update={"max_units_per_context": 2})
    sources = [
        RetrievalSource(unit_id=f"{doc}{i}", document_id=doc, text="complete content " * 3)
        for doc in ("a", "b")
        for i in range(3)
    ]
    original = [s.model_dump() for s in sources]
    contexts = bounded_contexts(
        [GenerationContext(context_id="both", unit_ids=[s.unit_id for s in sources])], sources, config
    )
    assert [c.unit_ids for c in contexts] == [["a0", "b0"], ["a1", "b1"], ["a2", "b2"]]
    assert [s.model_dump() for s in sources] == original
    chars = context_chars(["a0"], {s.unit_id: s for s in sources})
    config = config.model_copy(update={"max_context_chars": chars})
    assert len(bounded_contexts(contexts, sources, config)) == 6
    with pytest.raises(ValueError, match="smaller canonical units"):
        bounded_contexts(contexts, sources, config.model_copy(update={"max_context_chars": chars - 1}))


def test_related_summaries_build_pairs_and_triples_not_all_combinations(tmp_path):
    sources = [RetrievalSource(unit_id=key, document_id=key, text="Valve specification") for key in "abcd"]
    rows = [summary_row(key, [key]) for key in "abcd"]
    config = fixture_config(tmp_path).model_copy(update={"related_contexts_per_context": 2})
    contexts = related_contexts(rows, sources, config)
    assert contexts and {len(c.unit_ids) for c in contexts} == {2, 3}
    assert len(contexts) <= 2 * len(rows)
    assert len({tuple(sorted(c.unit_ids)) for c in contexts}) == len(contexts)
    rows[-1]["summary"]["summary"] = "Unrelated asteroid orbit"
    assert not any("d" in c.unit_ids for c in related_contexts(rows, sources, config))


def test_summary_quality_then_best_representative_then_fraction(tmp_path):
    config = fixture_config(tmp_path).model_copy(update={"summary_fraction": 0.5})
    rows = [
        summary_row("weak", ["a"], 4),
        summary_row("best", ["a"], 5),
        summary_row("other", ["b"], 5),
        summary_row("bad", ["c"], 2),
    ]
    chosen = select_summaries(rows, config)
    assert [r["context"]["context_id"] for r in chosen] == ["best"]
    assert rows[0]["representative_context_id"] == "best"
    assert rows[2]["selection_reason"] == "summary_budget"
    assert rows[3]["selection_reason"] == "summary_quality"
    assert len(rows) == 4  # No source judgment is discarded.
    count_config = config.model_copy(update={"summary_count": 2, "summary_fraction": None})
    assert len(select_summaries(rows, count_config)) == 2


def test_near_summary_dedup_protects_values_and_disjoint_evidence(tmp_path):
    config = fixture_config(tmp_path).model_copy(update={"summary_near_duplicate_threshold": 0.8})
    a = summary_row(
        "a",
        [str(i) for i in range(20)],
        text="A long substantive pressure specification describes the valve opening at 20 bar",
    )
    b = summary_row("b", [str(i) for i in range(21)], text=a["summary"]["summary"] + ".")
    assert duplicate_summary(a, b, config)
    b["summary"]["summary"] = b["summary"]["summary"].replace("20", "25")
    assert not duplicate_summary(a, b, config)
    b["summary"] = a["summary"].copy()
    b["context"]["unit_ids"] = ["unrelated"]
    assert not duplicate_summary(a, b, config)


def test_seeded_profiles_are_context_order_independent_and_not_gates(tmp_path):
    config = fixture_config(tmp_path).model_copy(update={"instructions_per_context": 2})
    first = context_instructions(config, "first")
    context_instructions(config, "second")
    assert context_instructions(config, "first") == first and len(first) == 2
    assert len({p.name for p in first}) == 2
    profile = QueryInstruction(
        name="expert",
        instruction="Seek an explanation",
        query_type="causal",
        format="keyword",
        persona="engineer",
        modality="image",
        answerability="distributed evidence",
    )
    config = config.model_copy(update={"instructions": [profile], "instructions_per_context": 1})
    sources = load_retrieval_sources(config.sources_file)
    inference = ProfileInference()
    candidates, outcomes = generate_candidates(config, sources, load_contexts(config, sources), inference)
    assert all(c.accepted and c.requested_instruction == profile for c in candidates)
    assert outcomes[0]["instructions"][0]["persona"] == "engineer"
    query_request = next(r for r in inference.requests if r.schema is QueryBatch)
    assert '"answerability": "distributed evidence"' in query_request.text


class ProfileInference(ScriptedInference):
    def generate(self, requests):
        self.requests.extend(requests)
        results = []
        for request in requests:
            if request.schema is QueryBatch:
                payload = json.loads(request.text.splitlines()[-1])
                results.append(
                    QueryBatch(
                        queries=[
                            QuerySlot(
                                slot=i,
                                query=f"What pressure opens {payload['sources'][0]['unit_id']}?",
                                evidence_modality="text",
                            )
                            for i in range(len(payload["instructions"]))
                        ]
                    )
                )
            else:
                results.append(scripted_response(request))
        return results


def test_quote_fidelity_is_diagnostic_unless_explicitly_strict(tmp_path):
    config, sources, candidates, _, _ = generated_fixture(tmp_path)
    context = load_contexts(config, sources)[0]
    localized = candidates[0].localization.model_copy(deep=True)
    localized.supports[0].quote = "Paraphrased pressure evidence"
    by_id = {s.unit_id: s for s in sources}
    assert not localization_reasons(localized, context, by_id)
    assert localization_reasons(localized, context, by_id, True) == ["unverified_text_evidence"]


def test_report_coverage_funnel_and_labels_are_independent(tmp_path):
    _, sources, candidates, outcomes, _ = generated_fixture(tmp_path)
    report = generation_report(sources, candidates, outcomes)
    assert report["coverage"]["generated"]["units"] == 11
    assert report["coverage"]["accepted_positive"]["documents"] == 1
    assert report["funnel"]["accepted_localized"] == 10
    assert report["generated_modalities"] == {"text": 10}
    assert report["multi_unit_positive_fraction"] == 1
    assert report["quote_verification"] == {"true": 20}
    assert report["requested_observed"]["query_type"] == {"unspecified -> numerical": 10}


def test_automatic_planning_selection_export_and_resume(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(
        update={
            "contexts_file": None,
            "context_strategy": "document",
            "max_units_per_context": 3,
            "related_contexts_per_context": 2,
            "summary_fraction": 0.5,
        }
    )
    source_rows = [json.loads(line) for line in config.sources_file.read_text().splitlines()]
    for row in source_rows:
        row["language"] = "en"
    config.sources_file.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
    handoff = run_multimodal_sdg(config)
    rows = json.loads((handoff.parent / "context_outcomes.json").read_text())
    report = json.loads((handoff.parent / "report.json").read_text())
    assert any("unselected" in r["context"]["unit_ids"] and len(r["context"]["unit_ids"]) > 1 for r in rows)
    assert all(len(r["context"]["unit_ids"]) <= 3 for r in rows)
    assert report["summary_selection"]["summary_budget"] > 0
    assert report["corpus_units"] == 13
    assert all(r["summary_judgment"] is not None for r in rows)
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff


def test_summary_judging_can_be_disabled_without_fabricating_passes(tmp_path):
    config = fixture_config(tmp_path).model_copy(update={"judge_summaries": False})
    sources = load_retrieval_sources(config.sources_file)
    _, outcomes = generate_candidates(config, sources, load_contexts(config, sources), ScriptedInference())
    assert all(row["summary_judgment"] is None for row in outcomes)


class PartialInference(DataDesignerInference):
    def run_batch(self, items):
        self.batches = getattr(self, "batches", []) + [[key for key, _ in items]]
        key, _ = items[0]
        payload = {"summary": "cached", "visual_evidence": ""}
        write_json(
            self.root / "responses" / f"{key}.json",
            {"request_id": key, "response": payload, "response_sha256": fingerprint(payload)},
        )


def test_missing_retries_never_regenerate_completed_responses(tmp_path):
    inference = PartialInference(tmp_path / "inference", fixture_config(tmp_path))
    requests = [Request(str(i), ContextSummary, "generator") for i in range(3)]
    assert len(inference.generate(requests)) == 3
    assert [len(batch) for batch in inference.batches] == [3, 2, 1]
    inference.generate(requests)
    assert len(inference.batches) == 3


def test_runtime_exception_stops_without_automatic_retry(tmp_path, monkeypatch):
    inference = DataDesignerInference(tmp_path / "inference", fixture_config(tmp_path))
    monkeypatch.setattr(inference, "run_batch", fail_batch)
    with pytest.raises(PermissionError):
        inference.generate([Request("one", ContextSummary, "judge")])


def fail_batch(items):
    raise PermissionError("Provider failed")


def test_completed_rows_can_be_collected_after_failed_batch(tmp_path):
    inference = DataDesignerInference(tmp_path / "inference", fixture_config(tmp_path))
    request = Request("one", ContextSummary, "generator")
    key = inference.request_key(request)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [{"request_id": key, "response": json.dumps({"summary": "completed", "visual_evidence": ""})}]
        ),
        dataset / "batch_0.parquet",
    )
    inference.collect_responses([(key, request)], dataset, tmp_path / "attempt")
    assert inference.cached(request).summary == "completed"


@pytest.mark.parametrize(
    "updates",
    [
        {"summary_fraction": 0.5, "summary_count": 2},
        {"instructions_per_context": 4},
        {"missing_response_attempts": 4},
        {"summary_near_duplicate_threshold": 0},
        {"related_contexts_per_context": 3},
        {"max_context_chars": 0},
    ],
)
def test_new_controls_are_validated(tmp_path, updates):
    with pytest.raises(ValidationError):
        MultimodalSDGConfig.model_validate({**fixture_config(tmp_path).model_dump(), **updates})
