# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed records use the real DD parser, retries, persistence, workflow and export."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from data_designer.config import RunConfig
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.testing import make_stub_completion_response
from data_designer.interface.errors import DataDesignerGenerationError
from test_multimodal_sdg import fixture_config, scripted_response
from test_multimodal_sections import SectionInference
from workflow_test_utils import install_inference

from data_designer_retrieval_sdg.multimodal import models, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.execution import verify_boundaries
from data_designer_retrieval_sdg.multimodal.inference import Request
from data_designer_retrieval_sdg.multimodal.storage import digest, fingerprint


class InvalidJSON:
    """Return invalid transport content without bypassing the Data Designer parser."""

    def model_dump_json(self):
        return '{"duplicate": 1, "duplicate": 2}'


class FailSelected(SectionInference):
    """Fail a bounded number of calls to one schema and record actual attempts."""

    def __init__(self, schema, failures=2, provider_error=False):
        super().__init__()
        self.schemas = {schema} if isinstance(schema, str) else set(schema)
        self.failures = failures
        self.provider_error = provider_error
        self.calls = Counter()

    def generate(self, requests):
        responses = super().generate(requests)
        for index, request in enumerate(requests):
            name = "QueryBatch" if issubclass(request.schema, models.QueryBatch) else request.schema.__name__
            self.calls[name] += 1
            if name in self.schemas and self.calls[name] <= self.failures:
                if self.provider_error:
                    raise PermissionError("Provider permission denied")
                responses[index] = InvalidJSON()
        return responses


@pytest.mark.parametrize(
    "schema,stage",
    [
        ("VisualDescription", "visual-enrichment"),
        ("Description", "document-description"),
        ("ContextSummary", "summaries"),
        ("SummaryJudgment", "summary-judgments"),
        ("QueryBatch", "queries-0"),
        ("QueryMetadata", "query-judgments"),
        ("SelfSufficiencyJudgment", "query-judgments"),
        ("RelevanceJudgment", "query-judgments"),
        ("Localization", "query-judgments"),
    ],
)
def test_skip_after_validation_retries_preserves_export_and_resume(tmp_path, monkeypatch, schema, stage):
    config = fixture_config(tmp_path).model_copy(
        update={
            "context_strategy": "sections",
            "failed_record_policy": "skip",
            "run_config": RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=0),
            "concurrency": 1,
            "batch_size": 30,
        }
    )
    inference = FailSelected(schema)
    completion = install_inference(monkeypatch, inference)
    handoff = run_multimodal_sdg(config)
    bundle = handoff.parent
    failures = json.loads((bundle / "generation_failures.json").read_text())
    assert len(failures["records"]) == 1
    failure = failures["records"][0]
    assert failure["stage"] == stage
    assert failure["row_id"] == 0
    assert failure["reason"] == "model_validation_exhausted"
    assert next(iter(failure["columns"].values()))["max_model_attempts"] == 2
    assert next(iter(failure["columns"].values()))["last_invalid_response"] == InvalidJSON().model_dump_json()
    assert inference.calls[schema] >= 2
    assert len((bundle / "retrieval_units.jsonl").read_text().splitlines()) == 13
    assert json.loads((bundle / "report.json").read_text())["accepted_query_ids"] > 0
    assert (
        all(
            record["query"] != failure["query"]
            for record in map(json.loads, (bundle / "source_records.jsonl").read_text().splitlines())
        )
        if failure["query"]
        else True
    )
    if schema == "QueryBatch":
        report = json.loads((bundle / "report.json").read_text())
        assert report["requested_slots"] == 30
        assert report["generation_failures"]["failed_requested_slots"] == 3
        assert report["abstentions"] == 18
    if schema == "SummaryJudgment":
        outcomes = json.loads((bundle / "context_outcomes.json").read_text())
        assert outcomes[0]["selection_reason"] == "summary_judgment_failed"
        assert outcomes[0]["slots"]["queries"] == []
    report = json.loads((bundle / "report.json").read_text())
    assert failures == json.loads((config.output_dir / "generation_failures.json").read_text())
    assert "excluded_candidates" not in failures
    assert "excluded_joint_candidates" not in report["generation_failures"]
    if stage == "query-judgments":
        assert set(failure["observed_responses"]) == {
            "metadata",
            "sufficiency",
            "relevance",
            "localization",
        }
        outcome = json.loads((bundle / "context_outcomes.json").read_text())[0]
        assert failure["query_id"] == "q_" + fingerprint(["context-0", 0, failure["query"]])[:32]
        assert failure["slot"] == 0
        assert failure["context_id"] == "context-0"
        assert failure["source_unit_ids"] == ["unit-0", "unit-11"]
        assert failure["language"] == outcome["context"]["language"]
        assert failure["requested_instruction"] == outcome["instructions"][0]
        assert not {"accepted", "rejection_reasons", "query_judgment", "localization"}.intersection(failure)
        candidates = list(map(json.loads, (bundle / "source_records.jsonl").read_text().splitlines()))
        assert len(candidates) == report["candidate_records"] == 9
        assert failure["query_id"] not in {candidate["query_id"] for candidate in candidates}
        assert report["generation_failures"]["excluded_query_ids"] == 1
        assert report["funnel"] == {
            "generated": 10,
            "after_generation_failures": 9,
            "after_insufficient_standalone_query": 9,
            "after_query_contains_answer": 9,
            "after_insufficient_relevance": 9,
            "accepted_localized": 9,
        }
        assert report["retention_fraction"] == pytest.approx(0.9)
        assert report["rejection_reasons"] == {}
    else:
        assert "query_id" not in failure
        assert report["generation_failures"]["excluded_query_ids"] == 0
        assert "after_generation_failures" not in report["funnel"]
    verify_boundaries(config.output_dir / "data_designer")
    files = {p: digest(p) for p in config.output_dir.rglob("*") if p.is_file()}
    calls = completion.await_count
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    assert completion.await_count == calls
    assert all(digest(p) == sha for p, sha in files.items())
    # Completion integrity includes the portable failure ledger.
    (bundle / "generation_failures.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))


def test_no_accepted_queries_still_fails_and_retains_failure_audit(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(
        update={
            "summary_count": 1,
            "failed_record_policy": "skip",
            "run_config": RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=0),
        }
    )
    inference = FailSelected("QueryBatch", failures=1000)
    install_inference(monkeypatch, inference)
    with pytest.raises(ValueError, match="No candidates passed"):
        run_multimodal_sdg(config)
    assert inference.calls["QueryBatch"] == 2
    assert not (config.output_dir / "complete.json").exists()
    assert len(json.loads((config.output_dir / "generation_failures.json").read_text())["records"]) == 1


def test_provider_errors_are_not_skipped(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 1, "failed_record_policy": "skip"})
    install_inference(monkeypatch, FailSelected("QueryBatch", failures=1000, provider_error=True))
    with pytest.raises(DataDesignerGenerationError):
        run_multimodal_sdg(config)
    assert not (config.output_dir / "complete.json").exists()
    assert not (config.output_dir / "generation_failures.json").exists()


class IncompleteLocalization:
    """Preserve one valid support in a response with an invalid second entry."""

    def __init__(self, source):
        self.source = source

    def model_dump_json(self):
        return json.dumps(
            {
                "supports": [
                    {
                        "unit_id": self.source["unit_id"],
                        "modality": "text",
                        "grade": 1,
                        "quote": self.source["text"],
                        "visual_evidence": "",
                        "contribution": "Partial pressure evidence",
                    },
                    {"unit_id": "invalid-missing-fields"},
                ]
            }
        )


class FailLocalization(FailSelected):
    """Retain partial response evidence when combined localization fails validation."""

    def generate(self, requests):
        responses = super().generate(requests)
        for index, (request, response) in enumerate(zip(requests, responses, strict=True)):
            if request.schema is models.Localization and isinstance(response, InvalidJSON):
                payload = json.loads(request.text.splitlines()[-1])
                responses[index] = IncompleteLocalization(payload["sources"][0])
        return responses


def test_failed_localization_excludes_whole_query_and_preserves_observed_evidence(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(
        update={
            "summary_count": 2,
            "failed_record_policy": "skip",
            "run_config": RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=0),
            "concurrency": 1,
        }
    )
    inference = FailLocalization("Localization")
    install_inference(monkeypatch, inference)
    handoff = run_multimodal_sdg(config)
    audit = json.loads((handoff.parent / "generation_failures.json").read_text())
    assert len(audit["records"]) == 1
    failed = audit["records"][0]
    invalid_response = failed["columns"]["localization"]["last_invalid_response"]
    original = json.loads(invalid_response.removeprefix("```json\n").removesuffix("\n```"))
    assert original["supports"][0]["unit_id"] in failed["source_unit_ids"]
    assert original["supports"][0]["quote"] == "Valve 0 opens at 20 bar."
    assert failed["observed_responses"]["relevance"]["relevance"] == 5
    accepted = list(map(json.loads, (handoff.parent / "source_records.jsonl").read_text().splitlines()))
    assert len(accepted) == 1
    assert failed["query"] not in {row["query"] for row in accepted}
    assert inference.calls["Localization"] == 3  # Two queries plus one fresh retry.
    report = json.loads((handoff.parent / "report.json").read_text())
    assert report["generation_failures"]["excluded_query_ids"] == 1
    assert report["funnel"]["generated"] == 2
    assert report["funnel"]["after_generation_failures"] == 1
    assert report["retention_fraction"] == pytest.approx(0.5)


def test_skip_resume_recovers_only_missing_rows_and_keeps_every_saved_batch(tmp_path, monkeypatch):
    from test_execution import InterruptedQueries

    config = fixture_config(tmp_path).model_copy(
        update={
            "summary_count": 2,
            "failed_record_policy": "skip",
            "run_config": RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=0),
            "concurrency": 1,
        }
    )
    install_inference(monkeypatch, InterruptedQueries())
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    saved = {p: digest(p) for p in config.output_dir.rglob("parquet-files/*.parquet")}
    inference = FailSelected("QueryBatch", failures=1000)
    install_inference(monkeypatch, inference)
    handoff = run_multimodal_sdg(config.model_copy(update={"resume": True}))
    assert inference.calls["QueryBatch"] == 2
    assert all(digest(p) == sha for p, sha in saved.items())
    audit = json.loads((handoff.parent / "generation_failures.json").read_text())
    assert len(audit["records"]) == 1
    assert "recovery/attempt-" in audit["records"][0]["response_artifact"]
    assert json.loads((handoff.parent / "report.json").read_text())["accepted_query_ids"] == 1


def test_multiple_failed_judge_columns_count_one_excluded_query(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(
        update={
            "summary_count": 2,
            "failed_record_policy": "skip",
            "run_config": RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=0),
            "concurrency": 1,
        }
    )
    inference = FailSelected(("QueryMetadata", "RelevanceJudgment"))
    completion = install_inference(monkeypatch, inference)
    handoff = run_multimodal_sdg(config)
    audit = json.loads((handoff.parent / "generation_failures.json").read_text())
    report = json.loads((handoff.parent / "report.json").read_text())
    (failure,) = audit["records"]
    assert set(failure["columns"]) == {"metadata", "relevance"}
    assert report["generation_failures"]["failed_records"] == 1
    assert report["generation_failures"]["excluded_query_ids"] == 1
    assert report["generation_failures"]["by_stage"] == {"query-judgments": 1}
    assert report["candidate_records"] == report["funnel"]["after_generation_failures"] == 1
    assert report["funnel"]["generated"] == 2
    assert report["funnel"]["accepted_localized"] == 1
    assert report["retention_fraction"] == pytest.approx(0.5)
    assert report["rejection_reasons"] == {}
    assert report["corpus_units"] == 13
    candidates = json.loads((config.output_dir / "candidates.json").read_text())
    assert len(candidates) == 1 and candidates[0]["query_id"] != failure["query_id"]
    files = {path: digest(path) for path in config.output_dir.rglob("*") if path.is_file()}
    calls = completion.await_count
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    assert completion.await_count == calls
    assert all(digest(path) == checksum for path, checksum in files.items())


class ControlledTransport:
    """Fail one schema/source at the transport boundary without replacing generation."""

    def __init__(self, schema: str | tuple[str, ...], unit: str | None = "unit-0"):
        self.schemas = {schema} if isinstance(schema, str) else set(schema)
        self.unit = unit
        self.provider_unit: str | None = None
        self.prompt_contains: str | None = None
        self.calls = Counter()

    async def complete(self, messages, **kwargs):
        """Read the real rendered schema and return controlled model text."""
        content = messages[0].to_dict()["content"]
        text = (
            "\n".join(item["text"] for item in content if item["type"] == "text")
            if isinstance(content, list)
            else content
        )
        schema = json.loads(text.split("<response_schema>\n")[1].split("\n</response_schema>")[0])
        title = schema["title"]
        model = (
            models.QueryBatch.for_slot_count(schema["properties"]["queries"]["minItems"])
            if title.startswith("QueryBatch")
            else getattr(models, title)
        )
        prompt = text.split("\n\n* Your response must be in JSON format.")[0]
        match = re.search(r"unit-\d+", prompt)
        unit = match.group() if match else "unselected"
        name = "QueryBatch" if issubclass(model, models.QueryBatch) else title
        self.calls[name, unit] += 1
        if name == "QueryBatch" and unit == self.provider_unit:
            raise PermissionError("Controlled provider permission failure")
        if (
            name in self.schemas
            and (self.unit is None or unit == self.unit)
            and (self.prompt_contains is None or self.prompt_contains in prompt)
        ):
            return make_stub_completion_response(content="not JSON")
        if model is models.ContextSummary:
            response = models.ContextSummary(summary=f"Valve specification for {unit}", visual_evidence="")
        elif model is models.VisualDescription:
            response = models.VisualDescription(description="A pressure diagram", has_visual_content=True)
        elif model is models.Description:
            response = models.Description(description="Valve documentation")
        else:
            response = scripted_response(Request(prompt, model, "generator"))
        return make_stub_completion_response(content=response.model_dump_json())


def install_transport(monkeypatch, transport: ControlledTransport) -> AsyncMock:
    """Replace only the hosted model call; every parser and workflow consumer is real."""
    monkeypatch.setenv("TEST_SDG_KEY", "mocked-credential")
    completion = AsyncMock(side_effect=transport.complete)
    monkeypatch.setattr(ModelFacade, "acompletion", completion)
    return completion


def skip_config(tmp_path: Path, strategy: str = "sections") -> models.MultimodalSDGConfig:
    """Keep three contexts plus a corpus-only distractor in a small real workflow."""
    config = fixture_config(tmp_path)
    sources = [json.loads(line) for line in config.sources_file.read_text().splitlines()]
    config.sources_file.write_text("".join(json.dumps(row) + "\n" for row in [*sources[:3], sources[-1]]))
    config.contexts_file.write_text(
        "".join(json.dumps({"context_id": f"context-{i}", "unit_ids": [f"unit-{i}"]}) + "\n" for i in range(3))
    )
    return config.model_copy(update={"context_strategy": strategy, "failed_record_policy": "skip", "concurrency": 1})


@pytest.mark.parametrize("schema", ["ContextSummary", "SummaryJudgment"])
def test_unit_contexts_exclude_failed_summaries(tmp_path, monkeypatch, schema):
    config = skip_config(tmp_path, "unit")
    install_transport(monkeypatch, ControlledTransport(schema))
    handoff = run_multimodal_sdg(config)
    assert json.loads((handoff.parent / "report.json").read_text())["accepted_query_ids"] == 2
    records = json.loads((handoff.parent / "generation_failures.json").read_text())["records"]
    assert len(records) == 1
    diagnostic = next(iter(records[0]["columns"].values()))
    native = RunConfig()
    assert diagnostic["max_model_attempts"] == (
        (native.max_conversation_restarts + 1) * (native.max_conversation_correction_steps + 1)
    )


@pytest.mark.parametrize("schema", ["VisualDescription", "Description"])
def test_all_enrichment_failures_keep_sources_and_allow_generation(tmp_path, monkeypatch, schema):
    config = skip_config(tmp_path)
    install_transport(monkeypatch, ControlledTransport(schema, unit=None))
    handoff = run_multimodal_sdg(config)
    report = json.loads((handoff.parent / "report.json").read_text())
    assert report["corpus_units"] == 4 and report["accepted_query_ids"] == 3
    assert report["generation_failures"]["failed_records"] > 0


def test_toc_failure_keeps_sources_without_unverified_projection(tmp_path, monkeypatch):
    config = skip_config(tmp_path).model_copy(update={"contexts_file": None, "section_size": 1})
    rows = [json.loads(line) for line in config.sources_file.read_text().splitlines()]
    rows[0]["text"] = "Table of contents\nAlpha ... 1\nBeta ... 2\nGamma ... 3"
    config.sources_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    install_transport(monkeypatch, ControlledTransport("TOCConfirmation", unit=None))
    handoff = run_multimodal_sdg(config)
    report = json.loads((handoff.parent / "report.json").read_text())
    assert report["corpus_units"] == 4
    assert report["generation_failures"]["by_stage"] == {"toc-confirmation": 1}
    audit = json.loads((config.output_dir / "planning/section_boundaries.json").read_text())
    assert audit["coverage"]["exact_ordered_coverage"]
    assert "unit-0" not in audit["toc_confirmations"]
    assert not audit["documents"][0]["accepted"]


def one_combination(rows, config):
    """Choose a deterministic pair without loading an embedding model."""
    return [(0, 1)]


def test_failed_combination_omits_only_unavailable_summary(tmp_path, monkeypatch):
    config = skip_config(tmp_path)
    transport = ControlledTransport("ContextSummary")
    transport.prompt_contains = "Write a combined summary"
    install_transport(monkeypatch, transport)
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.stages.semantic_combinations", one_combination)
    handoff = run_multimodal_sdg(config)
    report = json.loads((handoff.parent / "report.json").read_text())
    assert report["accepted_query_ids"] == 3
    records = json.loads((handoff.parent / "generation_failures.json").read_text())["records"]
    assert len(records) == 1 and records[0]["stage"] == "summary-combinations"
    assert records[0]["source_unit_ids"] == ["unit-0", "unit-1"]
    assert records[0]["summary_context_ids"] == ["context-0", "context-1"]


def test_failed_related_summary_keeps_original_contexts(tmp_path, monkeypatch):
    config = skip_config(tmp_path, "unit").model_copy(update={"related_contexts_per_context": 1})
    sources = [json.loads(line) for line in config.sources_file.read_text().splitlines()]
    sources[1]["document_id"] = "another-document"
    config.sources_file.write_text("".join(json.dumps(row) + "\n" for row in sources))
    transport = ControlledTransport("ContextSummary")
    transport.prompt_contains = '"unit_id": "unit-1"'
    install_transport(monkeypatch, transport)
    handoff = run_multimodal_sdg(config)
    records = json.loads((handoff.parent / "generation_failures.json").read_text())["records"]
    assert records and all(
        record["stage"] == "related-summaries" and len(record["source_unit_ids"]) > 1 for record in records
    )
    candidates = json.loads((config.output_dir / "candidates.json").read_text())
    assert {"context-0", "context-1", "context-2"} <= {candidate["context_id"] for candidate in candidates}


@pytest.mark.parametrize("schema", ["ContextSummary", "QueryBatch", "SummaryJudgment", "Localization"])
def test_all_required_responses_fail_without_fabricating_export(tmp_path, monkeypatch, schema):
    config = skip_config(tmp_path)
    install_transport(monkeypatch, ControlledTransport(schema, unit=None))
    with pytest.raises(ValueError, match="No candidates passed"):
        run_multimodal_sdg(config)
    assert not (config.output_dir / "complete.json").exists()
    assert not (config.output_dir / "bundle").exists()
    assert json.loads((config.output_dir / "generation_failures.json").read_text())["records"]


def test_interrupted_resume_reuses_successes_and_terminal_failures(tmp_path, monkeypatch):
    config = skip_config(tmp_path)
    transport = ControlledTransport("QueryBatch")
    transport.provider_unit = "unit-2"
    install_transport(monkeypatch, transport)
    with pytest.raises((RuntimeError, DataDesignerGenerationError)):
        run_multimodal_sdg(config)
    assert not (config.output_dir / "complete.json").exists()
    saved = {p: digest(p) for p in config.output_dir.rglob("*.parquet")}
    before = dict(transport.calls)
    assert before["QueryBatch", "unit-0"] == (
        (config.run_config.max_conversation_restarts + 1) * (config.run_config.max_conversation_correction_steps + 1)
    )
    assert before["QueryBatch", "unit-1"] == 1
    assert saved
    transport.provider_unit = None
    handoff = run_multimodal_sdg(config.model_copy(update={"resume": True}))
    assert all(digest(p) == checksum for p, checksum in saved.items())
    assert transport.calls["QueryBatch", "unit-0"] == before["QueryBatch", "unit-0"]
    assert transport.calls["QueryBatch", "unit-1"] == before["QueryBatch", "unit-1"]
    assert transport.calls["QueryBatch", "unit-2"] == before["QueryBatch", "unit-2"] + 1
    report = json.loads((handoff.parent / "report.json").read_text())
    assert report["accepted_query_ids"] == 2
    assert report["generation_failures"]["failed_records"] == 1


@pytest.mark.parametrize("artifact", ["generation_failures.json", "bundle/generation_failures.json"])
def test_completed_resume_checks_failure_ledger(tmp_path, monkeypatch, artifact):
    config = skip_config(tmp_path)
    install_transport(monkeypatch, ControlledTransport("QueryBatch"))
    run_multimodal_sdg(config)
    (config.output_dir / artifact).write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))


def test_default_policy_still_stops_persistent_validation_failure(tmp_path, monkeypatch):
    config = skip_config(tmp_path).model_copy(update={"failed_record_policy": "error"})
    install_transport(monkeypatch, ControlledTransport("QueryBatch"))
    with pytest.raises((RuntimeError, DataDesignerGenerationError)):
        run_multimodal_sdg(config)
    assert not (config.output_dir / "complete.json").exists()
    assert not (config.output_dir / "generation_failures.json").exists()
