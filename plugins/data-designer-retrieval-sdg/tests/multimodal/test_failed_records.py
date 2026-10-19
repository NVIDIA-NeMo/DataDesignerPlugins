# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed records use the real DD parser, retries, persistence, workflow and export."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.testing import make_stub_completion_response
from data_designer.interface.errors import DataDesignerGenerationError
from test_multimodal_sdg import fixture_config, scripted_response

from data_designer_retrieval_sdg.multimodal import models, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.inference import DataDesignerInference, Request
from data_designer_retrieval_sdg.multimodal.storage import digest
from data_designer_retrieval_sdg.multimodal.structured import FAILURE_KEY


class ControlledTransport:
    """Fail one schema/source at the transport boundary without replacing generation."""

    def __init__(self, schema: str, unit: str | None = "unit-0"):
        self.schema = schema
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
            name == self.schema
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


@pytest.mark.parametrize(
    "schema,stage,accepted",
    [
        ("VisualDescription", "visual-enrichment", 3),
        ("Description", "document-description", 3),
        ("ContextSummary", "summary", 2),
        ("SummaryJudgment", "summary-judgment", 2),
        ("QueryBatch", "query-generation", 2),
        ("QueryMetadata", "query-metadata", 2),
        ("SelfSufficiencyJudgment", "self-sufficiency", 2),
        ("RelevanceJudgment", "relevance", 2),
        ("Localization", "localization", 2),
    ],
)
def test_failed_record_preserves_other_queries_and_completed_resume(tmp_path, monkeypatch, schema, stage, accepted):
    config = skip_config(tmp_path)
    original_sources = config.sources_file.read_bytes()
    transport = ControlledTransport(schema)
    completion = install_transport(monkeypatch, transport)
    handoff = run_multimodal_sdg(config)
    report = json.loads((handoff.parent / "report.json").read_text())
    ledger = json.loads((handoff.parent / "generation_failures.json").read_text())
    assert ledger == json.loads((config.output_dir / "generation_failures.json").read_text())
    assert report["accepted_query_ids"] == accepted
    assert report["corpus_units"] == 4
    assert config.sources_file.read_bytes() == original_sources
    assert len(ledger["records"]) == 1
    record = ledger["records"][0]
    assert record["stage"] == stage
    assert "unit-0" in record["source_unit_ids"]
    assert record["failure"]["kind"] == "model_validation_exhausted"
    assert transport.calls[schema, "unit-0"] == record["failure"]["max_model_attempts"]
    assert report["generation_failures"]["by_stage"] == {stage: 1}
    if schema == "QueryBatch":
        assert report["requested_slots"] == 9
        assert report["generation_failures"]["failed_requested_slots"] == 3
        assert report["abstentions"] == 4
    if schema == "SummaryJudgment":
        outcomes = json.loads((handoff.parent / "context_outcomes.json").read_text())
        failed = next(row for row in outcomes if row["context"]["context_id"] == "context-0")
        assert failed["selection_reason"] == "summary_judgment_failed"
        assert failed["summary_judgment"] is None and not failed["slots"]["queries"]
    if "query_id" in record:
        candidates = [json.loads(line) for line in (handoff.parent / "source_records.jsonl").read_text().splitlines()]
        assert record["query_id"] not in {row["query_id"] for row in candidates}
        assert report["generation_failures"]["excluded_query_ids"] == 1
    files = {p: digest(p) for p in config.output_dir.rglob("*") if p.is_file()}
    calls = completion.await_count
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    assert completion.await_count == calls
    assert all(digest(p) == checksum for p, checksum in files.items())


@pytest.mark.parametrize("schema", ["ContextSummary", "SummaryJudgment"])
def test_unit_contexts_exclude_failed_summaries(tmp_path, monkeypatch, schema):
    config = skip_config(tmp_path, "unit")
    install_transport(monkeypatch, ControlledTransport(schema))
    handoff = run_multimodal_sdg(config)
    assert json.loads((handoff.parent / "report.json").read_text())["accepted_query_ids"] == 2


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
    assert audit["toc_confirmations"]["unit-0"] == {"generation_failed": True}
    assert not audit["documents"][0]["accepted"]


def one_combination(rows, config):
    """Choose a deterministic pair without loading an embedding model."""
    return [(0, 1)]


def test_failed_combination_omits_only_unavailable_summary(tmp_path, monkeypatch):
    config = skip_config(tmp_path)
    transport = ControlledTransport("ContextSummary")
    transport.prompt_contains = "Write a combined summary"
    install_transport(monkeypatch, transport)
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.summary_stages.semantic_combinations", one_combination)
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
    assert records and all(record["stage"] == "summary" and len(record["source_unit_ids"]) > 1 for record in records)
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


@pytest.mark.parametrize("keys", [["unexpected"], ["expected", "expected"]])
def test_skip_never_relaxes_native_row_identity_checks(tmp_path, keys):
    config = skip_config(tmp_path)
    inference = DataDesignerInference(tmp_path / "inference", config)
    request = Request("Summarize", models.ContextSummary, "generator")
    dataset = tmp_path / "native"
    dataset.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"request_id": key, "response": None} for key in keys]), dataset / "batch_000.parquet"
    )
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        inference.collect_responses([("expected", request)], dataset, tmp_path / "attempt")


def test_cached_failure_integrity_and_policy_are_checked(tmp_path, monkeypatch):
    config = skip_config(tmp_path)
    install_transport(monkeypatch, ControlledTransport("ContextSummary", unit=None))
    inference = DataDesignerInference(tmp_path / "inference", config)
    request = Request("Summarize", models.ContextSummary, "generator")
    assert isinstance(inference.generate([request])[0], models.FailedGeneration)
    strict = DataDesignerInference(inference.root, config.model_copy(update={"failed_record_policy": "error"}))
    with pytest.raises(ValueError, match="requires the skip policy"):
        strict.generate([request])
    path = next((inference.root / "responses").glob("*.json"))
    value = json.loads(path.read_text())
    value["response"][FAILURE_KEY]["detail"] = "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity"):
        inference.generate([request])
