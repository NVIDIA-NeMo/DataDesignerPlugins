# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer execution and checkpoint contracts, with controlled model transport."""

import base64
import json
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_designer.config import RunConfig
from data_designer.interface.errors import DataDesignerGenerationError
from test_multimodal_sdg import fixture_config
from test_multimodal_sections import SectionInference
from test_source_directory import directory_config
from workflow_test_utils import install_inference

from data_designer_retrieval_sdg.multimodal import execution, models, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.execution import generate_candidates
from data_designer_retrieval_sdg.multimodal.source_file import load_retrieval_sources
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource
from data_designer_retrieval_sdg.multimodal.storage import digest
from data_designer_retrieval_sdg.multimodal.workflow import load_contexts


@pytest.mark.parametrize("strategy", ["unit", "sections"])
def test_workflow_completed_native_resume(tmp_path, monkeypatch, strategy):
    config = fixture_config(tmp_path).model_copy(update={"context_strategy": strategy, "concurrency": 2})
    sources = load_retrieval_sources(config.sources_file)
    contexts = load_contexts(config, sources)
    completion = install_inference(monkeypatch, SectionInference())
    candidates, outcomes = generate_candidates(config, sources, contexts)
    assert candidates and outcomes
    assert any(candidate.accepted for candidate in candidates)
    calls = completion.await_count
    assert calls > 0
    assert generate_candidates(config.model_copy(update={"resume": True}), sources, contexts) == (
        candidates,
        outcomes,
    )
    assert completion.await_count == calls


class CorpusInference(SectionInference):
    """Ground controlled answers in each fixture's available text or image evidence."""

    def generate(self, requests):
        responses = super().generate(requests)
        for request, response in zip(requests, responses, strict=True):
            if isinstance(response, models.QueryBatch):
                payload = json.loads(request.text.splitlines()[-1])
                kinds = {"text" if source["text"] else "image" for source in payload["sources"]}
                response.queries[0].evidence_modality = next(iter(kinds)) if len(kinds) == 1 else "text_and_image"
            elif isinstance(response, models.Localization):
                payload = json.loads(request.text.splitlines()[-1])
                response.supports = [
                    models.Support(
                        unit_id=source["unit_id"],
                        grade=1,
                        modality="image",
                        quote="",
                        visual_evidence="Fixture pressure diagram",
                        contribution="Partial specification",
                    )
                    for source in payload["sources"]
                    if not source["text"]
                ] + [support for support in response.supports if support.quote]
        return responses


@pytest.mark.parametrize("suffix", [".png", ".bin", ""])
def test_workflow_image_references_preserve_bytes_and_resume(tmp_path, monkeypatch, suffix):
    config = fixture_config(tmp_path).model_copy(update={"batch_size": 1})
    rows = [json.loads(line) for line in config.sources_file.read_text().splitlines()][:2]
    original = (tmp_path / rows[1]["images"][0]).read_bytes()
    image = tmp_path / ("uploaded" + suffix)
    image.write_bytes(original)
    rows[0]["images"] = []
    rows[1].update(text="", images=[image.name])
    config.sources_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config.contexts_file.write_text(
        "".join(json.dumps({"context_id": row["unit_id"], "unit_ids": [row["unit_id"]]}) + "\n" for row in rows)
    )
    completion = install_inference(monkeypatch, CorpusInference())
    handoff = run_multimodal_sdg(config)
    attachments = []
    for call in completion.await_args_list:
        for message in call.args[1]:
            content = message.to_dict()["content"]
            if isinstance(content, list):
                attachments.extend(item["source"]["data"] for item in content if item["type"] == "image")
    assert attachments and all(base64.b64decode(value) == original for value in attachments)
    references = []
    for seed in (config.output_dir / "data_designer").rglob("seed.parquet"):
        table = pq.read_table(seed)
        for name in table.column_names:
            if name.endswith("_images"):
                references.extend(value for values in table[name].to_pylist() for value in values)
    assert references and all(
        Path(value).is_absolute() and Path(value).read_bytes() == original for value in references
    )
    calls = completion.await_count
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    assert completion.await_count == calls
    assert image.read_bytes() == original


@pytest.mark.parametrize("mix", ["text", "image", "heterogeneous"])
def test_workflow_sections_handle_available_modalities(tmp_path, monkeypatch, mix):
    config = fixture_config(tmp_path).model_copy(update={"context_strategy": "sections", "summary_count": 2})
    rows = [json.loads(line) for line in config.sources_file.read_text().splitlines()]
    for index, row in enumerate(rows):
        if mix == "text" or (mix == "heterogeneous" and index % 3 == 0):
            row["images"] = []
        if mix == "image" or (mix == "heterogeneous" and index % 3 == 1):
            row["text"] = ""
    config.sources_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inference = CorpusInference()
    install_inference(monkeypatch, inference)
    handoff = run_multimodal_sdg(config)
    candidates = json.loads((config.output_dir / "candidates.json").read_text())
    assert len(candidates) == 2 and all(candidate["accepted"] for candidate in candidates)
    visual_requests = [request for request in inference.requests if request.schema is models.VisualDescription]
    assert len(visual_requests) == sum(bool(row["images"]) for row in rows)
    if mix == "text":
        assert not any(request.images for request in inference.requests)
    judge_seeds = [
        table
        for path in (config.output_dir / "data_designer").rglob("seed.parquet")
        if "metadata_prompt" in (table := pq.read_table(path)).column_names
    ]
    assert len(judge_seeds) == 1
    review_rows = judge_seeds[0].to_pylist()
    assert {row["query"] for row in review_rows} == {candidate["query"] for candidate in candidates}
    for row in review_rows:
        candidate = next(candidate for candidate in candidates if candidate["query"] == row["query"])
        assert row["context_id"] == candidate["context_id"]
        assert row["source_unit_ids"] == candidate["source_unit_ids"]
        assert set(row["image_unit_ids"]) <= set(row["source_unit_ids"])
    report = json.loads((handoff.parent / "report.json").read_text())
    assert report["views"]["mixed"]["accepted_queries"] == 2
    assert report["corpus_units"] == len(rows)


class OptionalMetadataDefaults(SectionInference):
    """Omit valid optional fields in one row and supply them in another."""

    def __init__(self):
        super().__init__()
        self.metadata_calls = 0
        self.raw_metadata = []

    def generate(self, requests):
        responses = super().generate(requests)
        for request, response in zip(requests, responses, strict=True):
            if request.schema is models.QueryMetadata:
                self.metadata_calls += 1
                if self.metadata_calls == 1:
                    delattr(response, "classification_reasoning")
                    delattr(response, "classification_confidence")
                else:
                    response.classification_confidence = 0.75
                self.raw_metadata.append(response.model_dump(mode="json"))
        return responses


@pytest.mark.parametrize("batch_size", [1, 30])
def test_workflow_optional_metadata_defaults_survive_mixed_rows_and_resume(tmp_path, monkeypatch, batch_size):
    config = fixture_config(tmp_path).model_copy(
        update={"summary_count": 2, "concurrency": 1, "batch_size": batch_size}
    )
    inference = OptionalMetadataDefaults()
    completion = install_inference(monkeypatch, inference)
    handoff = run_multimodal_sdg(config)
    assert handoff.is_file()
    assert inference.metadata_calls == 2
    assert "classification_reasoning" not in inference.raw_metadata[0]
    assert "classification_confidence" not in inference.raw_metadata[0]
    stage = next((config.output_dir / "data_designer/multimodal").glob("stage-*-query-judgments"))
    metadata = [
        row["metadata"]
        for path in sorted((stage / "parquet-files").glob("*.parquet"))
        for row in pq.read_table(path, columns=["metadata"]).to_pylist()
    ]
    assert {row["classification_reasoning"] for row in metadata} == {"", "Standalone"}
    assert {row["classification_confidence"] for row in metadata} == {None, 0.75}
    candidates = json.loads((config.output_dir / "candidates.json").read_text())
    assert len(candidates) == 2 and all(candidate["accepted"] for candidate in candidates)
    calls = completion.await_count
    saved = {path: digest(path) for path in config.output_dir.rglob("*.parquet")}
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    assert completion.await_count == calls
    assert all(digest(path) == value for path, value in saved.items())


class WhitespaceSupport(SectionInference):
    """Return one schema-invalid support before a valid fresh retry."""

    def __init__(self, field):
        super().__init__()
        self.field = field
        self.localization_calls = 0
        self.raw_localizations = []

    def generate(self, requests):
        responses = super().generate(requests)
        for request, response in zip(requests, responses, strict=True):
            if request.schema is models.Localization:
                self.localization_calls += 1
                if self.localization_calls == 1:
                    setattr(response.supports[0], self.field, "   ")
                self.raw_localizations.append(response.model_dump(mode="json"))
        return responses


@pytest.mark.parametrize("field", ["unit_id", "contribution"])
def test_workflow_whitespace_support_uses_bounded_retry_before_persistence(tmp_path, monkeypatch, field):
    config = fixture_config(tmp_path).model_copy(
        update={
            "summary_count": 2,
            "concurrency": 1,
            "batch_size": 30,
            "run_config": RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=0),
        }
    )
    inference = WhitespaceSupport(field)
    completion = install_inference(monkeypatch, inference)
    handoff = run_multimodal_sdg(config)
    assert handoff.is_file()
    assert inference.localization_calls == 3  # Two original rows and exactly one fresh retry.
    assert completion.await_count == 31  # The 30-call joint-localization fixture plus one retry.
    assert inference.raw_localizations[0]["supports"][0][field] == "   "
    candidates = json.loads((config.output_dir / "candidates.json").read_text())
    assert len(candidates) == 2 and all(candidate["accepted"] for candidate in candidates)
    assert all(support[field].strip() for candidate in candidates for support in candidate["localization"]["supports"])
    calls = completion.await_count
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    assert completion.await_count == calls


class InvalidQueries(CorpusInference):
    """Expose controlled schema failures without changing the Data Designer parser."""

    def __init__(self, fail_forever=False):
        super().__init__()
        self.query_calls = 0
        self.fail_forever = fail_forever

    def generate(self, requests):
        responses = super().generate(requests)
        for index, request in enumerate(requests):
            if issubclass(request.schema, models.QueryBatch):
                self.query_calls += 1
                if self.fail_forever or self.query_calls == 1:
                    responses[index] = models.QueryBatch(queries=[])
        return responses


@pytest.mark.parametrize(
    "fail_forever,overrides",
    [
        (False, {}),
        (True, {}),
        (True, {"max_conversation_restarts": 0, "max_conversation_correction_steps": 0}),
        (True, {"max_conversation_restarts": 1, "max_conversation_correction_steps": 0}),
        (True, {"max_conversation_restarts": 4, "max_conversation_correction_steps": 0}),
        (False, {"max_conversation_restarts": 0, "max_conversation_correction_steps": 1}),
        (True, {"max_conversation_restarts": 1, "max_conversation_correction_steps": 2}),
    ],
)
def test_workflow_native_retries_are_bounded_and_failed_rows_stop_export(
    tmp_path, monkeypatch, fail_forever, overrides
):
    run_config = RunConfig(**overrides)
    config = fixture_config(tmp_path).model_copy(
        update={"summary_count": 1, "concurrency": 1, "run_config": run_config}
    )
    inference = InvalidQueries(fail_forever)
    completion = install_inference(monkeypatch, inference)
    if fail_forever:
        with pytest.raises(DataDesignerGenerationError, match="could not be parsed"):
            run_multimodal_sdg(config)
        assert inference.query_calls == (
            (run_config.max_conversation_restarts + 1) * (run_config.max_conversation_correction_steps + 1)
        )
        assert not (config.output_dir / "complete.json").exists()
        assert not (config.output_dir / "bundle").exists()
    else:
        assert run_multimodal_sdg(config).is_file()
        assert inference.query_calls == 2
    query_messages = []
    for messages in completion.request_messages:
        if "QueryBatch3" in json.dumps(messages):
            query_messages.append(messages)
    assert len(query_messages) == inference.query_calls
    assert all(messages[0] == query_messages[0][0] for messages in query_messages)
    assert [len(messages) for messages in query_messages] == [
        1 + 2 * (index % (run_config.max_conversation_correction_steps + 1)) for index in range(inference.query_calls)
    ]


def test_completed_resume_rejects_changed_native_retry_settings(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 1})
    completion = install_inference(monkeypatch, CorpusInference())
    handoff = run_multimodal_sdg(config)
    calls = completion.await_count
    restored = models.MultimodalSDGConfig.model_validate_json(config.model_dump_json())
    restored.resume = True
    assert run_multimodal_sdg(restored) == handoff
    assert completion.await_count == calls
    restored.run_config.max_conversation_restarts += 1
    with pytest.raises(ValueError, match="Artifact changed"):
        run_multimodal_sdg(restored)
    assert completion.await_count == calls


def test_workflow_resume_rejects_changed_boundary_before_any_calls(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 1})
    sources = load_retrieval_sources(config.sources_file)
    contexts = load_contexts(config, sources)
    completion = install_inference(monkeypatch, SectionInference())
    generate_candidates(config, sources, contexts)
    path = next((config.output_dir / "data_designer").glob("multimodal/stage-*/boundary/state.json"))
    path.write_text("{}")
    completion.reset_mock()
    with pytest.raises(ValueError, match="integrity"):
        generate_candidates(config.model_copy(update={"resume": True}), sources, contexts)
    completion.assert_not_awaited()


def test_workflow_replans_queries_when_actual_judgment_needs_smaller_context(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"contexts_file": None, "judge_summaries": False})
    config.judge = config.judge.model_copy(update={"context_window_tokens": 400000})
    sources = [RetrievalSource(unit_id=str(index), document_id="doc", text="x" * 140000) for index in range(2)]
    context = models.GenerationContext(context_id="parent", unit_ids=["0", "1"])
    inference = CorpusInference()
    install_inference(monkeypatch, inference)
    candidates, outcomes = generate_candidates(config, sources, [context])
    assert outcomes[0]["selection_reason"] == "request_size_replanned"
    assert outcomes[0]["slots"]["queries"]
    assert [candidate.source_unit_ids for candidate in candidates] == [["0"], ["1"]]
    assert sum(issubclass(request.schema, models.QueryBatch) for request in inference.requests) == 3


class InterruptedQueries(CorpusInference):
    """Fail one workflow row after another has completed."""

    def __init__(self, failed_units=None, schema=models.QueryBatch):
        super().__init__()
        self.fail = True
        self.failed_units = {"unit-1"} if failed_units is None else failed_units
        self.schema = schema

    def generate(self, requests):
        if self.fail and any(
            issubclass(request.schema, self.schema) and any(f'"{unit}"' in request.text for unit in self.failed_units)
            for request in requests
        ):
            raise PermissionError("Controlled provider failure")
        return super().generate(requests)


@pytest.mark.parametrize("batch_size", [1, 30, 64])
def test_workflow_partial_stage_never_exports_missing_queries_and_can_resume(tmp_path, monkeypatch, batch_size):
    config = fixture_config(tmp_path).model_copy(
        update={"summary_count": 2, "concurrency": 1, "batch_size": batch_size}
    )
    (tmp_path / "page.png").rename(tmp_path / "page.bin")
    config.sources_file.write_text(config.sources_file.read_text().replace("page.png", "page.bin"))
    inference = InterruptedQueries()
    install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    assert not (config.output_dir / "complete.json").exists()
    summaries = sum(request.schema is models.ContextSummary for request in inference.requests)
    completed_queries = sum(issubclass(request.schema, models.QueryBatch) for request in inference.requests)
    boundary = config.output_dir / "data_designer/multimodal/stage-2-queries-0/boundary"
    abandoned = boundary / "images/.sdg-interrupted/content"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_bytes(b"partial alias write")
    saved = {path: digest(path) for path in config.output_dir.rglob("parquet-files/*.parquet")}
    inference.fail = False
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
    assert abandoned.read_bytes() == b"partial alias write"
    artifacts = json.loads((boundary / "integrity.json").read_text())
    assert all(".sdg-interrupted" not in record["path"] for record in artifacts)
    assert sum(request.schema is models.ContextSummary for request in inference.requests) == summaries
    assert sum(issubclass(request.schema, models.QueryBatch) for request in inference.requests) == completed_queries + 1
    assert all(digest(path) == checksum for path, checksum in saved.items())
    calls = len(inference.requests)
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
    assert len(inference.requests) == calls


def test_directory_workflow_resume_preserves_prepared_inputs_and_successful_rows(tmp_path, monkeypatch):
    """Resume a mixed directory run without regenerating successful native rows."""
    config = directory_config(tmp_path).model_copy(update={"summary_count": 3, "concurrency": 1, "batch_size": 1})
    inference = InterruptedQueries({"other.md#chunk-000001"})
    completion = install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    assert not (config.output_dir / "complete.json").exists()
    summaries = sum(request.schema is models.ContextSummary for request in inference.requests)
    queries = sum(issubclass(request.schema, models.QueryBatch) for request in inference.requests)
    assert (summaries, queries) == (3, 2)
    saved = {
        path: digest(path)
        for path in [
            *config.output_dir.rglob("parquet-files/*.parquet"),
            *(config.output_dir / "inputs").rglob("*"),
        ]
        if path.is_file()
    }
    manifest = json.loads((config.output_dir / "inputs/source_manifest.json").read_text())
    identity = json.loads((config.output_dir / "identity.json").read_text())
    assert identity["source_files"] == manifest["files"]
    inference.fail = False
    handoff = run_multimodal_sdg(config.model_copy(update={"resume": True}))
    assert handoff.is_file()
    assert sum(request.schema is models.ContextSummary for request in inference.requests) == summaries
    assert sum(issubclass(request.schema, models.QueryBatch) for request in inference.requests) == queries + 1
    assert all(digest(path) == checksum for path, checksum in saved.items())
    completion.reset_mock()
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    completion.assert_not_awaited()


@pytest.mark.parametrize("batch_size,buffer_size", [(2, None), (30, None), (30, 2), (2, 30)])
def test_workflow_repeated_recovery_preserves_successful_rows(tmp_path, monkeypatch, batch_size, buffer_size):
    config = fixture_config(tmp_path).model_copy(
        update={
            "summary_count": 4,
            "concurrency": 1,
            "batch_size": batch_size,
            "run_config": RunConfig() if buffer_size is None else RunConfig(buffer_size=buffer_size),
        }
    )
    inference = InterruptedQueries({"unit-1", "unit-2", "unit-3"})
    install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    saved = {path: digest(path) for path in config.output_dir.rglob("parquet-files/*.parquet")}
    # At size 2, group 1 remains wholly absent and must be retried by Data Designer
    # resume. At size 30, all missing identities belong to a finalized group.
    inference.failed_units = {"unit-2", "unit-3"} if (buffer_size or batch_size) == 2 else {"unit-3"}
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))
    assert all(digest(path) == checksum for path, checksum in saved.items())
    saved.update({path: digest(path) for path in config.output_dir.rglob("parquet-files/*.parquet")})
    inference.fail = False
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
    assert sum(issubclass(request.schema, models.QueryBatch) for request in inference.requests) == 4
    assert all(digest(path) == checksum for path, checksum in saved.items())
    assert len(json.loads((config.output_dir / "candidates.json").read_text())) == 4
    calls = len(inference.requests)
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
    assert len(inference.requests) == calls


def test_workflow_recovery_preserves_multicolumn_judgments_and_text_only_localization(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 2, "concurrency": 1, "batch_size": 30})
    rows = [json.loads(line) for line in config.sources_file.read_text().splitlines()]
    config.sources_file.write_text("".join(json.dumps({**row, "images": []}) + "\n" for row in rows))
    inference = InterruptedQueries(schema=models.Localization)
    install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    saved = {path: digest(path) for path in config.output_dir.rglob("parquet-files/*.parquet")}
    inference.fail = False
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
    assert all(digest(path) == checksum for path, checksum in saved.items())
    assert sum(request.schema is models.Localization for request in inference.requests) == 2
    assert all(not request.images for request in inference.requests if request.schema is models.Localization)
    assert all(row["accepted"] for row in json.loads((config.output_dir / "candidates.json").read_text()))


class InterruptRecoverySeed:
    """Interrupt after attempt creation, before an atomic seed is durable."""

    def __init__(self, writer):
        self.writer = writer

    def __call__(self, path, table):
        if "recovery" in path.parts:
            path.parent.mkdir(parents=True, exist_ok=True)
            raise KeyboardInterrupt("Controlled interruption before recovery seed publication")
        return self.writer(path, table)


def test_workflow_recovery_continues_after_interrupted_seed_publication(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 2, "concurrency": 1, "batch_size": 30})
    inference = InterruptedQueries()
    completion = install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    inference.fail = False
    completion.reset_mock()
    with monkeypatch.context() as scoped:
        scoped.setattr(execution, "write_parquet", InterruptRecoverySeed(execution.write_parquet))
        with pytest.raises(KeyboardInterrupt, match="seed publication"):
            run_multimodal_sdg(config.model_copy(update={"resume": True}))
    completion.assert_not_awaited()
    attempts = list(config.output_dir.rglob("recovery/attempt-*"))
    assert len(attempts) == 1 and not (attempts[0] / "seed.parquet").exists()
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
    assert attempts[0].is_dir()
    assert len(list(config.output_dir.rglob("recovery/attempt-*"))) == 2


class InterruptRecoveredBoundary:
    """Interrupt after durable recovery output but before boundary publication."""

    def __init__(self, reader):
        self.reader = reader

    def __call__(self, path, seed, stage, **kwargs):
        result = self.reader(path, seed, stage, **kwargs)
        if (path / "recovery").exists():
            raise KeyboardInterrupt("Controlled interruption after recovery checkpoint")
        return result


def test_workflow_recovery_reuses_checkpoint_after_boundary_interruption(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 2, "concurrency": 1, "batch_size": 30})
    inference = InterruptedQueries()
    install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    inference.fail = False
    with monkeypatch.context() as scoped:
        scoped.setattr(execution, "stage_responses", InterruptRecoveredBoundary(execution.stage_responses))
        with pytest.raises(KeyboardInterrupt, match="recovery checkpoint"):
            run_multimodal_sdg(config.model_copy(update={"resume": True}))
    saved = {path: digest(path) for path in config.output_dir.rglob("parquet-files/*.parquet")}
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
    assert sum(issubclass(request.schema, models.QueryBatch) for request in inference.requests) == 2
    assert all(digest(path) == checksum for path, checksum in saved.items())


@pytest.mark.parametrize("corruption", ["seed", "response", "identity"])
def test_workflow_recovery_rejects_corrupt_evidence_before_any_resume_calls(tmp_path, monkeypatch, corruption):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 4, "concurrency": 1, "batch_size": 2})
    inference = InterruptedQueries({"unit-1", "unit-2", "unit-3"})
    completion = install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    inference.failed_units = {"unit-2", "unit-3"}
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))
    attempt = next(config.output_dir.rglob("recovery/attempt-*"))
    if corruption == "seed":
        (attempt / "seed.parquet").unlink()
    else:
        path = next((attempt / "responses/parquet-files").glob("*.parquet"))
        rows = pq.read_table(path).to_pylist()
        rows[0]["response" if corruption == "response" else "_row_id"] = "{}" if corruption == "response" else 0
        pq.write_table(pa.Table.from_pylist(rows), path)
    inference.fail = False
    completion.reset_mock()
    with pytest.raises(ValueError):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))
    completion.assert_not_awaited()


@pytest.mark.parametrize("missing", ["workflow", "workflow_prefix", "unfinished_stage"])
def test_workflow_resume_preserves_saved_stages_without_metadata(tmp_path, monkeypatch, missing):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 2, "concurrency": 1, "batch_size": 1})
    inference = InterruptedQueries()
    completion = install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    workflow_root = config.output_dir / "data_designer/multimodal"
    metadata_path = workflow_root / "workflow-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    unfinished = next(stage for stage in metadata["stages"] if stage["status"] == "failed")
    stage_path = workflow_root / unfinished["stage_dir"]
    assert list((stage_path / "parquet-files").glob("*.parquet"))
    if missing == "workflow":
        affected = metadata["stages"][0]["stage_dir"]
        metadata_path.unlink()
    elif missing == "workflow_prefix":
        affected = metadata["stages"][1]["stage_dir"]
        metadata["stages"] = metadata["stages"][:1]
        metadata_path.write_text(json.dumps(metadata))
    else:
        affected = unfinished["stage_dir"]
        (stage_path / "metadata.json").unlink()
    saved = {path.relative_to(workflow_root): digest(path) for path in workflow_root.rglob("*") if path.is_file()}
    inference.fail = False
    completion.reset_mock()
    with pytest.raises(ValueError, match="metadata.*preserved") as error:
        run_multimodal_sdg(config.model_copy(update={"resume": True}))
    assert affected in str(error.value)
    completion.assert_not_awaited()
    assert {
        path.relative_to(workflow_root): digest(path) for path in workflow_root.rglob("*") if path.is_file()
    } == saved
    assert not (config.output_dir / "complete.json").exists()


def test_workflow_completed_stage_reuse_does_not_require_analysis_metadata(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 1})
    sources = load_retrieval_sources(config.sources_file)
    contexts = load_contexts(config, sources)
    completion = install_inference(monkeypatch, SectionInference())
    expected = generate_candidates(config, sources, contexts)
    workflow_root = config.output_dir / "data_designer/multimodal"
    next(workflow_root.glob("stage-*/metadata.json")).unlink()
    saved = {path.relative_to(workflow_root): digest(path) for path in workflow_root.rglob("*.parquet")}
    completion.reset_mock()
    assert generate_candidates(config.model_copy(update={"resume": True}), sources, contexts) == expected
    completion.assert_not_awaited()
    assert {path.relative_to(workflow_root): digest(path) for path in workflow_root.rglob("*.parquet")} == saved


class AbstainingInference(CorpusInference):
    """Return explicit unsupported slots, never fabricated query rows."""

    def generate(self, requests):
        responses = super().generate(requests)
        for response in responses:
            if isinstance(response, models.QueryBatch):
                for slot in response.queries:
                    slot.query, slot.evidence_modality = None, "none"
        return responses


def test_workflow_empty_query_stages_preserve_abstentions_without_judge_calls(tmp_path, monkeypatch):
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 1})
    inference = AbstainingInference()
    install_inference(monkeypatch, inference)
    sources = load_retrieval_sources(config.sources_file)
    candidates, outcomes = generate_candidates(config, sources, load_contexts(config, sources))
    assert not candidates
    assert sum(len(row["slots"]["queries"]) for row in outcomes) == 3
    assert not any(
        request.schema in (models.QueryMetadata, models.RelevanceJudgment)
        or issubclass(request.schema, models.Localization)
        for request in inference.requests
    )


def test_workflow_relative_output_resume_preserves_successful_stages(tmp_path, monkeypatch):
    """The documented relative run directory must resolve to one native checkpoint tree."""
    config = fixture_config(tmp_path).model_copy(
        update={"output_dir": Path("run"), "summary_count": 2, "concurrency": 1, "batch_size": 1}
    )
    monkeypatch.chdir(tmp_path)
    inference = InterruptedQueries()
    install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    summaries = sum(request.schema is models.ContextSummary for request in inference.requests)
    queries = sum(issubclass(request.schema, models.QueryBatch) for request in inference.requests)
    saved = {path: digest(path) for path in config.output_dir.rglob("parquet-files/*.parquet")}
    inference.fail = False
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
    assert sum(request.schema is models.ContextSummary for request in inference.requests) == summaries
    assert sum(issubclass(request.schema, models.QueryBatch) for request in inference.requests) == queries + 1
    assert all(digest(path) == checksum for path, checksum in saved.items())
    metadata = json.loads((config.output_dir / "data_designer/multimodal/workflow-metadata.json").read_text())
    assert metadata["stages"][0]["output_seed_path"] == "stage-0-summaries/boundary/seed.parquet"


@pytest.mark.parametrize(
    "corruption",
    [
        "workflow_name",
        "missing_fingerprint",
        "changed_fingerprint",
        "stage_name",
        "stage_order",
        "stage_config",
        "output_path",
        "dataset_target",
        "dataset_buffer",
    ],
)
def test_workflow_invalid_metadata_preserves_every_saved_artifact(tmp_path, monkeypatch, corruption):
    """Native IF_POSSIBLE must never reset evidence after a malformed progress record."""
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 2, "concurrency": 1, "batch_size": 1})
    inference = InterruptedQueries()
    completion = install_inference(monkeypatch, inference)
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        run_multimodal_sdg(config)
    workflow = config.output_dir / "data_designer/multimodal"
    path = workflow / "workflow-metadata.json"
    metadata = json.loads(path.read_text())
    if corruption == "workflow_name":
        metadata["name"] = "different-workflow"
    elif corruption == "missing_fingerprint":
        del metadata["stages"][0]["fingerprint"]
    elif corruption == "changed_fingerprint":
        metadata["stages"][1]["fingerprint"] = "0" * 64
    elif corruption == "stage_name":
        metadata["stages"][1]["name"] = "different-stage"
    elif corruption == "stage_order":
        metadata["stages"][0], metadata["stages"][1] = metadata["stages"][1], metadata["stages"][0]
    elif corruption == "stage_config":
        metadata["stages"][1]["config"]["model_configs"][0]["model"] = "different/model"
    elif corruption == "output_path":
        metadata["stages"][0]["output_seed_path"] = "nonexistent.parquet"
    else:
        failed = next(stage for stage in metadata["stages"] if stage["status"] == "failed")
        path = workflow / failed["stage_dir"] / "metadata.json"
        metadata = json.loads(path.read_text())
        if corruption == "dataset_target":
            del metadata["target_num_records"]
        else:
            metadata["buffer_size"] += 1
    path.write_text(json.dumps(metadata))
    saved = {path: digest(path) for path in config.output_dir.rglob("*") if path.is_file()}
    inference.fail = False
    completion.reset_mock()
    with pytest.raises(ValueError, match="metadata.*preserved"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))
    completion.assert_not_awaited()
    assert {path: digest(path) for path in config.output_dir.rglob("*") if path.is_file()} == saved


@pytest.mark.parametrize(
    "corruption",
    ["state", "response", "missing_manifest", "empty_manifest", "missing_native_tree", "truncated_metadata"],
)
def test_completed_run_resume_verifies_native_stage_evidence(tmp_path, monkeypatch, corruption):
    """A complete portable bundle does not bypass checks on its native generation evidence."""
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 1})
    completion = install_inference(monkeypatch, SectionInference())
    assert run_multimodal_sdg(config).is_file()
    workflow = config.output_dir / "data_designer/multimodal"
    stage = next(workflow.glob("stage-*"))
    if corruption == "state":
        (stage / "boundary/state.json").write_text("{}")
    elif corruption == "response":
        response = next((stage / "parquet-files").glob("*.parquet"))
        response.write_bytes(response.read_bytes() + b"changed")
    elif corruption == "missing_manifest":
        (stage / "boundary/integrity.json").unlink()
    elif corruption == "missing_native_tree":
        (config.output_dir / "data_designer").rename(tmp_path / "saved-native-evidence")
    elif corruption == "truncated_metadata":
        metadata_path = workflow / "workflow-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["stages"] = metadata["stages"][:1]
        metadata_path.write_text(json.dumps(metadata))
        for path in workflow.glob("stage-*"):
            if path.name != metadata["stages"][0]["stage_dir"]:
                path.rename(tmp_path / ("saved-" + path.name))
    else:
        (stage / "boundary/integrity.json").write_text("[]")
    saved = {path: digest(path) for path in config.output_dir.rglob("*") if path.is_file()}
    completion.reset_mock()
    with pytest.raises(ValueError, match="integrity"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))
    completion.assert_not_awaited()
    assert {path: digest(path) for path in config.output_dir.rglob("*") if path.is_file()} == saved


@pytest.mark.parametrize(
    "corruption",
    ["state", "seed", "image_alias", "missing_manifest", "missing_inputs", "frozen_image", "missing_frozen_image"],
)
def test_completed_run_resume_verifies_initial_generation_inputs(tmp_path, monkeypatch, corruption):
    """Initial seeds, state and image aliases remain checked after bundle publication."""
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 1})
    if corruption not in {"frozen_image", "missing_frozen_image"}:
        image = tmp_path / "page.png"
        image.rename(image.with_suffix(".bin"))
        sources = [json.loads(line) for line in config.sources_file.read_text().splitlines()]
        for source in sources:
            source["images"] = ["page.bin"]
        config.sources_file.write_text("".join(json.dumps(source) + "\n" for source in sources))
    completion = install_inference(monkeypatch, SectionInference())
    assert run_multimodal_sdg(config).is_file()
    inputs = next((config.output_dir / "data_designer/inputs").iterdir())
    if corruption in {"frozen_image", "missing_frozen_image"}:
        image = next((config.output_dir / "inputs/assets").iterdir())
        if corruption == "frozen_image":
            image.write_bytes(image.read_bytes() + b"changed")
        else:
            image.unlink()
    elif corruption == "missing_inputs":
        inputs.rename(tmp_path / "saved-inputs")
    elif corruption == "missing_manifest":
        (inputs / "integrity.json").unlink()
    else:
        path = (
            next((inputs / "images").iterdir())
            if corruption == "image_alias"
            else inputs / ("state.json" if corruption == "state" else "seed.parquet")
        )
        path.write_bytes(path.read_bytes() + b"changed")
    saved = {path: digest(path) for path in config.output_dir.rglob("*") if path.is_file()}
    completion.reset_mock()
    with pytest.raises(ValueError, match="integrity"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))
    completion.assert_not_awaited()
    assert {path: digest(path) for path in config.output_dir.rglob("*") if path.is_file()} == saved


def test_workflow_resumes_before_first_native_checkpoint(tmp_path, monkeypatch):
    """An interruption after input publication needs no nonexistent stage metadata."""
    config = fixture_config(tmp_path).model_copy(update={"summary_count": 1})
    completion = install_inference(monkeypatch, SectionInference())
    with monkeypatch.context() as scoped:
        scoped.setattr(
            execution.CompositeWorkflow, "run", Mock(side_effect=KeyboardInterrupt("Before native workflow start"))
        )
        with pytest.raises(KeyboardInterrupt, match="Before native workflow start"):
            run_multimodal_sdg(config)
    completion.assert_not_awaited()
    assert list((config.output_dir / "data_designer/inputs").glob("*/seed.parquet"))
    assert not (config.output_dir / "data_designer/multimodal/workflow-metadata.json").exists()
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
