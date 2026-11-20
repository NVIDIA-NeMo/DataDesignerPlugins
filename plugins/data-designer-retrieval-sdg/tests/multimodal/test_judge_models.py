# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage routing, native transport, and provenance using offline model responses."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.testing import make_stub_completion_response
from pydantic import ValidationError
from test_multimodal_sdg import ScriptedInference, fixture_config
from test_multimodal_sections import SectionInference
from workflow_test_utils import ScriptedTransport, generate_candidates, install_inference

from data_designer_retrieval_sdg.multimodal import JudgeModels, MultimodalSDGConfig, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.bounds import RequestTooLarge, check_request, validate_deployment
from data_designer_retrieval_sdg.multimodal.inference import Request
from data_designer_retrieval_sdg.multimodal.models import ContextSummary, TOCConfirmation
from data_designer_retrieval_sdg.multimodal.source_file import load_retrieval_sources
from data_designer_retrieval_sdg.multimodal.storage import digest
from data_designer_retrieval_sdg.multimodal.workflow import load_contexts


class RecordingCompletion:
    """Record native facade settings and return a valid schema response offline."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.transport = ScriptedTransport(SectionInference())

    async def complete(self, facade: ModelFacade, messages, **kwargs):
        self.calls.append(
            {
                "alias": facade.model_alias,
                "model": facade.model_name,
                "provider": facade.model_provider,
                "parameters": facade.consolidate_kwargs(),
                "timeout": facade.request_timeout,
            }
        )
        return await self.transport.complete(facade, messages, **kwargs)


class TOCRecordingCompletion:
    """Return native workflow responses and record the model confirming TOCs."""

    def __init__(self) -> None:
        self.transport = ScriptedTransport(SectionInference())
        self.toc_models: list[str] = []

    async def complete(self, facade: ModelFacade, messages, **kwargs):
        """Handle TOC confirmation and delegate other schemas to the test transport."""
        content = messages[0].to_dict()["content"]
        text = (
            "\n".join(item["text"] for item in content if item["type"] == "text")
            if isinstance(content, list)
            else content
        )
        schema = json.loads(text.split("<response_schema>\n")[1].split("\n</response_schema>")[0])
        if schema["title"] == "TOCConfirmation":
            self.toc_models.append(facade.model_name)
            response = TOCConfirmation(has_table_of_contents=True, explanation="Three contents entries")
            return make_stub_completion_response(content=response.model_dump_json())
        return await self.transport.complete(facade, messages, **kwargs)


def toc_judge_config(tmp_path: Path) -> MultimodalSDGConfig:
    """Use all five overrides with a separate text-only fallback and a real TOC candidate."""
    config = fixture_config(tmp_path).model_copy(
        update={"contexts_file": None, "context_strategy": "sections", "section_size": 1}
    )
    sources = [json.loads(line) for line in config.sources_file.read_text().splitlines()]
    sources[0]["text"] = "Table of contents\nAlpha ... 1\nBeta ... 2\nGamma ... 3"
    config.sources_file.write_text("".join(json.dumps(source) + "\n" for source in sources))
    tokenizer = tmp_path / "toc-tokenizer.json"
    tokenizer.write_bytes(config.generator.tokenizer_file.read_bytes())
    config.judge = config.judge.model_copy(
        update={"model": "operator/toc-judge", "tokenizer_file": tokenizer, "image_tokens_per_image": None}
    )
    config.judges = JudgeModels(**dict.fromkeys(JudgeModels.model_fields, config.generator))
    return config


@pytest.mark.parametrize("judge_summaries", [True, False])
def test_automatic_toc_judge_is_validated_before_creating_run(tmp_path: Path, judge_summaries: bool) -> None:
    config = toc_judge_config(tmp_path)
    config.judge_summaries = judge_summaries
    config.judge.tokenizer_file = None
    with pytest.raises(ValueError, match="Configure"):
        run_multimodal_sdg(config)
    assert not config.output_dir.exists()


def test_automatic_toc_judge_tokenizer_is_frozen_for_resume(tmp_path: Path, monkeypatch) -> None:
    config = toc_judge_config(tmp_path)
    monkeypatch.setenv("TEST_SDG_KEY", "test-placeholder")
    recorder = TOCRecordingCompletion()
    with patch.object(ModelFacade, "acompletion", autospec=True, side_effect=recorder.complete) as completion:
        handoff = run_multimodal_sdg(config)
        assert recorder.toc_models == [config.judge.model]
        identity = json.loads((config.output_dir / "identity.json").read_text())
        assert identity["tokenizers"]["judge"] == digest(config.judge.tokenizer_file)
        calls = completion.await_count
        assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
        assert completion.await_count == calls
        config.judge.tokenizer_file.write_text(config.judge.tokenizer_file.read_text() + "\n")
        with pytest.raises(ValueError, match="Artifact changed"):
            run_multimodal_sdg(config.model_copy(update={"resume": True}))
        assert completion.await_count == calls


def test_automatic_toc_judge_prevents_false_shared_model_flag(tmp_path: Path, monkeypatch) -> None:
    config = toc_judge_config(tmp_path)
    monkeypatch.setenv("TEST_SDG_KEY", "test-placeholder")
    recorder = TOCRecordingCompletion()
    with patch.object(ModelFacade, "acompletion", autospec=True, side_effect=recorder.complete):
        handoff = run_multimodal_sdg(config)
    assert recorder.toc_models == [config.judge.model]
    manifest = json.loads((handoff.parent / "run_manifest.json").read_text())
    assert not manifest["shared_generator_judge"]
    assert manifest["models"]["judge"]["model"] == config.judge.model
    assert all(model["model"] == config.generator.model for model in manifest["models"]["judges"].values())


@pytest.mark.parametrize("strategy,explicit", [("unit", False), ("document", False), ("sections", True)])
def test_unused_fallback_does_not_affect_preflight_or_provenance(
    tmp_path: Path, monkeypatch, strategy: str, explicit: bool
) -> None:
    config = toc_judge_config(tmp_path)
    config.context_strategy = strategy
    config.contexts_file = tmp_path / "contexts.jsonl" if explicit else None
    config.judge.tokenizer_file = None
    monkeypatch.setenv("TEST_SDG_KEY", "test-placeholder")
    recorder = TOCRecordingCompletion()
    with patch.object(ModelFacade, "acompletion", autospec=True, side_effect=recorder.complete):
        handoff = run_multimodal_sdg(config)
    assert not recorder.toc_models
    identity = json.loads((config.output_dir / "identity.json").read_text())
    assert "judge" not in identity["tokenizers"]
    manifest = json.loads((handoff.parent / "run_manifest.json").read_text())
    assert manifest["shared_generator_judge"]


def test_complete_stage_settings_replace_fallback_and_reject_invalid_keys(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    for stage in JudgeModels.model_fields:
        assert config.model_for("judge", stage) == config.judge
    payload = config.judge.model_dump(include={"model", "endpoint", "credential_env"})
    payload["model"] = "text-judge"
    config.judge.temperature = 0.1
    config.judges = JudgeModels(query_metadata=payload)
    assert config.model_for("judge", "query_metadata").temperature == 0.6  # Its own ModelSettings default.
    assert config.model_for("judge", "relevance") == config.judge
    assert config.model_for("generator") == config.generator
    with pytest.raises(ValidationError):
        JudgeModels(query_metadata={"model": "partial-settings"})
    with pytest.raises(ValidationError):
        JudgeModels(relevancy=payload)
    with pytest.raises(ValueError, match="Invalid judge stage"):
        config.model_for("judge", "unknown")
    with pytest.raises(ValueError, match="Invalid judge stage"):
        config.model_for("generator", "summary")


def test_every_workflow_judge_request_names_its_stage(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    sources = load_retrieval_sources(config.sources_file)
    inference = ScriptedInference()
    generate_candidates(config, sources, load_contexts(config, sources), inference)
    stages = {request.judge_stage for request in inference.requests if request.role == "judge"}
    assert stages == set(JudgeModels.model_fields)
    for request in inference.requests:
        if request.judge_stage in ("summary", "query_metadata", "self_sufficiency"):
            assert not request.images
        elif request.role == "judge":
            assert request.images
        else:
            assert request.judge_stage is None


def test_section_planning_summary_uses_summary_judge(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    sources = load_retrieval_sources(config.sources_file)
    inference = SectionInference()
    config.context_strategy = "sections"
    generate_candidates(config, sources, load_contexts(config, sources), inference)
    judges = [request for request in inference.requests if request.role == "judge"]
    assert any(request.judge_stage == "summary" for request in judges)
    assert all(request.judge_stage == "summary" for request in judges if request.schema.__name__ == "SummaryJudgment")


def test_native_columns_route_full_settings_and_resume_without_calls(tmp_path: Path, monkeypatch) -> None:
    config = fixture_config(tmp_path)
    monkeypatch.setenv("TEST_SDG_KEY", "test-placeholder")
    monkeypatch.setenv("TEST_JUDGE_KEY", "judge-placeholder")
    config.judges = JudgeModels(
        **{
            stage: config.judge.model_copy(
                update={
                    "model": "operator/" + stage,
                    "endpoint": "https://" + stage.replace("_", "-") + ".example.invalid/v1",
                    "credential_env": "TEST_JUDGE_KEY",
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "timeout": 123,
                    "extra_body": {"response_format": {"type": "json_object"}},
                }
            )
            for stage in JudgeModels.model_fields
        }
    )
    recorder = RecordingCompletion()
    with patch.object(ModelFacade, "acompletion", autospec=True, side_effect=recorder.complete):
        handoff = run_multimodal_sdg(config)
        calls = len(recorder.calls)
        assert calls > 0
        assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
        assert len(recorder.calls) == calls
        config.judges.query_metadata = config.judges.query_metadata.model_copy(update={"model": "replacement"})
        with pytest.raises(ValueError, match="Artifact changed"):
            run_multimodal_sdg(config.model_copy(update={"resume": True}))
        assert len(recorder.calls) == calls
    config.judges.query_metadata = config.judges.query_metadata.model_copy(update={"model": "operator/query_metadata"})
    assert {call["alias"] for call in recorder.calls} == {"generator", *JudgeModels.model_fields}
    for routed in recorder.calls:
        model = config.generator if routed["alias"] == "generator" else getattr(config.judges, routed["alias"])
        assert routed["model"] == model.model
        assert routed["provider"].endpoint == model.endpoint
        assert routed["provider"].api_key == model.credential_env
        assert routed["parameters"]["temperature"] == model.temperature
        assert routed["parameters"]["max_tokens"] == model.max_tokens
        assert routed["parameters"].get("extra_body", {}) == model.extra_body
        assert routed["timeout"] == model.timeout


def test_unconfigured_native_columns_use_fallback_judge(tmp_path: Path, monkeypatch) -> None:
    config = fixture_config(tmp_path)
    config.judge = config.judge.model_copy(update={"model": "operator/default-judge"})
    monkeypatch.setenv("TEST_SDG_KEY", "test-placeholder")
    recorder = RecordingCompletion()
    with patch.object(ModelFacade, "acompletion", autospec=True, side_effect=recorder.complete):
        assert run_multimodal_sdg(config).is_file()
    judges = [call for call in recorder.calls if call["alias"] != "generator"]
    assert {call["alias"] for call in judges} == set(JudgeModels.model_fields)
    assert all(call["model"] == config.judge.model for call in judges)


def test_text_judges_need_no_image_allowance_but_vision_routes_do(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    text = config.judge.model_copy(update={"image_tokens_per_image": None})
    vision = config.judge.model_copy(update={"model": "vision-judge"})
    config.judge = text
    config.judges = JudgeModels(relevance=vision, localization=vision)
    validate_deployment(config, has_images=True)
    check_request(Request("Judge", ContextSummary, "judge", judge_stage="query_metadata"), config)
    config.judges.localization = text
    with pytest.raises(ValueError, match="image_tokens_per_image for localization"):
        validate_deployment(config, has_images=True)
    with pytest.raises(ValueError, match="image_tokens_per_image"):
        check_request(Request("Judge", ContextSummary, "judge", (tmp_path / "page.png",), "localization"), config)


def test_stage_token_budget_and_tokenizer_control_preflight(tmp_path: Path) -> None:
    from tokenizers import Tokenizer

    config = fixture_config(tmp_path)
    judge_tokenizer = tmp_path / "judge-tokenizer.json"
    judge_tokenizer.write_bytes(config.judge.tokenizer_file.read_bytes())
    config.judges = JudgeModels(
        query_metadata=config.judge.model_copy(
            update={"tokenizer_file": judge_tokenizer, "context_window_tokens": 9000}
        )
    )
    request = Request("x " * 1000, ContextSummary, "judge", judge_stage="query_metadata")
    check_request(Request(request.text, ContextSummary, "judge", judge_stage="summary"), config)
    with pytest.raises(RequestTooLarge, match="query_metadata request needs"):
        check_request(request, config)
    encoder = Tokenizer.from_file(str(judge_tokenizer))
    encoder.enable_truncation(max_length=1)
    encoder.save(str(judge_tokenizer))
    with pytest.raises(RequestTooLarge):
        check_request(request, config)


def test_invalid_active_override_fails_before_creating_run(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    config.judges = JudgeModels(summary=config.judge.model_copy(update={"tokenizer_file": None}))
    with pytest.raises(ValueError, match="Configure"):
        run_multimodal_sdg(config)
    assert not config.output_dir.exists()
    config.judge_summaries = False
    validate_deployment(config, has_images=True)


def test_export_and_resume_record_resolved_models_and_override_tokenizers(tmp_path: Path, monkeypatch) -> None:
    config = fixture_config(tmp_path)
    judge_tokenizer = tmp_path / "judge-tokenizer.json"
    judge_tokenizer.write_bytes(config.judge.tokenizer_file.read_bytes())
    config.judges = JudgeModels(
        query_metadata=config.judge.model_copy(update={"model": "small-judge", "tokenizer_file": judge_tokenizer})
    )
    install_inference(monkeypatch, ScriptedInference())
    handoff = run_multimodal_sdg(config)
    manifest = json.loads((handoff.parent / "run_manifest.json").read_text())
    assert manifest["models"]["judge"]["model"] == config.judge.model
    assert manifest["models"]["judges"]["query_metadata"]["model"] == "small-judge"
    assert manifest["models"]["judges"]["localization"]["model"] == config.judge.model
    assert not manifest["shared_generator_judge"]
    identity = json.loads((config.output_dir / "identity.json").read_text())
    assert identity["tokenizers"]["query_metadata"] == digest(judge_tokenizer)
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})) == handoff
    judge_tokenizer.write_text(judge_tokenizer.read_text() + "\n")
    with pytest.raises(ValueError, match="Artifact changed"):
        run_multimodal_sdg(config.model_copy(update={"resume": True}))


def test_documented_stage_configuration_validates() -> None:
    docs = Path(__file__).parents[2] / "docs/multimodal-reference.md"
    section = docs.read_text().split("### Judge Model Overrides", 1)[1]
    example = section.split("```yaml\n", 1)[1].split("```", 1)[0]
    config = MultimodalSDGConfig.model_validate(yaml.safe_load(example))
    assert config.model_for("judge", "summary").model == "your-text-judge-model"
    assert config.model_for("judge", "localization").model == "your-vision-judge-model"
