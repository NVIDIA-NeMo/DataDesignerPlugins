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

from data_designer_retrieval_sdg.multimodal import JudgeModels, MultimodalSDGConfig, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.bounds import RequestTooLarge, check_request, validate_deployment
from data_designer_retrieval_sdg.multimodal.inference import DataDesignerInference, Request
from data_designer_retrieval_sdg.multimodal.models import ContextSummary, GenerationContext
from data_designer_retrieval_sdg.multimodal.source_file import load_retrieval_sources
from data_designer_retrieval_sdg.multimodal.storage import digest
from data_designer_retrieval_sdg.multimodal.summary_stages import plan_summaries
from data_designer_retrieval_sdg.multimodal.workflow import generate_candidates, load_contexts


class RecordingCompletion:
    """Record native facade settings and return a valid schema response offline."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(self, facade: ModelFacade, *args, **kwargs):
        self.calls.append(
            {
                "model": facade.model_name,
                "provider": facade.model_provider,
                "parameters": facade.consolidate_kwargs(),
                "timeout": facade.request_timeout,
            }
        )
        return make_stub_completion_response(content=json.dumps({"summary": facade.model_name, "visual_evidence": ""}))


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
    from test_multimodal_sections import SectionInference

    config = fixture_config(tmp_path)
    sources = load_retrieval_sources(config.sources_file)
    inference = SectionInference()
    plan_summaries(sources, [GenerationContext(context_id="page", unit_ids=[sources[0].unit_id])], config, inference)
    judges = [request for request in inference.requests if request.role == "judge"]
    assert judges and all(request.judge_stage == "summary" for request in judges)


def test_native_batches_route_full_settings_and_selectively_reuse_cache(tmp_path: Path, monkeypatch) -> None:
    config = fixture_config(tmp_path)
    monkeypatch.setenv("TEST_SDG_KEY", "test-placeholder")
    monkeypatch.setenv("TEST_JUDGE_KEY", "judge-placeholder")
    override = config.judge.model_copy(
        update={
            "model": "operator/small-text-judge",
            "endpoint": "https://judge.example.invalid/v1",
            "credential_env": "TEST_JUDGE_KEY",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout": 123,
            "extra_body": {"response_format": {"type": "json_object"}},
        }
    )
    config.judges = JudgeModels(query_metadata=override)
    # Matching schemas and roles must still use separate native model/provider batches.
    requests = [
        Request("Summarize", ContextSummary, "judge", judge_stage=stage) for stage in ("summary", "query_metadata")
    ]
    recorder = RecordingCompletion()
    inference = DataDesignerInference(tmp_path / "inference", config)
    with patch.object(ModelFacade, "acompletion", autospec=True, side_effect=recorder.complete):
        responses = inference.generate([*requests, requests[1]])
        assert [r.summary for r in responses] == [config.judge.model, override.model, override.model]
        assert len(recorder.calls) == 2
        assert inference.generate(requests) == responses[:2]
        assert len(recorder.calls) == 2
        config.judges.query_metadata = override.model_copy(update={"model": "operator/replacement"})
        assert [r.summary for r in inference.generate(requests)] == [config.judge.model, "operator/replacement"]
        assert len(recorder.calls) == 3  # The unaffected summary judgment remains cached.
    routed = recorder.calls[1]
    assert routed["provider"].endpoint == override.endpoint
    assert routed["provider"].api_key == override.credential_env
    assert routed["parameters"]["temperature"] == override.temperature
    assert routed["parameters"]["max_tokens"] == override.max_tokens
    assert routed["parameters"]["extra_body"] == override.extra_body
    assert routed["timeout"] == override.timeout


def test_unconfigured_stages_keep_original_judge_cache_keys(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    inference = DataDesignerInference(tmp_path / "inference", config)
    legacy = Request("Judge", ContextSummary, "judge")
    for stage in JudgeModels.model_fields:
        assert inference.request_key(
            Request("Judge", ContextSummary, "judge", judge_stage=stage)
        ) == inference.request_key(legacy)
    config.judges = JudgeModels(summary=config.judge)
    assert inference.request_key(
        Request("Judge", ContextSummary, "judge", judge_stage="summary")
    ) == inference.request_key(legacy)


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


def test_stage_token_budget_and_tokenizer_control_preflight_and_cache(tmp_path: Path) -> None:
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
    inference = DataDesignerInference(tmp_path / "inference", config)
    before = inference.request_key(request)
    encoder = Tokenizer.from_file(str(judge_tokenizer))
    encoder.enable_truncation(max_length=1)
    encoder.save(str(judge_tokenizer))
    assert inference.request_key(request) != before
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
    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedInference)
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
