# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real native readiness and workflow paths with only model transport replaced."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from data_designer.engine.errors import SecretResolutionError
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
from data_designer.engine.models.clients.types import EmbeddingResponse
from data_designer.engine.models.errors import ModelAPIConnectionError, ModelAuthenticationError, ModelNotFoundError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.testing import make_stub_completion_response
from test_failed_records import ControlledTransport, skip_config

from data_designer_retrieval_sdg.multimodal import JudgeModels, MultimodalSDGConfig, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.preflight import check_models


class ReadinessTransport(ControlledTransport):
    """Record native probes separately from real structured generation requests."""

    def __init__(self):
        super().__init__(())
        self.probes = []
        self.failing_model = None
        self.failure_kind = ProviderErrorKind.AUTHENTICATION

    async def complete(self, facade, messages, **kwargs):
        """Return a basic response or exercise native provider error translation."""
        if kwargs.get("skip_usage_tracking"):
            assert messages[-1].to_dict()["content"] == [{"type": "text", "text": "Hello!"}]
            self.probes.append(facade)
            if facade.model_name == self.failing_model:
                raise ProviderError(self.failure_kind, "Controlled readiness failure")
            return make_stub_completion_response(content="Hello!")
        return await super().complete(messages, **kwargs)


@pytest.fixture
def transport(monkeypatch, offline_model_credential):
    """Run native checks while keeping both probes and generation offline."""
    monkeypatch.delenv("DATA_DESIGNER_SKIP_MODEL_HEALTH_CHECKS", raising=False)
    transport = ReadinessTransport()
    with patch.object(ModelFacade, "acompletion", autospec=True, side_effect=transport.complete):
        yield transport


def stage_config(tmp_path: Path, stage: str, **settings) -> MultimodalSDGConfig:
    """Isolate one deployment while every other stage uses the valid fixture."""
    config = skip_config(tmp_path, "unit")
    config.judges = JudgeModels(**dict.fromkeys(JudgeModels.model_fields, config.generator))
    model = config.generator.model_copy(update={"model": f"operator/{stage}", **settings})
    if stage == "generator":
        config.generator = model
    elif stage == "toc_confirmation":
        config.context_strategy = "sections"
        config.contexts_file = None
        config.judge = model
    else:
        setattr(config.judges, stage, model)
    return config


@pytest.mark.parametrize("stage", ["generator", *JudgeModels.model_fields, "toc_confirmation"])
def test_active_model_failure_precedes_generation_and_run_creation(tmp_path, transport, stage):
    config = stage_config(tmp_path, stage)
    transport.failing_model = f"operator/{stage}"
    with pytest.raises(ModelAuthenticationError, match=f"operator/{stage}"):
        run_multimodal_sdg(config)
    assert not transport.calls
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    "kind,error",
    [(ProviderErrorKind.NOT_FOUND, ModelNotFoundError), (ProviderErrorKind.API_CONNECTION, ModelAPIConnectionError)],
)
def test_native_provider_diagnostics_propagate(tmp_path, transport, kind, error):
    config = skip_config(tmp_path, "unit")
    transport.failing_model = config.generator.model
    transport.failure_kind = kind
    with pytest.raises(error, match="running health checks"):
        run_multimodal_sdg(config)
    assert not transport.calls
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"endpoint": "https://second.example.invalid/v1"},
        {"credential_env": "SECOND_TEST_KEY"},
        {"temperature": 0.1},
        {"max_tokens": 1234},
        {"timeout": 321},
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ],
)
def test_distinct_serving_settings_are_probed_without_collapsing_them(tmp_path, monkeypatch, transport, settings):
    config = stage_config(tmp_path, "relevance", model="operator/vlm", **settings)
    monkeypatch.setenv("SECOND_TEST_KEY", "offline-placeholder")
    check_models(config)
    assert len(transport.probes) == 2
    probe = next(model for model in transport.probes if model.model_alias == "relevance")
    expected = config.judges.relevance
    assert probe.model_provider.endpoint == expected.endpoint
    assert probe.model_provider.api_key == expected.credential_env
    params = probe.consolidate_kwargs()
    assert params["temperature"] == expected.temperature
    assert params["max_tokens"] == expected.max_tokens
    assert params["timeout"] == expected.timeout
    assert params.get("extra_body", {}) == expected.extra_body
    assert not config.output_dir.exists()


def test_shared_deployment_is_checked_once_and_not_repeated_by_batches(tmp_path, transport):
    config = stage_config(tmp_path, "summary", model="operator/vlm", request_overhead_tokens=64)
    assert run_multimodal_sdg(config).is_file()
    assert len(transport.probes) == 1
    assert transport.calls


def test_unused_judges_are_not_probed(tmp_path, monkeypatch, transport):
    config = stage_config(tmp_path, "summary", credential_env="UNUSED_PREFLIGHT_KEY")
    config.judge_summaries = False
    config.judge = config.judges.summary
    config.context_strategy = "sections"  # Explicit contexts bypass the fallback TOC judge.
    monkeypatch.delenv("UNUSED_PREFLIGHT_KEY", raising=False)
    assert run_multimodal_sdg(config).is_file()
    assert [model.model_name for model in transport.probes] == [config.generator.model]
    assert not any(schema in {"SummaryJudgment", "TOCConfirmation"} for schema, _ in transport.calls)


def test_missing_credential_can_be_fixed_without_changing_configuration(tmp_path, monkeypatch, transport):
    config = stage_config(tmp_path, "self_sufficiency", credential_env="RECOVER_PREFLIGHT_KEY")
    monkeypatch.delenv("RECOVER_PREFLIGHT_KEY", raising=False)
    with pytest.raises(SecretResolutionError, match="RECOVER_PREFLIGHT_KEY"):
        run_multimodal_sdg(config)
    assert not transport.calls
    assert not config.output_dir.exists()
    monkeypatch.setenv("RECOVER_PREFLIGHT_KEY", "offline-placeholder")
    assert run_multimodal_sdg(config).is_file()


def test_completed_resume_needs_no_credentials_and_still_checks_integrity(tmp_path, monkeypatch, transport):
    config = skip_config(tmp_path, "unit")
    handoff = run_multimodal_sdg(config)
    calls = transport.calls.copy()
    monkeypatch.delenv("TEST_SDG_KEY", raising=False)
    resumed = config.model_copy(update={"resume": True})
    assert run_multimodal_sdg(resumed) == handoff
    (handoff.parent / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="Completed run integrity failure"):
        run_multimodal_sdg(resumed)
    assert len(transport.probes) == 1
    assert transport.calls == calls


def test_incomplete_resume_checks_models_and_preserves_cached_generation(tmp_path, transport):
    config = skip_config(tmp_path, "unit")
    handoff = run_multimodal_sdg(config)
    (config.output_dir / "complete.json").unlink()
    identity = (config.output_dir / "identity.json").read_bytes()
    calls = transport.calls.copy()
    resumed = config.model_copy(update={"resume": True})
    transport.failing_model = config.generator.model
    with pytest.raises(ModelAuthenticationError):
        run_multimodal_sdg(resumed)
    assert (config.output_dir / "identity.json").read_bytes() == identity
    transport.failing_model = None
    assert run_multimodal_sdg(resumed) == handoff
    assert len(transport.probes) == 3
    assert transport.calls == calls


def test_existing_or_incompatible_runs_fail_before_probes(tmp_path, transport):
    config = skip_config(tmp_path, "unit")
    run_multimodal_sdg(config)
    with pytest.raises(FileExistsError, match="Run exists"):
        run_multimodal_sdg(config)
    with pytest.raises(ValueError, match="Artifact changed"):
        run_multimodal_sdg(config.model_copy(update={"resume": True, "persona": "Changed persona"}))
    assert len(transport.probes) == 1


def test_native_health_check_opt_out_is_honored(tmp_path, monkeypatch, transport):
    monkeypatch.setenv("DATA_DESIGNER_SKIP_MODEL_HEALTH_CHECKS", "1")
    assert run_multimodal_sdg(skip_config(tmp_path, "unit")).is_file()
    assert not transport.probes
    assert transport.calls


def test_hosted_summary_embeddings_use_native_embedding_probe(tmp_path, transport):
    config = skip_config(tmp_path, "unit")
    config.combination_iterations = 1
    config.summary_embedding_endpoint = "https://embeddings.example.invalid/v1"
    config.summary_embedding_model = "operator/embed"
    config.summary_embedding_credential_env = "TEST_SDG_KEY"
    config.summary_embedding_extra_body = {"input_type": "passage", "truncate": "NONE"}
    with patch.object(
        ModelRequestExecutor, "aembeddings", AsyncMock(return_value=EmbeddingResponse(vectors=[[0.5, 1.0]]))
    ) as embedding:
        check_models(config)
    request = embedding.call_args.args[0]
    assert request.model == config.summary_embedding_model
    assert request.inputs == ["Hello!"]
    assert request.encoding_format == "float"
    assert request.extra_body == config.summary_embedding_extra_body
    assert not config.output_dir.exists()


@pytest.mark.parametrize("hosted", [False, True])
def test_inactive_or_local_embeddings_do_not_make_remote_probes(tmp_path, transport, hosted):
    config = skip_config(tmp_path, "unit")
    config.combination_iterations = 0 if hosted else 1
    config.summary_embedding_endpoint = "https://unused.example.invalid/v1" if hosted else None
    with patch.object(ModelRequestExecutor, "aembeddings", AsyncMock()) as embedding:
        check_models(config)
    embedding.assert_not_called()
    assert len(transport.probes) == 1
