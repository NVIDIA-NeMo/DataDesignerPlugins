# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Missing credentials fail before model calls or immutable run creation."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from data_designer.engine.models.facade import ModelFacade
from test_failed_records import ControlledTransport, skip_config

from data_designer_retrieval_sdg.multimodal import JudgeModels, MultimodalSDGConfig, run_multimodal_sdg


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
def test_missing_active_credential_fails_before_calls_or_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    config = stage_config(tmp_path, stage, credential_env="MISSING_PREFLIGHT_KEY", context_window_tokens=None)
    monkeypatch.delenv("MISSING_PREFLIGHT_KEY", raising=False)
    transport = ControlledTransport(())
    with patch.object(ModelFacade, "acompletion", AsyncMock(side_effect=transport.complete)) as completion:
        with pytest.raises(ValueError, match="MISSING_PREFLIGHT_KEY") as error:
            run_multimodal_sdg(config)
    assert stage in str(error.value) and f"operator/{stage}" in str(error.value)
    completion.assert_not_called()
    assert not config.output_dir.exists()


@pytest.mark.parametrize("value", ["", " \t\n"])
def test_empty_credentials_are_reported_without_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    config = stage_config(tmp_path, "self_sufficiency", credential_env="EMPTY_PREFLIGHT_KEY")
    monkeypatch.setenv("EMPTY_PREFLIGHT_KEY", value)
    monkeypatch.setenv("TEST_SDG_KEY", "do-not-print-this-placeholder")
    with pytest.raises(ValueError, match="EMPTY_PREFLIGHT_KEY") as error:
        run_multimodal_sdg(config)
    assert "do-not-print-this-placeholder" not in str(error.value)
    assert not config.output_dir.exists()


def test_unused_judges_need_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = stage_config(tmp_path, "summary")
    config.judge_summaries = False
    config.judges.summary.credential_env = "UNUSED_PREFLIGHT_KEY"
    config.judge = config.judges.summary
    config.context_strategy = "sections"  # Explicit contexts bypass the fallback TOC judge.
    monkeypatch.delenv("UNUSED_PREFLIGHT_KEY", raising=False)
    transport = ControlledTransport(())
    with patch.object(ModelFacade, "acompletion", AsyncMock(side_effect=transport.complete)):
        assert run_multimodal_sdg(config).is_file()
    assert not any(schema in {"SummaryJudgment", "TOCConfirmation"} for schema, _ in transport.calls)


def test_missing_credential_can_be_fixed_without_changing_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = stage_config(tmp_path, "self_sufficiency", credential_env="RECOVER_PREFLIGHT_KEY")
    monkeypatch.delenv("RECOVER_PREFLIGHT_KEY", raising=False)
    transport = ControlledTransport(())
    with patch.object(ModelFacade, "acompletion", AsyncMock(side_effect=transport.complete)) as completion:
        with pytest.raises(ValueError, match="RECOVER_PREFLIGHT_KEY"):
            run_multimodal_sdg(config)
        completion.assert_not_called()
        assert not config.output_dir.exists()
        monkeypatch.setenv("RECOVER_PREFLIGHT_KEY", "offline-placeholder")
        assert run_multimodal_sdg(config).is_file()


def test_completed_resume_needs_no_credentials_and_still_checks_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = skip_config(tmp_path, "unit")
    transport = ControlledTransport(())
    with patch.object(ModelFacade, "acompletion", AsyncMock(side_effect=transport.complete)) as completion:
        handoff = run_multimodal_sdg(config)
        calls = completion.await_count
        monkeypatch.delenv("TEST_SDG_KEY", raising=False)
        resumed = config.model_copy(update={"resume": True})
        assert run_multimodal_sdg(resumed) == handoff
        assert completion.await_count == calls
        (handoff.parent / "report.json").write_text("{}")
        with pytest.raises(ValueError, match="Completed run integrity failure"):
            run_multimodal_sdg(resumed)
        assert completion.await_count == calls


def test_incomplete_resume_checks_credentials_before_new_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = skip_config(tmp_path, "unit")
    transport = ControlledTransport(())
    with patch.object(ModelFacade, "acompletion", AsyncMock(side_effect=transport.complete)) as completion:
        handoff = run_multimodal_sdg(config)
        (config.output_dir / "complete.json").unlink()  # Simulate interruption before the completion marker.
        identity = (config.output_dir / "identity.json").read_bytes()
        calls = completion.await_count
        monkeypatch.delenv("TEST_SDG_KEY", raising=False)
        resumed = config.model_copy(update={"resume": True})
        with pytest.raises(ValueError, match="TEST_SDG_KEY"):
            run_multimodal_sdg(resumed)
        assert completion.await_count == calls
        assert (config.output_dir / "identity.json").read_bytes() == identity
        monkeypatch.setenv("TEST_SDG_KEY", "offline-placeholder")
        assert run_multimodal_sdg(resumed) == handoff
        assert completion.await_count == calls
