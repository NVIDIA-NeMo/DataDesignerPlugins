# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bare JSON support must retain schemas, native retries, and negative decisions."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import data_designer.config as dd
import data_designer.interface as ddi
import pandas as pd
import pytest
from data_designer.engine.column_generators.generators.llm_completion import LLMStructuredCellGenerator
from data_designer.engine.models.errors import ModelGenerationValidationFailureError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe
from data_designer.engine.testing import make_stub_completion_response
from data_designer.engine.testing.utils import assert_valid_plugin
from pydantic import BaseModel, ConfigDict

from data_designer_retrieval_sdg.multimodal.config import RetrievalStructuredColumnConfig
from data_designer_retrieval_sdg.multimodal.plugins import retrieval_structured_plugin
from data_designer_retrieval_sdg.multimodal.structured import (
    RetrievalStructuredCellGenerator,
    RetrievalStructuredResponseRecipe,
)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accepted: bool
    reason: str


def response_recipe() -> RetrievalStructuredResponseRecipe:
    return RetrievalStructuredResponseRecipe(Decision.model_json_schema(), pruning=False)


def test_generation_keeps_native_sync_and_async_implementations() -> None:
    assert RetrievalStructuredCellGenerator.generate is LLMStructuredCellGenerator.generate
    assert RetrievalStructuredCellGenerator.agenerate is LLMStructuredCellGenerator.agenerate


@pytest.mark.parametrize("fenced", [False, True])
def test_valid_decisions_are_preserved_without_coercion(fenced: bool) -> None:
    expected = {"accepted": False, "reason": "Insufficient evidence; 日本語 and ``` remain unchanged."}
    value = json.dumps(expected, ensure_ascii=False)
    if fenced:
        value = f"```json\n{value}\n```"
    assert response_recipe().parse(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "[]",
        "null",
        "true",
        '{"accepted": false}',
        '{"accepted": "false", "reason": "bad type"}',
        '{"accepted": false, "reason": "x", "extra": 1}',
        '{"accepted": true, "accepted": false, "reason": "ambiguous"}',
        '{"accepted": false, "reason": NaN}',
        '{"accepted": false, "reason": Infinity}',
        '{"accepted": false, "reason": 1e999}',
        '{"accepted": false, "reason": "truncated',
        'prefix {"accepted": false, "reason": "x"}',
        '{"accepted": false, "reason": "x"} trailing',
        '```json\n{"accepted": false, "reason": "x"}\n```\n```json\n{}\n```',
    ],
)
def test_invalid_or_ambiguous_responses_fail_closed(value: str) -> None:
    with pytest.raises(ParserException):
        response_recipe().parse(value)


def test_nested_duplicate_judgements_are_rejected() -> None:
    recipe = RetrievalStructuredResponseRecipe({"type": "object"}, pruning=False)
    with pytest.raises(ParserException, match="Duplicate"):
        recipe.parse('{"judge": {"accepted": false, "accepted": true}}')


def test_builtin_parser_is_not_monkey_patched() -> None:
    original_parse = StructuredResponseRecipe.parse
    expected = {"accepted": False, "reason": "x"}
    assert response_recipe().parse(json.dumps(expected)) == expected
    assert StructuredResponseRecipe.parse is original_parse
    assert (
        StructuredResponseRecipe(Decision.model_json_schema()).parse(f"```json\n{json.dumps(expected)}\n```")
        == expected
    )


def model_facade() -> ModelFacade:
    return ModelFacade(
        dd.ModelConfig(alias="judge", model="nvidia/test-model", provider="nvidia"),
        MagicMock(),
        client=MagicMock(),
    )


def test_native_corrections_are_bounded_and_keep_a_valid_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock only transport completion; run the host's real correction loop."""
    facade = model_facade()
    monkeypatch.setattr(facade, "_get_mcp_facade", lambda _: None)
    completion = AsyncMock(
        side_effect=[
            make_stub_completion_response(content='{"accepted": "false", "reason": "wrong type"}'),
            make_stub_completion_response(content='{"accepted": false, "reason": "not grounded"}'),
        ]
    )
    monkeypatch.setattr(facade, "acompletion", completion)
    result, _ = asyncio.run(
        facade.agenerate(
            "Judge the evidence.",
            parser=response_recipe().parse,
            max_correction_steps=1,
            max_conversation_restarts=0,
            skip_usage_tracking=True,
        )
    )
    assert result == {"accepted": False, "reason": "not grounded"}
    assert completion.await_count == 2


def test_invalid_response_exhausts_native_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    facade = model_facade()
    monkeypatch.setattr(facade, "_get_mcp_facade", lambda _: None)
    completion = AsyncMock(return_value=make_stub_completion_response(content="not JSON"))
    monkeypatch.setattr(facade, "acompletion", completion)
    with pytest.raises(ModelGenerationValidationFailureError):
        asyncio.run(
            facade.agenerate(
                "Judge the evidence.",
                parser=response_recipe().parse,
                max_correction_steps=1,
                max_conversation_restarts=0,
                skip_usage_tracking=True,
            )
        )
    assert completion.await_count == 2


@pytest.mark.parametrize("fenced", [False, True])
def test_registered_column_runs_through_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fenced: bool) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-a-real-credential")
    value = '{"accepted": false, "reason": "not grounded"}'
    if fenced:
        value = f"```json\n{value}\n```"
    completion = AsyncMock(return_value=make_stub_completion_response(content=value))
    monkeypatch.setattr(ModelFacade, "acompletion", completion)
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="judge",
                model="nvidia/test-model",
                provider="nvidia",
                skip_health_check=True,
                inference_parameters=dd.ChatCompletionInferenceParams(),
            )
        ]
    )
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=pd.DataFrame([{"source": "Public example."}])))
    builder.add_column(
        RetrievalStructuredColumnConfig(
            name="decision",
            model_alias="judge",
            prompt="Judge {{ source }}.",
            output_format=Decision,
        )
    )
    designer = ddi.DataDesigner(artifact_path=tmp_path)
    designer.set_run_config(
        dd.RunConfig(
            max_conversation_correction_steps=1,
            max_conversation_restarts=0,
            otel_metrics_port=None,
        )
    )
    result = designer.preview(builder, num_records=1).dataset.iloc[0]
    assert result["decision"] == {"accepted": False, "reason": "not grounded"}
    assert completion.await_count == 1


def test_retrieval_structured_plugin_valid() -> None:
    assert_valid_plugin(retrieval_structured_plugin)


@pytest.mark.parametrize(
    "initial_import",
    [
        "import data_designer.interface",
        "import data_designer_retrieval_sdg.multimodal",
        'from importlib.metadata import entry_points; entry_points(group="data_designer.plugins")["retrieval-structured"].load()',
    ],
    ids=["data-designer-first", "multimodal-first", "entry-point-first"],
)
def test_plugin_discovery_in_fresh_interpreter(initial_import: str) -> None:
    """Plugin discovery must not load the workflow or engine through a cycle."""
    checks = """
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer_retrieval_sdg.multimodal.plugins import retrieval_structured_plugin
from data_designer_retrieval_sdg.multimodal import run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.workflow import run_multimodal_sdg as implementation
assert_valid_plugin(retrieval_structured_plugin)
assert run_multimodal_sdg is implementation
"""
    result = subprocess.run(
        [sys.executable, "-c", initial_import + "\n" + checks],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "Failed to load plugin" not in result.stdout + result.stderr
