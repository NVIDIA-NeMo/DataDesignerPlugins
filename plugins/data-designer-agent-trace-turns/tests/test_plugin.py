# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pandas as pd
import pytest
from data_designer.engine.testing.utils import assert_valid_plugin

from data_designer_agent_trace_turns.config import AgentTraceTurnsColumnConfig
from data_designer_agent_trace_turns.impl import AgentTraceTurnsColumnGenerator
from data_designer_agent_trace_turns.plugin import plugin


def test_valid_plugin():
    assert_valid_plugin(plugin)


@pytest.fixture()
def make_config():
    def _make(
        *,
        name: str = "turn_text",
        source_column: str = "trace",
        keep_empty_rows: bool = False,
        emit_raw_turn: bool = True,
    ) -> AgentTraceTurnsColumnConfig:
        return AgentTraceTurnsColumnConfig(
            name=name,
            source_column=source_column,
            keep_empty_rows=keep_empty_rows,
            emit_raw_turn=emit_raw_turn,
        )

    return _make


def _make_generator(config: AgentTraceTurnsColumnConfig) -> AgentTraceTurnsColumnGenerator:
    generator = AgentTraceTurnsColumnGenerator.__new__(AgentTraceTurnsColumnGenerator)
    generator._config = config
    return generator


def test_expands_data_designer_trace_messages(make_config) -> None:
    trace = [
        {"role": "system", "content": [{"type": "text", "text": "Stay concise."}]},
        {"role": "user", "content": [{"type": "text", "text": "Plan a trip."}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Where do you want to go?"}],
            "reasoning_content": "Need clarification before answering.",
        },
    ]
    source_df = pd.DataFrame({"trace_id": ["trace-1"], "trace": [trace]})

    result = _make_generator(make_config()).generate(source_df)

    assert list(result["trace_id"]) == ["trace-1", "trace-1", "trace-1"]
    assert list(result["turn_text"]) == ["Stay concise.", "Plan a trip.", "Where do you want to go?"]
    assert list(result["turn_text__role"]) == ["system", "user", "assistant"]
    assert list(result["turn_text__speaker"]) == ["system", "user", "assistant"]
    assert list(result["turn_text__turn_kind"]) == ["message", "message", "message"]
    assert list(result["turn_text__conversation_index"]) == [0, 0, 0]
    assert list(result["turn_text__turn_index"]) == [0, 1, 2]
    assert list(result["turn_text__trace_path"]) == ["$[0]", "$[1]", "$[2]"]
    assert result["turn_text__raw_turn"].str.contains('"role":"assistant"').any()


def test_extracts_nested_conversations_from_json_strings(make_config) -> None:
    trace = json.dumps(
        {
            "conversations": [
                {
                    "messages": [
                        {"speaker": "planner", "text": "Need two candidate cities.", "kind": "question"},
                        {"role": "assistant", "content": {"parts": ["Paris", "Rome"]}},
                    ]
                },
                {
                    "turns": [
                        {
                            "author": {"name": "tool", "role": "tool"},
                            "message": {"value": '{"latency_ms": 12}'},
                        }
                    ]
                },
            ]
        }
    )
    source_df = pd.DataFrame({"trace": [trace], "batch_id": ["batch-7"]})

    result = _make_generator(make_config()).generate(source_df)

    assert list(result["batch_id"]) == ["batch-7", "batch-7", "batch-7"]
    assert list(result["turn_text"]) == ["Need two candidate cities.", "Paris\n\nRome", '{"latency_ms": 12}']
    assert list(result["turn_text__conversation_index"]) == [0, 0, 1]
    assert list(result["turn_text__turn_index"]) == [0, 1, 0]
    assert list(result["turn_text__speaker"]) == ["planner", "assistant", "tool"]
    assert list(result["turn_text__role"]) == [None, "assistant", "tool"]
    assert list(result["turn_text__turn_kind"]) == ["question", "message", "turn"]
    assert list(result["turn_text__trace_path"]) == [
        "$.conversations[0].messages[0]",
        "$.conversations[0].messages[1]",
        "$.conversations[1].turns[0]",
    ]


def test_drops_rows_without_turns_by_default(make_config) -> None:
    source_df = pd.DataFrame({"trace": [{}, {"metadata": {"status": "ok"}}]})

    result = _make_generator(make_config()).generate(source_df)

    assert result.empty
    assert "turn_text" in result.columns


def test_keep_empty_rows_preserves_unmatched_traces(make_config) -> None:
    source_df = pd.DataFrame({"trace_id": ["trace-2"], "trace": [{}]})

    result = _make_generator(make_config(keep_empty_rows=True)).generate(source_df)

    assert len(result) == 1
    assert result.iloc[0]["trace_id"] == "trace-2"
    assert pd.isna(result.iloc[0]["turn_text"])
    assert pd.isna(result.iloc[0]["turn_text__role"])
    assert pd.isna(result.iloc[0]["turn_text__trace_path"])


def test_emit_raw_turn_false_omits_raw_turn_column(make_config) -> None:
    source_df = pd.DataFrame({"trace": [[{"role": "user", "content": "hello"}]]})

    config = make_config(emit_raw_turn=False)
    result = _make_generator(config).generate(source_df)

    assert config.side_effect_columns == [
        "turn_text__conversation_index",
        "turn_text__turn_index",
        "turn_text__role",
        "turn_text__speaker",
        "turn_text__turn_kind",
        "turn_text__trace_path",
    ]
    assert "turn_text__raw_turn" not in result.columns
