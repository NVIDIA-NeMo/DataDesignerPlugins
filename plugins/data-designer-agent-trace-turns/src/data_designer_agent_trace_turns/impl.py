# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from data_designer_agent_trace_turns.config import (
    CONVERSATION_INDEX_COLUMN_POSTFIX,
    RAW_TURN_COLUMN_POSTFIX,
    ROLE_COLUMN_POSTFIX,
    SPEAKER_COLUMN_POSTFIX,
    TRACE_PATH_COLUMN_POSTFIX,
    TURN_INDEX_COLUMN_POSTFIX,
    TURN_KIND_COLUMN_POSTFIX,
    AgentTraceTurnsColumnConfig,
)
from data_designer_agent_trace_turns.trace_parser import ExtractedTraceTurn, extract_trace_turns


class AgentTraceTurnsColumnGenerator(ColumnGeneratorFullColumn[AgentTraceTurnsColumnConfig]):
    """Normalize trace payloads into one row per conversational step."""

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Expand the source trace column into a turn-level DataFrame.

        Args:
            data: Input dataset containing a source trace column.

        Returns:
            A resized DataFrame containing one row per extracted conversational
            step, with metadata side-effect columns added alongside the main
            normalized text column.
        """

        output_columns = _resolve_output_columns(list(data.columns), self.config)
        expanded_records: list[dict[str, Any]] = []

        for record in data.to_dict(orient="records"):
            expanded_records.extend(_expand_record(record, self.config))

        return pd.DataFrame(expanded_records, columns=output_columns)


def _expand_record(
    source_record: dict[str, Any],
    config: AgentTraceTurnsColumnConfig,
) -> list[dict[str, Any]]:
    turns = extract_trace_turns(source_record.get(config.source_column))
    if not turns:
        if config.keep_empty_rows:
            return [_build_empty_record(source_record, config)]
        return []
    return [_build_turn_record(source_record, config, turn) for turn in turns]


def _build_turn_record(
    source_record: dict[str, Any],
    config: AgentTraceTurnsColumnConfig,
    turn: ExtractedTraceTurn,
) -> dict[str, Any]:
    record = dict(source_record)
    record[config.name] = turn.text
    record[f"{config.name}{CONVERSATION_INDEX_COLUMN_POSTFIX}"] = turn.conversation_index
    record[f"{config.name}{TURN_INDEX_COLUMN_POSTFIX}"] = turn.turn_index
    record[f"{config.name}{ROLE_COLUMN_POSTFIX}"] = turn.role
    record[f"{config.name}{SPEAKER_COLUMN_POSTFIX}"] = turn.speaker
    record[f"{config.name}{TURN_KIND_COLUMN_POSTFIX}"] = turn.turn_kind
    record[f"{config.name}{TRACE_PATH_COLUMN_POSTFIX}"] = turn.trace_path
    if config.emit_raw_turn:
        record[f"{config.name}{RAW_TURN_COLUMN_POSTFIX}"] = turn.raw_turn_json
    return record


def _build_empty_record(
    source_record: dict[str, Any],
    config: AgentTraceTurnsColumnConfig,
) -> dict[str, Any]:
    record = dict(source_record)
    record[config.name] = None
    for column_name in config.side_effect_columns:
        record[column_name] = None
    return record


def _resolve_output_columns(
    base_columns: list[str],
    config: AgentTraceTurnsColumnConfig,
) -> list[str]:
    output_columns = list(base_columns)
    for column_name in [config.name, *config.side_effect_columns]:
        if column_name not in output_columns:
            output_columns.append(column_name)
    return output_columns
