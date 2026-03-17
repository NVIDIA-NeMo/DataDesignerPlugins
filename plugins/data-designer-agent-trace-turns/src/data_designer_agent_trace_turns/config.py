# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.base import SingleColumnConfig

CONVERSATION_INDEX_COLUMN_POSTFIX = "__conversation_index"
TURN_INDEX_COLUMN_POSTFIX = "__turn_index"
ROLE_COLUMN_POSTFIX = "__role"
SPEAKER_COLUMN_POSTFIX = "__speaker"
TURN_KIND_COLUMN_POSTFIX = "__turn_kind"
TRACE_PATH_COLUMN_POSTFIX = "__trace_path"
RAW_TURN_COLUMN_POSTFIX = "__raw_turn"


class AgentTraceTurnsColumnConfig(SingleColumnConfig):
    """Expand agent traces into one row per normalized conversational step.

    The source column may contain a Python dict/list payload, a JSON string, or a
    Data Designer `__trace` column containing serialized chat messages. The
    generator duplicates each source row for every extracted conversational step
    and writes the normalized text for that step into `name`.

    Attributes:
        source_column: Column containing the raw agent trace payload.
        keep_empty_rows: If True, preserve rows that yield no conversational
            steps by emitting a single row with null turn metadata.
        emit_raw_turn: If True, emit a `{name}__raw_turn` side-effect column
            containing the JSON-serialized source object for each extracted turn.
        allow_resize: Always True because this plugin expands or drops rows while
            normalizing a trace.
        column_type: Discriminator field, always "agent-trace-turns".
    """

    column_type: Literal["agent-trace-turns"] = "agent-trace-turns"
    source_column: str
    keep_empty_rows: bool = False
    emit_raw_turn: bool = True
    allow_resize: Literal[True] = True

    @staticmethod
    def get_column_emoji() -> str:
        return "🧵"

    @property
    def required_columns(self) -> list[str]:
        return [self.source_column]

    @property
    def side_effect_columns(self) -> list[str]:
        columns = [
            f"{self.name}{CONVERSATION_INDEX_COLUMN_POSTFIX}",
            f"{self.name}{TURN_INDEX_COLUMN_POSTFIX}",
            f"{self.name}{ROLE_COLUMN_POSTFIX}",
            f"{self.name}{SPEAKER_COLUMN_POSTFIX}",
            f"{self.name}{TURN_KIND_COLUMN_POSTFIX}",
            f"{self.name}{TRACE_PATH_COLUMN_POSTFIX}",
        ]
        if self.emit_raw_turn:
            columns.append(f"{self.name}{RAW_TURN_COLUMN_POSTFIX}")
        return columns
