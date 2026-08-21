# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time
from typing import TYPE_CHECKING

from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn
from pydantic import JsonValue

from data_designer_group_consistent.config import GroupConsistentColumnConfig

if TYPE_CHECKING:
    import pandas as pd


def normalize_key_component(value: object) -> dict[str, JsonValue]:
    """Convert a group-key value to a stable JSON representation.

    Args:
        value: Scalar value from a group-key column.

    Returns:
        Type-tagged JSON data suitable for deterministic hashing.
    """
    item_method = getattr(value, "item", None)
    if callable(item_method):
        value = item_method()

    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, float) and math.isnan(value):
        return {"type": "null", "value": None}
    if isinstance(value, float) and math.isinf(value):
        return {"type": "float", "value": "infinity" if value > 0 else "-infinity"}
    if isinstance(value, (date, datetime, time)):
        return {"type": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, (bool, int, float, str)):
        return {"type": type(value).__name__, "value": value}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def select_record_index(group_values: tuple[object, ...], role: str, seed: int, record_count: int) -> int:
    """Select a deterministic candidate record for a logical group.

    Args:
        group_values: Ordered values from the configured group-key columns.
        role: Namespace for independently generated entities in the same group.
        seed: User-controlled generation seed.
        record_count: Number of candidate records.

    Returns:
        An index into the configured candidate record list.
    """
    payload = {
        "version": 1,
        "group": [normalize_key_component(value) for value in group_values],
        "role": role,
        "seed": seed,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest(), byteorder="big") % record_count


class GroupConsistentColumnGenerator(ColumnGeneratorFullColumn[GroupConsistentColumnConfig]):
    """Generate correlated fields that remain stable within each logical group."""

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add fields from one deterministic candidate record per group.

        Args:
            data: Batch containing the configured group-key columns.

        Returns:
            The batch with the configured output columns added.
        """
        records = self.config.records
        indexes = [
            select_record_index(group_values, self.config.role, self.config.seed, len(records))
            for group_values in data[self.config.group_by].itertuples(index=False, name=None)
        ]
        selected_records = [records[index] for index in indexes]
        for output_column, record_field in self.config.field_mapping.items():
            data[output_column] = [record[record_field] for record in selected_records]
        return data
