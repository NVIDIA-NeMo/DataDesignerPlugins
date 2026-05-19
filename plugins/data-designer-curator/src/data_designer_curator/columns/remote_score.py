# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.engine.column_generators.generators.base import ColumnGenerator, ColumnGeneratorCellByCell

from data_designer_curator.config import RemoteScoreColumnConfig
from data_designer_curator.errors import RemoteScoringError


def get_path(payload: dict[str, Any], path: str) -> Any:
    """Read a dot-separated path from a response payload."""
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise RemoteScoringError(f"Missing response path {path!r}.")
        value = value[part]
    return value


def validate_response_payload(payload: Any) -> dict[str, Any]:
    """Validate the remote scoring response shape."""
    if not isinstance(payload, dict):
        raise RemoteScoringError("Remote scoring response must be a JSON object.")
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RemoteScoringError("Remote scoring response must contain a non-empty data list.")
    return data[0]


class RemoteScoreColumnGenerator(ColumnGeneratorCellByCell[RemoteScoreColumnConfig]):
    """Generate a score column by calling an external HTTP endpoint."""

    @property
    def is_llm_bound(self) -> bool:
        """Treat remote scoring as network/model-bound work."""
        return True

    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Bridge sync callers to the native async implementation."""
        return ColumnGenerator.generate(self, data)

    async def agenerate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Call the configured scoring endpoint for one row."""
        missing = [column for column in self.config.target_columns if column not in data]
        if missing:
            raise RemoteScoringError(f"Missing target columns: {missing!r}")

        request_row = {column: data[column] for column in self.config.target_columns}
        response_payload = await self._post_score({"data": [request_row]})
        result = validate_response_payload(response_payload)

        data[self.config.name] = get_path(result, self.config.score_path)
        if self.config.side_effect_output_column is not None:
            data[self.config.side_effect_output_column] = result
        return data

    async def _post_score(self, payload: dict[str, Any]) -> Any:
        try:
            import httpx
        except Exception as error:
            raise RemoteScoringError("Install data-designer-curator[remote] to use remote-score.") from error

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    str(self.config.endpoint_url),
                    json=payload,
                    headers=self.config.headers,
                )
            response.raise_for_status()
            return response.json()
        except RemoteScoringError:
            raise
        except Exception as error:
            raise RemoteScoringError(f"Remote scoring failed for column {self.config.name!r}: {error}") from error
