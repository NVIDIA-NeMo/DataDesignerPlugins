# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from data_designer.engine.resources.resource_provider import ResourceProvider

from data_designer_curator.columns.remote_score import RemoteScoreColumnGenerator
from data_designer_curator.config import RemoteScoreColumnConfig
from data_designer_curator.errors import RemoteScoringError


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self.payload


class FakeAsyncClient:
    response = FakeResponse({"data": [{"score": 0.93, "label": "high_quality"}]})
    requests: list[dict[str, Any]] = []

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str] | None) -> FakeResponse:
        self.requests.append({"url": url, "json": json, "headers": headers, "timeout": self.timeout})
        return self.response


def test_remote_score_extracts_score_and_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    resource_provider: ResourceProvider,
) -> None:
    import httpx

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse({"data": [{"score": 0.93, "label": "high_quality"}]})
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    config = RemoteScoreColumnConfig(
        name="quality_score",
        endpoint_url="https://quality.example/score",
        target_columns=["question", "answer"],
        headers={"Authorization": "Bearer token"},
        side_effect_output_column="quality_metadata",
    )

    output = asyncio.run(
        RemoteScoreColumnGenerator(config, resource_provider).agenerate({"question": "q", "answer": "a", "extra": 1})
    )

    assert output["quality_score"] == 0.93
    assert output["quality_metadata"] == {"score": 0.93, "label": "high_quality"}
    assert FakeAsyncClient.requests == [
        {
            "url": "https://quality.example/score",
            "json": {"data": [{"question": "q", "answer": "a"}]},
            "headers": {"Authorization": "Bearer token"},
            "timeout": 30.0,
        }
    ]


def test_remote_score_extracts_nested_score(
    monkeypatch: pytest.MonkeyPatch,
    resource_provider: ResourceProvider,
) -> None:
    import httpx

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse({"data": [{"metrics": {"quality": 0.75}}]})
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    config = RemoteScoreColumnConfig(
        name="quality_score",
        endpoint_url="https://quality.example/score",
        target_columns=["text"],
        score_path="metrics.quality",
    )

    output = asyncio.run(RemoteScoreColumnGenerator(config, resource_provider).agenerate({"text": "hello"}))

    assert output["quality_score"] == 0.75


def test_remote_score_normalizes_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
    resource_provider: ResourceProvider,
) -> None:
    import httpx

    FakeAsyncClient.response = FakeResponse({"data": [{"label": "missing score"}]})
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    config = RemoteScoreColumnConfig(
        name="quality_score",
        endpoint_url="https://quality.example/score",
        target_columns=["text"],
    )

    with pytest.raises(RemoteScoringError, match="Missing response path"):
        asyncio.run(RemoteScoreColumnGenerator(config, resource_provider).agenerate({"text": "hello"}))


def test_remote_score_raises_for_missing_target_column(resource_provider: ResourceProvider) -> None:
    config = RemoteScoreColumnConfig(
        name="quality_score",
        endpoint_url="https://quality.example/score",
        target_columns=["text"],
    )

    with pytest.raises(RemoteScoringError, match="Missing target columns"):
        asyncio.run(RemoteScoreColumnGenerator(config, resource_provider).agenerate({"other": "hello"}))
