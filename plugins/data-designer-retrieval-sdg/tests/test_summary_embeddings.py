# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public summary embedding requests preserve identities and support safe resume."""

import json
import sys
from types import SimpleNamespace

import httpx
import pytest
from test_multimodal_sdg import fixture_config

from data_designer_retrieval_sdg.multimodal.combinations import hosted_summary_embeddings, semantic_combinations
from data_designer_retrieval_sdg.multimodal.models import MultimodalSDGConfig
from data_designer_retrieval_sdg.multimodal.workflow import run_multimodal_sdg


def api_config(tmp_path):
    raw = fixture_config(tmp_path).model_dump()
    raw.update(
        context_strategy="sections",
        combination_iterations=20,
        summary_embedding_model="nvidia/nemotron-3-embed-1b",
        summary_embedding_endpoint="https://integrate.api.nvidia.com/v1",
        summary_embedding_credential_env="SUMMARY_API_KEY",
        summary_embedding_extra_body={"input_type": "passage", "truncate": "NONE"},
    )
    return MultimodalSDGConfig.model_validate(raw)


def mock_api(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv("SUMMARY_API_KEY", "test-placeholder")


def test_public_api_batches_reorders_and_reuses_exact_cache(tmp_path, monkeypatch):
    config = api_config(tmp_path)
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url) == "https://integrate.api.nvidia.com/v1/embeddings"
        assert request.headers["authorization"] == "Bearer test-placeholder"
        assert body["model"] == "nvidia/nemotron-3-embed-1b"
        assert body["input_type"] == "passage" and body["truncate"] == "NONE"
        assert body["encoding_format"] == "float"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(text), 1.0]}
                    for i, text in reversed(list(enumerate(body["input"])))
                ]
            },
        )

    mock_api(monkeypatch, respond)
    texts = list(map(str, range(35)))
    expected = [[float(i), 1.0] for i in range(35)]
    assert hosted_summary_embeddings(texts, config) == expected
    assert list(map(lambda request: len(request["input"]), requests)) == [32, 3]
    assert hosted_summary_embeddings(texts, config) == expected
    assert len(requests) == 2
    for path in (config.output_dir / "planning/summary_embeddings").glob("*.json"):
        assert "test-placeholder" not in path.read_text()
    path = next((config.output_dir / "planning/summary_embeddings").glob("*.json"))
    cached = json.loads(path.read_text())
    cached["embeddings"][0][0] = -100
    path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="cache integrity"):
        hosted_summary_embeddings(texts, config)


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"index": 1, "embedding": [1.0]}],
        [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],
        [{"index": 0, "embedding": []}],
        [{"index": 0, "embedding": ["NaN"]}],
    ],
)
def test_invalid_api_response_is_not_cached(tmp_path, monkeypatch, records):
    config = api_config(tmp_path)
    mock_api(monkeypatch, lambda request: httpx.Response(200, json={"data": records}))
    with pytest.raises(ValueError):
        hosted_summary_embeddings(["Summary"], config)
    assert not (config.output_dir / "planning/summary_embeddings").exists()


def test_provider_failure_is_not_cached_or_swallowed(tmp_path, monkeypatch):
    config = api_config(tmp_path)
    mock_api(monkeypatch, lambda request: httpx.Response(429))
    with pytest.raises(httpx.HTTPStatusError):
        hosted_summary_embeddings(["Summary"], config)
    assert not (config.output_dir / "planning/summary_embeddings").exists()


def test_missing_embedding_credential_fails_before_generation(tmp_path, monkeypatch):
    config = api_config(tmp_path)
    monkeypatch.delenv("SUMMARY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SUMMARY_API_KEY"):
        run_multimodal_sdg(config)
    assert not config.output_dir.exists()


def test_hosted_combinations_preserve_language_and_summary_order(tmp_path, monkeypatch):
    config = api_config(tmp_path)
    calls = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    def respond(request):
        texts = json.loads(request.content)["input"]
        calls.append(texts)
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [float(text), 1]} for i, text in enumerate(texts)]}
        )

    mock_api(monkeypatch, respond)

    def cluster(vectors, documents, iterations):
        assert [v[0] for v in vectors] == [float(d) for d in documents]
        return [(0, 11)]

    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.combinations.cluster_combinations", cluster)
    rows = [
        {"context": {"language": "en" if i % 2 else "de"}, "summary": {"summary": str(i)}, "document_ids": [str(i)]}
        for i in range(24)
    ]
    assert semantic_combinations(rows, config) == [(0, 22), (1, 23)]
    assert len(calls) == 2


def test_local_embedding_path_remains_configurable(tmp_path, monkeypatch):
    config = api_config(tmp_path).model_copy(
        update={
            "summary_embedding_endpoint": None,
            "summary_embedding_extra_body": {},
            "summary_embedding_model": "operator/local-model",
            "summary_embedding_revision": "a" * 40,
        }
    )
    calls = []

    def local_model(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(encode=lambda texts, batch_size: [[1.0, 2.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=local_model))
    monkeypatch.setattr(
        "data_designer_retrieval_sdg.multimodal.combinations.cluster_combinations", lambda *args: [(0, 1)]
    )
    rows = [{"context": {"language": "en"}, "summary": {"summary": str(i)}, "document_ids": ["doc"]} for i in range(12)]
    assert semantic_combinations(rows, config) == [(0, 1)]
    assert calls == [("operator/local-model", {"revision": "a" * 40, "device": "cpu", "trust_remote_code": False})]


@pytest.mark.parametrize(
    "update",
    [
        {"summary_embedding_revision": "a" * 40},
        {"summary_embedding_extra_body": {"model": "hidden-override"}},
        {"summary_embedding_endpoint": None},
    ],
)
def test_conflicting_embedding_configuration_fails(tmp_path, update):
    raw = api_config(tmp_path).model_dump()
    with pytest.raises(ValueError):
        MultimodalSDGConfig.model_validate({**raw, **update})
