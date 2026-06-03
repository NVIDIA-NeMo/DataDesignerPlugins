# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the embedding-dedup column generator."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from data_designer.config.errors import BuilderConfigurationError
from data_designer.config.models import (
    ChatCompletionInferenceParams,
    EmbeddingInferenceParams,
    ModelConfig,
)

from data_designer_retrieval_sdg.config import EmbeddingDedupColumnConfig
from data_designer_retrieval_sdg.dedup import EmbeddingDedupColumnGenerator


def _make_generator(
    *,
    source_column: str = "qa",
    items_key: str | None = "pairs",
    text_field: str = "question",
    threshold: float = 0.9,
) -> EmbeddingDedupColumnGenerator:
    """Instantiate the generator with minimal wiring for unit-level tests."""
    config = EmbeddingDedupColumnConfig(
        name="dedup",
        source_column=source_column,
        items_key=items_key,
        text_field=text_field,
        model_alias="embed",
        similarity_threshold=threshold,
    )
    gen = object.__new__(EmbeddingDedupColumnGenerator)
    gen._config = config
    gen._resource_provider = MagicMock()
    return gen


def test_dedupe_indices_empty() -> None:
    gen = _make_generator()
    assert gen.dedupe_indices([]) == []


def test_dedupe_indices_no_duplicates() -> None:
    gen = _make_generator()
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    assert gen.dedupe_indices(embeddings) == [0, 1, 2]


def test_dedupe_indices_identical_vectors() -> None:
    gen = _make_generator()
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    kept = gen.dedupe_indices(embeddings)
    assert 0 in kept
    assert 1 not in kept
    assert 2 in kept


def test_dedupe_indices_near_threshold() -> None:
    gen = _make_generator()
    v1 = [1.0, 0.0]
    v2 = [0.95, 0.3122]
    v3 = [0.0, 1.0]
    kept = gen.dedupe_indices([v1, v2, v3])
    assert 0 in kept
    assert 1 not in kept
    assert 2 in kept


def test_dedupe_indices_single_element() -> None:
    gen = _make_generator()
    assert gen.dedupe_indices([[1.0, 0.0]]) == [0]


def test_resolve_items_with_items_key() -> None:
    gen = _make_generator(items_key="pairs")
    items = gen.resolve_items({"qa": {"pairs": [{"question": "x"}]}})
    assert items == [{"question": "x"}]


def test_resolve_items_without_items_key() -> None:
    gen = _make_generator(items_key=None)
    items = gen.resolve_items({"qa": [{"question": "x"}]})
    assert items == [{"question": "x"}]


def test_resolve_items_missing_source_returns_empty_list() -> None:
    gen = _make_generator(items_key=None)
    assert gen.resolve_items({}) == []


def test_extract_text_dict_and_attribute() -> None:
    gen = _make_generator(text_field="question")
    assert gen.extract_text({"question": "hello"}) == "hello"

    class Item:
        question = "world"

    assert gen.extract_text(Item()) == "world"


def test_generate_calls_embedder_once_with_all_texts() -> None:
    gen = _make_generator()
    embedder = MagicMock()
    embedder.generate_text_embeddings.return_value = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    gen.resource_provider.model_registry.get_model.return_value = embedder

    row = {"qa": {"pairs": [{"question": "a"}, {"question": "b"}, {"question": "c"}]}}
    out = gen.generate(row)

    embedder.generate_text_embeddings.assert_called_once()
    call_kwargs = embedder.generate_text_embeddings.call_args.kwargs
    assert call_kwargs["input_texts"] == ["a", "b", "c"]
    assert call_kwargs["encoding_format"] == "float"
    assert out["dedup"] == [{"question": "a"}, {"question": "c"}]


def test_agenerate_uses_async_embedder() -> None:
    gen = _make_generator()
    embedder = MagicMock()
    embedder.agenerate_text_embeddings = AsyncMock(return_value=[[1.0, 0.0], [1.0, 0.0]])
    embedder.generate_text_embeddings = MagicMock()
    gen.resource_provider.model_registry.get_model.return_value = embedder

    row = {"qa": {"pairs": [{"question": "a"}, {"question": "b"}]}}
    out = asyncio.run(gen.agenerate(row))

    embedder.agenerate_text_embeddings.assert_awaited_once()
    embedder.generate_text_embeddings.assert_not_called()
    assert out["dedup"] == [{"question": "a"}]


def test_agenerate_empty_items_short_circuits() -> None:
    gen = _make_generator()
    embedder = MagicMock()
    embedder.agenerate_text_embeddings = AsyncMock()
    gen.resource_provider.model_registry.get_model.return_value = embedder

    out = asyncio.run(gen.agenerate({"qa": {"pairs": []}}))

    embedder.agenerate_text_embeddings.assert_not_awaited()
    assert out["dedup"] == []


def test_config_round_trip() -> None:
    cfg = EmbeddingDedupColumnConfig(
        name="dedup",
        source_column="qa_generation",
        model_alias="embed",
    )
    assert cfg.column_type == "embedding-dedup"
    assert cfg.required_columns == ["qa_generation"]
    assert cfg.side_effect_columns == []
    assert cfg.get_column_emoji() == "🔍"
    assert cfg.items_key == "pairs"
    assert cfg.text_field == "question"
    assert cfg.similarity_threshold == 0.9


def test_scheduling_metadata_uses_embedding_model_alias() -> None:
    """Embedding calls should route through DataDesigner's model scheduler."""
    gen = _make_generator()
    gen.resource_provider.model_registry.get_model_config.return_value = ModelConfig(
        alias="embed",
        model="mock-embedding-model",
        provider="mock-provider",
        inference_parameters=EmbeddingInferenceParams(max_parallel_requests=3),
    )
    gen.resource_provider.model_registry.get_model_provider.return_value.name = "mock-provider"

    metadata = gen.get_scheduling_metadata()

    assert metadata.kind == "model"
    assert metadata.identity == ("model", "mock-provider", "mock-embedding-model", "embedding")
    assert metadata.weight == 3
    assert metadata.diagnostics["aliases"] == ("embed",)


def test_validate_accepts_embedding_model() -> None:
    """``_validate()`` should succeed when the configured alias resolves to
    a ``ModelConfig`` whose inference parameters declare an embedding model."""
    gen = _make_generator()
    gen.resource_provider.model_registry.get_model_config.return_value = ModelConfig(
        alias="embed",
        model="some/embedding-model",
        provider="mock-provider",
        inference_parameters=EmbeddingInferenceParams(),
    )
    gen._validate()


def test_validate_rejects_chat_model() -> None:
    """``_validate()`` should fail fast at task construction when the alias
    resolves to a non-embedding model, naming the offending alias."""
    gen = _make_generator()
    gen.resource_provider.model_registry.get_model_config.return_value = ModelConfig(
        alias="embed",
        model="some/chat-model",
        provider="mock-provider",
        inference_parameters=ChatCompletionInferenceParams(),
    )
    with pytest.raises(BuilderConfigurationError, match="embed"):
        gen._validate()


def test_embedder_is_cached_across_calls() -> None:
    """Repeated access should hit ``model_registry.get_model`` exactly once
    so per-row dedup doesn't re-walk the registry."""
    gen = _make_generator()
    gen.resource_provider.model_registry.get_model.return_value = MagicMock()

    first = gen.embedder
    second = gen.embedder

    assert first is second
    gen.resource_provider.model_registry.get_model.assert_called_once_with(model_alias="embed")
