# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deduplication logic (pure numpy, no LLM needed)."""

from data_designer_retrieval_sdg.config import RetrievalSdgDedupColumnConfig


def _make_generator():
    """Instantiate the generator with minimal wiring for dedupe_qa_pairs.

    We only need the config for threshold; the embedder is not used in
    dedupe_qa_pairs itself.
    """
    from unittest.mock import MagicMock

    from data_designer_retrieval_sdg.dedup import RetrievalSdgDedupColumnGenerator

    config = RetrievalSdgDedupColumnConfig(
        name="dedup",
        qa_pairs_column="qa",
        embedding_alias="embed",
        dedupe_similarity_threshold=0.9,
    )
    gen = object.__new__(RetrievalSdgDedupColumnGenerator)
    gen._config = config
    gen._resource_provider = MagicMock()
    return gen


def test_dedupe_empty() -> None:
    gen = _make_generator()
    assert gen.dedupe_qa_pairs([]) == []


def test_dedupe_no_duplicates() -> None:
    gen = _make_generator()
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    kept = gen.dedupe_qa_pairs(embeddings)
    assert kept == [0, 1, 2]


def test_dedupe_identical_vectors() -> None:
    gen = _make_generator()
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    kept = gen.dedupe_qa_pairs(embeddings)
    assert 0 in kept
    assert 1 not in kept
    assert 2 in kept


def test_dedupe_near_threshold() -> None:
    gen = _make_generator()
    v1 = [1.0, 0.0]
    v2 = [0.95, 0.3122]  # cosine sim ≈ 0.95 > 0.9
    v3 = [0.0, 1.0]
    kept = gen.dedupe_qa_pairs([v1, v2, v3])
    assert 0 in kept
    assert 1 not in kept
    assert 2 in kept


def test_dedupe_single_element() -> None:
    gen = _make_generator()
    kept = gen.dedupe_qa_pairs([[1.0, 0.0]])
    assert kept == [0]


def test_config_column_type() -> None:
    cfg = RetrievalSdgDedupColumnConfig(name="dedup", qa_pairs_column="qa", embedding_alias="embed")
    assert cfg.column_type == "retrieval-sdg-dedup"
    assert cfg.required_columns == ["qa"]
    assert cfg.side_effect_columns == []
    assert cfg.get_column_emoji() == "🔍"
