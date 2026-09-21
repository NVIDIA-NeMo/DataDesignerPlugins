# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic summary pairs/triples over generic source identities; see THIRD_PARTY_NOTICE.txt."""

from __future__ import annotations

import random
from collections import defaultdict

from data_designer_retrieval_sdg.multimodal.models import MultimodalSDGConfig


def sample_members(groups: dict[str, list[int]], size: int, rng: random.Random) -> tuple[int, ...] | None:
    """Sample a reference within/across-document pair or triple with three bounded attempts."""
    for _ in range(3):
        try:
            if rng.random() < 0.5:
                if size == 2 or rng.random() < 0.7:
                    documents = rng.sample(list(groups), size)
                else:
                    documents = rng.sample(list(groups), 2)
                    documents.append(rng.choice(documents))
                members = []
                for document in documents:
                    members.append(rng.choice([i for i in groups[document] if i not in members]))
            else:
                document = rng.choice([key for key, values in groups.items() if len(values) >= size])
                members = rng.sample(groups[document], size)
            return tuple(sorted(members))
        except (IndexError, ValueError):
            continue
    return None


def cluster_combinations(vectors, documents: list[str], iterations: int) -> list[tuple[int, ...]]:
    """Apply the reference seeded UMAP/HDBSCAN and pair/triple sampling policy.

    Args:
        vectors: One embedding per section summary in stable input order.
        documents: Corresponding opaque document identities.
        iterations: Number of independent clustering seeds, starting at zero.

    Returns:
        Unique pairs followed by triples in deterministic order. Fewer than twelve
        summaries cannot satisfy the reference UMAP dimensions and yield no combinations.
    """
    if len(documents) < 12 or not iterations:
        return []
    import numpy as np
    from sklearn.cluster import HDBSCAN
    from umap import UMAP

    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or len(vectors) != len(documents) or not np.isfinite(vectors).all():
        raise ValueError("Summary embeddings must be finite and aligned with summaries")
    pairs, triples = set(), set()
    for seed in range(iterations):
        rng = random.Random(seed)
        reduced = UMAP(
            n_neighbors=5,
            n_components=rng.randint(5, 10),
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
            transform_seed=seed,
        ).fit_transform(vectors)
        reduced -= reduced.mean(axis=0)
        labels = HDBSCAN(min_cluster_size=2, metric="euclidean", cluster_selection_method="eom").fit_predict(reduced)
        for label in sorted(set(labels) - {-1}):
            groups = defaultdict(list)
            for index, value in enumerate(labels):
                if value == label:
                    groups[documents[index]].append(index)
            for size, target in ((2, pairs), (3, triples)):
                sampled = sample_members(groups, size, rng)
                if sampled is not None:
                    target.add(sampled)
    return sorted(pairs) + sorted(triples)


def semantic_combinations(rows: list[dict], config: MultimodalSDGConfig) -> list[tuple[int, ...]]:
    """Embed summaries locally and form combinations independently for each language."""
    if len(rows) < 12 or not config.combination_iterations:
        return []
    languages = defaultdict(list)
    for index, row in enumerate(rows):
        languages[row["context"]["language"]].append(index)
    if not any(len(indexes) >= 12 for indexes in languages.values()):
        return []
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config.summary_embedding_model,
        revision=config.summary_embedding_revision,
        device=config.summary_embedding_device,
        trust_remote_code=False,
    )
    result = []
    for indexes in languages.values():
        if len(indexes) < 12:
            continue
        vectors = model.encode([rows[i]["summary"]["summary"] for i in indexes], batch_size=32)
        combinations = cluster_combinations(
            vectors,
            [rows[i]["document_ids"][0] for i in indexes],
            config.combination_iterations,
        )
        result.extend(tuple(indexes[i] for i in group) for group in combinations)
    return result
