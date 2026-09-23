# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic summary pairs/triples over generic source identities; see THIRD_PARTY_NOTICE.txt."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from data_designer_retrieval_sdg.multimodal.models import MultimodalSDGConfig
from data_designer_retrieval_sdg.multimodal.storage import fingerprint, write_json


def hosted_summary_embeddings(texts: list[str], config: MultimodalSDGConfig) -> list[list[float]]:
    """Embed summary passages in bounded batches, caching exact requests for resume."""
    import httpx
    import numpy as np

    credential = os.environ.get(config.summary_embedding_credential_env)
    if not credential:
        raise ValueError(f"Set the credential environment variable {config.summary_embedding_credential_env}")
    vectors = []
    with httpx.Client(timeout=600) as client:
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            body = {
                "model": config.summary_embedding_model,
                "input": batch,
                "encoding_format": "float",
                **config.summary_embedding_extra_body,
            }
            key = fingerprint({"endpoint": config.summary_embedding_endpoint, "body": body})
            path = config.output_dir / "planning/summary_embeddings" / f"{key}.json"
            if path.exists():
                cached = json.loads(path.read_text())
                embeddings = cached["embeddings"]
                if cached["request_id"] != key or cached["sha256"] != fingerprint(embeddings):
                    raise ValueError("Summary embedding cache integrity failure")
            else:
                response = client.post(
                    config.summary_embedding_endpoint.rstrip("/") + "/embeddings",
                    headers={"Authorization": f"Bearer {credential}"},
                    json=body,
                )
                response.raise_for_status()
                records = response.json()["data"]
                if sorted(row["index"] for row in records) != list(range(len(batch))):
                    raise ValueError("Summary embedding response indexes must match the input batch")
                embeddings = [row["embedding"] for row in sorted(records, key=lambda row: row["index"])]
            matrix = np.asarray(embeddings, dtype=float)
            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(batch)
                or not matrix.shape[1]
                or not np.isfinite(matrix).all()
            ):
                raise ValueError("Summary embeddings must be finite and aligned with summaries")
            write_json(path, {"request_id": key, "embeddings": embeddings, "sha256": fingerprint(embeddings)})
            vectors.extend(embeddings)
    return vectors


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
    """Embed summaries with a configured local model or API; cluster each language separately."""
    if len(rows) < 12 or not config.combination_iterations:
        return []
    languages = defaultdict(list)
    for index, row in enumerate(rows):
        languages[row["context"]["language"]].append(index)
    if not any(len(indexes) >= 12 for indexes in languages.values()):
        return []
    if not config.summary_embedding_endpoint:
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
        texts = [rows[i]["summary"]["summary"] for i in indexes]
        vectors = (
            hosted_summary_embeddings(texts, config)
            if config.summary_embedding_endpoint
            else model.encode(texts, batch_size=32)
        )
        combinations = cluster_combinations(
            vectors,
            [rows[i]["document_ids"][0] for i in indexes],
            config.combination_iterations,
        )
        result.extend(tuple(indexes[i] for i in group) for group in combinations)
    return result
