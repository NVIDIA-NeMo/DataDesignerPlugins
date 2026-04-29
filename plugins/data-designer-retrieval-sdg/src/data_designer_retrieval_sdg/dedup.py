# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column generator that deduplicates QA pairs via embedding cosine similarity."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell

from data_designer_retrieval_sdg.config import RetrievalSdgDedupColumnConfig

logger = logging.getLogger(__name__)


class RetrievalSdgDedupColumnGenerator(ColumnGeneratorCellByCell[RetrievalSdgDedupColumnConfig]):
    """Remove near-duplicate QA pairs using embedding cosine similarity.

    For each cell the generator:
    1. Reads QA pairs from the configured source column.
    2. Embeds every question in parallel via the registered embedding model.
    3. Computes pairwise cosine similarity and greedily drops duplicates
       whose similarity exceeds ``dedupe_similarity_threshold``.
    4. Returns the surviving pairs under the column name.
    """

    @property
    def embedder(self):
        """Resolve the embedding model from the resource provider."""
        return self.resource_provider.model_registry.get_model(
            model_alias=self.config.embedding_alias,
        )

    def embed_text(self, text: str) -> list[float]:
        """Compute an embedding vector for *text* using the configured model.

        Args:
            text: Input string to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        vectors = self.embedder.generate_text_embeddings(
            input_texts=[text],
            encoding_format="float",
        )
        return vectors[0]

    def dedupe_qa_pairs(self, embeddings: list[list[float]]) -> list[int]:
        """Return indices of QA pairs to keep after greedy deduplication.

        Computes pairwise cosine similarity.  For every pair above the
        threshold the later item is dropped.

        Args:
            embeddings: 2-D list of embedding vectors, one per QA pair.

        Returns:
            Sorted list of integer indices to retain.

        Raises:
            ValueError: If *embeddings* is not a 2-D structure.
        """
        if not embeddings:
            return []

        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D array of shape (n, d).")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms

        cosine_sim = np.clip(normalized @ normalized.T, -1.0, 1.0)

        threshold = self.config.dedupe_similarity_threshold
        keep_indexes: list[int] = []
        dropped = np.zeros(len(embeddings), dtype=bool)

        for i in range(len(embeddings)):
            if dropped[i]:
                continue
            keep_indexes.append(i)
            if i == len(embeddings) - 1:
                continue
            close_matches = np.where(cosine_sim[i, i + 1 :] > threshold)[0] + i + 1
            dropped[close_matches] = True

        return keep_indexes

    def generate(self, data: dict) -> dict:
        """Deduplicate QA pairs for a single record.

        Args:
            data: Row dict containing at least the ``qa_pairs_column``.

        Returns:
            Updated row dict with the deduplicated pairs stored under
            ``self.config.name``.
        """
        logger.debug("Deduplicating QA pairs from column: %s", self.config.qa_pairs_column)

        qa_pairs: list = data[self.config.qa_pairs_column]["pairs"]
        max_parallel = self.embedder.max_parallel_requests
        workers = max(1, max_parallel or 1)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            embeddings = list(executor.map(self.embed_text, [qa["question"] for qa in qa_pairs]))

        retained_indexes = self.dedupe_qa_pairs(embeddings)
        dropped = len(qa_pairs) - len(retained_indexes)
        if dropped > 0:
            logger.info(
                "Dedup: retained %d of %d QA pairs (%d duplicates removed)",
                len(retained_indexes),
                len(qa_pairs),
                dropped,
            )
        else:
            logger.debug("Dedup: retained all %d QA pairs (no duplicates)", len(qa_pairs))

        retained_qa_pairs = [qa_pairs[i] for i in retained_indexes]
        return data | {self.config.name: retained_qa_pairs}
