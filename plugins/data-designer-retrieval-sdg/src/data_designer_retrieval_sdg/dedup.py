# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic embedding-cosine-similarity dedup column generator.

Implements both ``generate()`` (sync) and ``agenerate()`` (async-native)
so the column participates in DataDesigner's ``DATA_DESIGNER_ASYNC_ENGINE``
scheduler when enabled, falling back to the sync bridge otherwise.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import numpy as np
from data_designer.config.errors import BuilderConfigurationError
from data_designer.config.models import GenerationType
from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorWithModelRegistry,
    GenerationStrategy,
)
from data_designer.engine.models.facade import ModelFacade

from data_designer_retrieval_sdg.config import EmbeddingDedupColumnConfig

logger = logging.getLogger(__name__)


class EmbeddingDedupColumnGenerator(ColumnGeneratorWithModelRegistry[EmbeddingDedupColumnConfig]):
    """Remove near-duplicate items from a list-valued column.

    For each row the generator:

    1. Resolves the items list at ``data[source_column][items_key]``
       (or ``data[source_column]`` when ``items_key`` is ``None``).
    2. Pulls the text field from each item via :meth:`extract_text`.
    3. Embeds the texts in a single batched call to the embedding model.
    4. Computes pairwise cosine similarity and greedily drops items whose
       similarity exceeds ``similarity_threshold``.
    5. Returns the surviving items under ``self.config.name``.

    Extends :class:`ColumnGeneratorWithModelRegistry` so the column reports
    ``is_llm_bound = True`` to the async scheduler. Without this, embedding
    HTTP calls would bypass ``_llm_wait_semaphore`` and could fan out up to
    a full row group's worth of concurrent requests at the embedding endpoint.
    """

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        """Each row's items are deduplicated independently."""
        return GenerationStrategy.CELL_BY_CELL

    @functools.cached_property
    def embedder(self) -> ModelFacade:
        """Resolve the embedding model once and cache it on the instance."""
        return self.get_model(model_alias=self.config.model_alias)

    def _validate(self) -> None:
        """Fail fast at task construction if the alias isn't an embedding model.

        Without this guard, a misconfigured chat-model alias surfaces only on
        the first row's embedding call as either an :class:`AttributeError`
        from the facade or a 400 from the embeddings endpoint.

        Raises:
            BuilderConfigurationError: When ``self.config.model_alias`` resolves
                to a :class:`ModelConfig` whose inference parameters are not
                ``EmbeddingInferenceParams``.
        """
        super()._validate()
        model_config = self.get_model_config(model_alias=self.config.model_alias)
        if model_config.generation_type != GenerationType.EMBEDDING:
            raise BuilderConfigurationError(
                f"EmbeddingDedupColumnGenerator requires an embedding model, "
                f"but model alias {self.config.model_alias!r} resolves to a "
                f"{model_config.generation_type.value!r} model. Configure a "
                f"ModelConfig with EmbeddingInferenceParams for this alias."
            )

    def resolve_items(self, data: dict) -> list[Any]:
        """Return the list of items to deduplicate from a row dict.

        Args:
            data: Row dict containing the configured source column.

        Returns:
            The list referenced by ``source_column`` and (optionally)
            ``items_key``; an empty list if the source value is missing.

        Raises:
            TypeError: If the resolved value is not a list.
        """
        value = data.get(self.config.source_column)
        if self.config.items_key is not None:
            if value is None:
                return []
            value = value[self.config.items_key] if isinstance(value, dict) else getattr(value, self.config.items_key)
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError(
                f"EmbeddingDedupColumnGenerator expected a list at "
                f"{self.config.source_column!r}"
                f"{f'[{self.config.items_key!r}]' if self.config.items_key else ''}, "
                f"got {type(value).__name__}"
            )
        return value

    def extract_text(self, item: Any) -> str:
        """Pull the text field from an item.

        Supports dict items and Pydantic / attribute-style items.

        Args:
            item: One element of the resolved items list.

        Returns:
            The text to embed for similarity comparison.
        """
        field = self.config.text_field
        if isinstance(item, dict):
            return str(item.get(field, ""))
        return str(getattr(item, field, ""))

    def dedupe_indices(self, embeddings: list[list[float]]) -> list[int]:
        """Return indices to keep after greedy cosine-similarity dedup.

        Args:
            embeddings: 2-D list of embedding vectors, one per item.

        Returns:
            Sorted list of integer indices to retain.

        Raises:
            ValueError: If ``embeddings`` is not a 2-D structure.
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

        threshold = self.config.similarity_threshold
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

    def log_dedup_outcome(self, kept: int, total: int) -> None:
        """Log dedup statistics at info or debug level."""
        dropped = total - kept
        if dropped > 0:
            logger.info(
                "Dedup: retained %d of %d items (%d duplicates removed)",
                kept,
                total,
                dropped,
            )
        else:
            logger.debug("Dedup: retained all %d items (no duplicates)", total)

    def generate(self, data: dict) -> dict:
        """Synchronous dedup for a single row using the embedding model."""
        items = self.resolve_items(data)
        if not items:
            return data | {self.config.name: []}

        texts = [self.extract_text(item) for item in items]
        embeddings = self.embedder.generate_text_embeddings(input_texts=texts, encoding_format="float")
        retained_indexes = self.dedupe_indices(embeddings)
        self.log_dedup_outcome(len(retained_indexes), len(items))
        return data | {self.config.name: [items[i] for i in retained_indexes]}

    async def agenerate(self, data: dict) -> dict:
        """Async dedup using ``model.agenerate_text_embeddings``.

        Drives the cell-level concurrency the async engine enables when
        ``DATA_DESIGNER_ASYNC_ENGINE=1``; the framework's sync bridge runs
        this from synchronous callers transparently.
        """
        items = self.resolve_items(data)
        if not items:
            return data | {self.config.name: []}

        texts = [self.extract_text(item) for item in items]
        embeddings = await self.embedder.agenerate_text_embeddings(input_texts=texts, encoding_format="float")
        retained_indexes = self.dedupe_indices(embeddings)
        self.log_dedup_outcome(len(retained_indexes), len(items))
        return data | {self.config.name: [items[i] for i in retained_indexes]}
