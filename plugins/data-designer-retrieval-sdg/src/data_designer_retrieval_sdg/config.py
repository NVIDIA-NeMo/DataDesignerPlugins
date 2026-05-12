# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column configuration for the embedding-dedup plugin."""

from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig


class EmbeddingDedupColumnConfig(SingleColumnConfig):
    """Deduplicate items in a list-valued column via embedding cosine similarity.

    The column reads a list of items from ``source_column``, embeds a chosen
    text field on each item, computes pairwise cosine similarity, and greedily
    drops items above ``similarity_threshold``.  ``items_key`` selects whether
    the source column is a wrapper dict (``data[source_column][items_key]``)
    or a bare list (``items_key=None``).

    Attributes:
        source_column: Name of the upstream column containing the items to
            deduplicate.
        items_key: Key under ``source_column`` that holds the list of items.
            Set to ``None`` when ``source_column`` already evaluates to a list.
            Defaults to ``"pairs"`` for compatibility with the QA-pair shape.
        text_field: Attribute or dictionary key on each item that should be
            embedded for similarity comparison.  Defaults to ``"question"``.
        model_alias: Model alias registered in the DataDesigner model
            registry to use for computing embeddings.
        column_type: Fixed literal identifying this column type.
        similarity_threshold: Cosine similarity threshold above which two
            items are considered duplicates.  Defaults to ``0.9``.
    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        drop: If True, generate this column but remove it from the final dataset.
    """

    source_column: str
    items_key: str | None = "pairs"
    text_field: str = "question"
    model_alias: str
    column_type: Literal["embedding-dedup"] = "embedding-dedup"
    similarity_threshold: float = 0.9

    @property
    def required_columns(self) -> list[str]:
        """Columns that must be present before this column can run."""
        return [self.source_column]

    @property
    def side_effect_columns(self) -> list[str]:
        """Additional columns produced as side effects."""
        return []

    @staticmethod
    def get_column_emoji() -> str:
        """Emoji displayed in logs for this column type."""
        return "🔍"
