# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column configuration for the retrieval deduplication plugin."""

from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig


class RetrievalSdgDedupColumnConfig(SingleColumnConfig):
    """Deduplicate QA pairs from a retrieval generation set via embedding similarity.

    This column reads QA pairs from a source column, embeds each question,
    and removes near-duplicates whose cosine similarity exceeds a threshold.

    Args:
        qa_pairs_column: Name of the upstream column containing QA pairs
            with a ``pairs`` key.
        embedding_alias: Model alias registered in the DataDesigner model
            registry to use for computing embeddings.
        column_type: Fixed literal identifying this column type.
        dedupe_similarity_threshold: Cosine similarity threshold above which
            two questions are considered duplicates.  Defaults to ``0.9``.
    """

    qa_pairs_column: str
    embedding_alias: str
    column_type: Literal["retrieval-sdg-dedup"] = "retrieval-sdg-dedup"
    dedupe_similarity_threshold: float = 0.9

    @property
    def required_columns(self) -> list[str]:
        """Columns that must be present before this column can run."""
        return [self.qa_pairs_column]

    @property
    def side_effect_columns(self) -> list[str]:
        """Additional columns produced as side effects."""
        return []

    def get_column_emoji(self) -> str:
        """Emoji displayed in logs for this column type."""
        return "🔍"
