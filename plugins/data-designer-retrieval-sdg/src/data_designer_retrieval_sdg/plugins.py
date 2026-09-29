# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer plugin registrations exported by this package.

Three ``data_designer.plugins`` entry points are wired here:

- :data:`embedding_dedup_plugin` -- generic embedding-cosine-similarity
  deduplication column generator (``column_type="embedding-dedup"``).
- :data:`document_chunker_plugin` -- filesystem seed reader that loads
  text files, chunks them by sentence, and emits structured sections
  (``seed_type="document-chunker"``).
- :data:`retrieval_structured_plugin` -- native structured generation with
  strict bare-or-fenced JSON decoding (``column_type="retrieval-structured"``).
"""

from data_designer.plugins.plugin import Plugin, PluginType

embedding_dedup_plugin = Plugin(
    config_qualified_name="data_designer_retrieval_sdg.config.EmbeddingDedupColumnConfig",
    impl_qualified_name="data_designer_retrieval_sdg.dedup.EmbeddingDedupColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

document_chunker_plugin = Plugin(
    config_qualified_name="data_designer_retrieval_sdg.seed_source.DocumentChunkerSeedSource",
    impl_qualified_name="data_designer_retrieval_sdg.seed_reader.DocumentChunkerSeedReader",
    plugin_type=PluginType.SEED_READER,
)

retrieval_structured_plugin = Plugin(
    config_qualified_name="data_designer_retrieval_sdg.config.RetrievalStructuredColumnConfig",
    impl_qualified_name="data_designer_retrieval_sdg.structured.RetrievalStructuredCellGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
