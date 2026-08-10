# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer plugin registrations exported by this package.

Three ``data_designer.plugins`` entry points are wired here:

- :data:`embedding_dedup_plugin` -- generic embedding-cosine-similarity
  deduplication column generator (``column_type="embedding-dedup"``).
- :data:`document_chunker_plugin` -- filesystem seed reader that loads
  text files, chunks them by sentence, and emits structured sections
  (``seed_type="document-chunker"``).
- :data:`qa_evaluation_normalizer_plugin` -- post-batch processor that
  stabilizes QA evaluation score types before Parquet checkpointing
  (``processor_type="qa-evaluation-normalizer"``).
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

qa_evaluation_normalizer_plugin = Plugin(
    config_qualified_name="data_designer_retrieval_sdg.config.QAEvaluationNormalizerConfig",
    impl_qualified_name="data_designer_retrieval_sdg.evaluation_normalizer.QAEvaluationNormalizer",
    plugin_type=PluginType.PROCESSOR,
)
