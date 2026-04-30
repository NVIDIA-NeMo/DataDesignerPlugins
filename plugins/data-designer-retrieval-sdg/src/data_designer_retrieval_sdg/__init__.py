# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer plugins and pipeline for retriever synthetic data generation.

The package registers two ``data_designer.plugins`` entry points:

- ``embedding-dedup``: generic embedding-cosine-similarity column generator.
- ``document-chunker``: filesystem seed reader that loads text files,
  sentence-chunks them, and emits structured sections.

It also ships a ready-made four-column QA generation pipeline, a CLI for
running the pipeline end-to-end (``generate``) and exporting to NeMo
Retriever / BEIR formats (``convert``), and reusable post-processing
helpers.
"""

from data_designer_retrieval_sdg.config import EmbeddingDedupColumnConfig
from data_designer_retrieval_sdg.pipeline import build_qa_generation_pipeline
from data_designer_retrieval_sdg.postprocess import (
    filter_qa_pairs_by_quality,
    load_positive_docs_with_modality,
    postprocess_retriever_data,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

__all__ = [
    "DocumentChunkerSeedSource",
    "EmbeddingDedupColumnConfig",
    "build_qa_generation_pipeline",
    "filter_qa_pairs_by_quality",
    "load_positive_docs_with_modality",
    "postprocess_retriever_data",
]
