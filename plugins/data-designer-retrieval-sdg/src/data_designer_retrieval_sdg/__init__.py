# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer plugin for retriever synthetic data generation.

Provides a multi-step pipeline that generates QA pairs from text documents
for retriever finetuning, plus utilities for converting raw SDG output into
Automodel-compatible training formats.

Public API:

- :func:`build_qa_generation_pipeline` -- build the four-column DD pipeline
- :func:`load_text_files_from_directory` -- load and chunk text files
- :func:`postprocess_retriever_data` -- flatten to BEIR format
- :func:`filter_qa_pairs_by_quality` -- quality-based filtering
- :func:`load_positive_docs_with_modality` -- load BEIR docs with modality
"""

from data_designer_retrieval_sdg.ingest import load_text_files_from_directory
from data_designer_retrieval_sdg.pipeline import build_qa_generation_pipeline
from data_designer_retrieval_sdg.postprocess import (
    filter_qa_pairs_by_quality,
    load_positive_docs_with_modality,
    postprocess_retriever_data,
)

__all__ = [
    "build_qa_generation_pipeline",
    "filter_qa_pairs_by_quality",
    "load_positive_docs_with_modality",
    "load_text_files_from_directory",
    "postprocess_retriever_data",
]
