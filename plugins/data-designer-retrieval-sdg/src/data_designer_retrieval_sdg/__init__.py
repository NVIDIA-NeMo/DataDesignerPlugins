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

from importlib import import_module
from typing import Any

from data_designer_retrieval_sdg.config import EmbeddingDedupColumnConfig
from data_designer_retrieval_sdg.pipeline import build_model_providers, build_qa_generation_pipeline
from data_designer_retrieval_sdg.postprocess import (
    filter_qa_pairs_by_quality,
    load_positive_docs_with_modality,
    postprocess_retriever_data,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

__all__ = [
    "DocumentChunkerSeedSource",
    "EmbeddingDedupColumnConfig",
    "ConfigSource",
    "ConversionResult",
    "ConversionRunConfig",
    "GenerationResult",
    "GenerationPipelineConfig",
    "GenerationRunConfig",
    "GenerationPreviewResult",
    "LoadedRunConfig",
    "build_model_providers",
    "build_qa_generation_pipeline",
    "dump_resolved_config",
    "filter_qa_pairs_by_quality",
    "load_conversion_config",
    "load_generation_config",
    "load_positive_docs_with_modality",
    "postprocess_retriever_data",
    "preview_generation",
    "run_conversion",
    "run_conversion_with_config",
    "run_generation",
]

_LAZY_EXPORTS = {
    "ConfigSource": ("data_designer_retrieval_sdg.run_config", "ConfigSource"),
    "ConversionResult": ("data_designer_retrieval_sdg.convert", "ConversionResult"),
    "ConversionRunConfig": ("data_designer_retrieval_sdg.run_config", "ConversionRunConfig"),
    "GenerationPreviewResult": ("data_designer_retrieval_sdg.generation", "GenerationPreviewResult"),
    "GenerationResult": ("data_designer_retrieval_sdg.generation", "GenerationResult"),
    "GenerationPipelineConfig": ("data_designer_retrieval_sdg.run_config", "GenerationPipelineConfig"),
    "GenerationRunConfig": ("data_designer_retrieval_sdg.run_config", "GenerationRunConfig"),
    "LoadedRunConfig": ("data_designer_retrieval_sdg.run_config", "LoadedRunConfig"),
    "dump_resolved_config": ("data_designer_retrieval_sdg.run_config", "dump_resolved_config"),
    "load_conversion_config": ("data_designer_retrieval_sdg.run_config", "load_conversion_config"),
    "load_generation_config": ("data_designer_retrieval_sdg.run_config", "load_generation_config"),
    "preview_generation": ("data_designer_retrieval_sdg.generation", "preview_generation"),
    "run_conversion": ("data_designer_retrieval_sdg.convert", "run_conversion"),
    "run_conversion_with_config": ("data_designer_retrieval_sdg.convert", "run_conversion_with_config"),
    "run_generation": ("data_designer_retrieval_sdg.generation", "run_generation"),
}


def __getattr__(name: str) -> Any:
    """Load orchestration APIs lazily so Data Designer can discover plugins safely."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
