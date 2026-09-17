# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer plugins and pipelines for retrieval synthetic data generation.

The package registers the embedding-dedup and document-chunker plugin entry
points. It also provides one retrieval pipeline for text and optional images,
recipe-oriented conversion helpers, and reusable post-processing utilities.
"""

from importlib import import_module
from typing import Any

from data_designer_retrieval_sdg.config import EmbeddingDedupColumnConfig
from data_designer_retrieval_sdg.pipeline import (
    build_model_providers,
    build_qa_generation_pipeline,
    build_retrieval_pipeline,
)
from data_designer_retrieval_sdg.postprocess import (
    filter_qa_pairs_by_quality,
    load_positive_docs_with_modality,
    postprocess_retriever_data,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

__all__ = [
    "ConfigSource",
    "ConversionResult",
    "ConversionRunConfig",
    "DocumentChunkerSeedSource",
    "EmbeddingDedupColumnConfig",
    "ExportSummary",
    "GenerationPipelineConfig",
    "GenerationPreviewResult",
    "GenerationResult",
    "GenerationRunConfig",
    "GeneratedRetrievalRecord",
    "LoadedRunConfig",
    "RetrievalSource",
    "RetrievalSourcesFile",
    "RetrievalUnit",
    "SplitRatios",
    "build_model_providers",
    "build_qa_generation_pipeline",
    "build_retrieval_pipeline",
    "dump_resolved_config",
    "export_retrieval_data",
    "filter_qa_pairs_by_quality",
    "load_conversion_config",
    "load_generation_config",
    "load_positive_docs_with_modality",
    "load_retrieval_sources",
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
    "ExportSummary": ("data_designer_retrieval_sdg.retrieval", "ExportSummary"),
    "GenerationPreviewResult": ("data_designer_retrieval_sdg.generation", "GenerationPreviewResult"),
    "GenerationResult": ("data_designer_retrieval_sdg.generation", "GenerationResult"),
    "GenerationPipelineConfig": ("data_designer_retrieval_sdg.run_config", "GenerationPipelineConfig"),
    "GenerationRunConfig": ("data_designer_retrieval_sdg.run_config", "GenerationRunConfig"),
    "LoadedRunConfig": ("data_designer_retrieval_sdg.run_config", "LoadedRunConfig"),
    "GeneratedRetrievalRecord": ("data_designer_retrieval_sdg.retrieval", "GeneratedRetrievalRecord"),
    "RetrievalSource": ("data_designer_retrieval_sdg.retrieval", "RetrievalSource"),
    "RetrievalSourcesFile": ("data_designer_retrieval_sdg.retrieval.source_file", "RetrievalSourcesFile"),
    "load_retrieval_sources": ("data_designer_retrieval_sdg.retrieval.source_file", "load_retrieval_sources"),
    "RetrievalUnit": ("data_designer_retrieval_sdg.retrieval", "RetrievalUnit"),
    "SplitRatios": ("data_designer_retrieval_sdg.retrieval", "SplitRatios"),
    "dump_resolved_config": ("data_designer_retrieval_sdg.run_config", "dump_resolved_config"),
    "export_retrieval_data": ("data_designer_retrieval_sdg.retrieval", "export_retrieval_data"),
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
