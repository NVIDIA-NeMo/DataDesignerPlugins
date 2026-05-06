# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer recipe registration for the retrieval SDG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.recipes.recipe import DataDesignerRecipe
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from data_designer_retrieval_sdg.pipeline import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_PROVIDER,
    build_qa_generation_pipeline,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


class RetrievalSdgRecipeConfig(BaseModel):
    """Configuration for the installed ``retrieval-sdg`` recipe.

    The recipe is intentionally a thin composition wrapper around the existing
    package pipeline. It gives the main ``data-designer`` CLI a way to discover
    and run the pipeline without adding package-specific subcommands to
    DataDesigner itself.
    """

    input_dir: Path = Field(description="Directory containing input text files.")
    generated_output_dir: Path | None = Field(
        default=None,
        description="Optional directory where the generated dataset is also exported as JSON after create.",
    )
    conversion_output_dir: Path | None = Field(
        default=None,
        description="Optional output directory for retriever-format conversion when corpus_id is set.",
    )
    corpus_id: str | None = Field(
        default=None,
        description="Optional corpus identifier. When set, postprocess also runs retriever data conversion.",
    )

    file_pattern: str = "*"
    recursive: bool = True
    file_extensions: list[str] | None = Field(default_factory=lambda: [".txt", ".md", ".text"])
    min_text_length: int = Field(default=50, ge=0)
    sentences_per_chunk: int = Field(default=5, ge=1)
    num_sections: int = Field(default=1, ge=1)
    num_files: int | None = Field(default=None, ge=1)

    multi_doc: bool = False
    bundle_size: int = Field(default=2, ge=1)
    bundle_strategy: Literal["sequential", "doc_balanced", "interleaved"] = "sequential"
    max_docs_per_bundle: int = Field(default=3, ge=1)
    multi_doc_manifest: Path | None = None

    start_index: int = Field(default=0, ge=0)
    end_index: int = Field(default=199, ge=0)
    max_artifacts_per_type: int = Field(default=2, ge=1)
    num_pairs: int = Field(default=7, ge=1)
    min_hops: int = Field(default=2, ge=1)
    max_hops: int = Field(default=4, ge=1)
    min_complexity: int = Field(default=4, ge=1)
    similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    max_parallel_requests_for_gen: int | None = Field(default=None, ge=1)

    artifact_extraction_model: str = DEFAULT_CHAT_MODEL
    artifact_extraction_provider: str = DEFAULT_PROVIDER
    qa_generation_model: str = DEFAULT_CHAT_MODEL
    qa_generation_provider: str = DEFAULT_PROVIDER
    quality_judge_model: str = DEFAULT_CHAT_MODEL
    quality_judge_provider: str = DEFAULT_PROVIDER
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_provider: str = DEFAULT_PROVIDER

    @model_validator(mode="after")
    def validate_index_range(self) -> Self:
        """Validate the configured seed index range."""
        if self.end_index < self.start_index:
            raise ValueError("end_index must be greater than or equal to start_index")
        return self


def build_recipe_config(recipe_config: BaseModel) -> DataDesignerConfigBuilder:
    """Build the retrieval SDG Data Designer config from a recipe config."""
    config = cast(RetrievalSdgRecipeConfig, recipe_config)
    seed_source = DocumentChunkerSeedSource(
        path=str(config.input_dir),
        file_pattern=config.file_pattern,
        recursive=config.recursive,
        file_extensions=config.file_extensions,
        min_text_length=config.min_text_length,
        sentences_per_chunk=config.sentences_per_chunk,
        num_sections=config.num_sections,
        num_files=config.num_files,
        multi_doc=config.multi_doc,
        bundle_size=config.bundle_size,
        bundle_strategy=config.bundle_strategy,
        max_docs_per_bundle=config.max_docs_per_bundle,
        multi_doc_manifest=str(config.multi_doc_manifest) if config.multi_doc_manifest else None,
    )
    return build_qa_generation_pipeline(
        seed_source=seed_source,
        start_index=config.start_index,
        end_index=config.end_index,
        max_artifacts_per_type=config.max_artifacts_per_type,
        num_pairs=config.num_pairs,
        min_hops=config.min_hops,
        max_hops=config.max_hops,
        min_complexity=config.min_complexity,
        similarity_threshold=config.similarity_threshold,
        max_parallel_requests_for_gen=config.max_parallel_requests_for_gen,
        artifact_extraction_model=config.artifact_extraction_model,
        artifact_extraction_provider=config.artifact_extraction_provider,
        qa_generation_model=config.qa_generation_model,
        qa_generation_provider=config.qa_generation_provider,
        quality_judge_model=config.quality_judge_model,
        quality_judge_provider=config.quality_judge_provider,
        embed_model=config.embed_model,
        embed_provider=config.embed_provider,
    )


def postprocess_recipe_results(results: Any, recipe_config: BaseModel) -> None:
    """Export generated recipe results and optionally run retriever conversion."""
    config = cast(RetrievalSdgRecipeConfig, recipe_config)
    if config.generated_output_dir is None:
        return

    config.generated_output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = config.generated_output_dir / "generated.json"
    generated_df = results.load_dataset()
    generated_df.to_json(generated_path, orient="records", indent=2)

    if config.corpus_id is None:
        return

    from data_designer_retrieval_sdg.convert import run_conversion

    run_conversion(
        input_path=str(generated_path),
        corpus_id=config.corpus_id,
        output_dir=str(config.conversion_output_dir) if config.conversion_output_dir else None,
    )


recipe = DataDesignerRecipe(
    name="retrieval-sdg",
    description="Generate synthetic retriever QA data from text documents.",
    config_model=RetrievalSdgRecipeConfig,
    build_config=build_recipe_config,
    postprocess=postprocess_recipe_results,
)
