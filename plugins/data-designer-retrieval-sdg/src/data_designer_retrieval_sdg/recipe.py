# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer-backed Data Designer recipe entry point for retrieval SDG."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import click
import data_designer.config as dd
import typer

from data_designer_retrieval_sdg.pipeline import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_PROVIDER,
    build_qa_generation_pipeline,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


def load_config_builder(params: dd.DataDesignerScriptParams | None = None) -> dd.DataDesignerConfigBuilder:
    """Build the retrieval SDG pipeline from forwarded Data Designer CLI args.

    Args:
        params: Data Designer script parameters. ``params.argv`` contains the
            arguments supplied after ``data-designer preview/create --recipe
            retrieval-sdg --``.

    Returns:
        A configured Data Designer config builder for retrieval SDG generation.
    """
    argv = list(tuple(getattr(params, "argv", ())))
    command = typer.main.get_command(build_typer_app())
    config_builder = command.main(
        args=argv,
        prog_name="data-designer preview/create --recipe retrieval-sdg --",
        standalone_mode=False,
    )

    if config_builder == 0 and any(arg in {"--help", "-h"} for arg in argv):
        raise SystemExit(0)
    if not isinstance(config_builder, dd.DataDesignerConfigBuilder):
        raise TypeError(f"Recipe returned {type(config_builder).__name__}, expected DataDesignerConfigBuilder")
    return config_builder


def build_typer_app() -> typer.Typer:
    """Build the Typer app used for recipe inspection and execution.

    Returns:
        Typer app describing the retrieval SDG recipe interface.
    """
    app = typer.Typer(add_completion=False, help="Build the retrieval SDG Data Designer workflow.")
    app.command(name=None, help="Build the retrieval SDG Data Designer workflow.")(recipe_command)
    return app


def recipe_command(
    input_dir: Annotated[Path, typer.Option("--input-dir", help="Directory containing text files")],
    file_pattern: Annotated[str, typer.Option("--file-pattern", help="Filename glob (basenames only)")] = "*",
    recursive: Annotated[
        bool,
        typer.Option("--recursive/--no-recursive", help="Enable recursive search"),
    ] = True,
    file_extensions: Annotated[
        list[str] | None,
        typer.Option(
            "--file-extensions",
            help="Allowed file extensions (use empty string '' to match files without extensions)",
        ),
    ] = None,
    min_text_length: Annotated[int, typer.Option("--min-text-length", help="Minimum document text length")] = 50,
    sentences_per_chunk: Annotated[int, typer.Option("--sentences-per-chunk", help="Sentences per chunk")] = 5,
    num_sections: Annotated[int, typer.Option("--num-sections", help="Sections to divide chunks into")] = 1,
    num_files: Annotated[int | None, typer.Option("--num-files", help="Max files to process")] = None,
    multi_doc: Annotated[bool, typer.Option("--multi-doc", help="Enable multi-doc bundling")] = False,
    bundle_size: Annotated[int, typer.Option("--bundle-size", help="Docs per bundle")] = 2,
    bundle_strategy: Annotated[
        str,
        typer.Option(
            "--bundle-strategy",
            help="Section splitting strategy",
            click_type=click.Choice(["sequential", "doc_balanced", "interleaved"]),
        ),
    ] = "sequential",
    max_docs_per_bundle: Annotated[int, typer.Option("--max-docs-per-bundle", help="Max docs per bundle")] = 3,
    multi_doc_manifest: Annotated[
        Path | None, typer.Option("--multi-doc-manifest", help="Manifest for explicit bundles")
    ] = None,
    start_index: Annotated[int, typer.Option("--start-index", help="Start seed row index")] = 0,
    end_index: Annotated[int, typer.Option("--end-index", help="End seed row index")] = 199,
    max_artifacts_per_type: Annotated[int, typer.Option("--max-artifacts-per-type", help="Max artifacts per type")] = 2,
    num_pairs: Annotated[int, typer.Option("--num-pairs", help="QA pairs per document")] = 7,
    min_hops: Annotated[int, typer.Option("--min-hops", help="Min hops for multi-hop questions")] = 2,
    max_hops: Annotated[int, typer.Option("--max-hops", help="Max hops for multi-hop questions")] = 4,
    min_complexity: Annotated[int, typer.Option("--min-complexity", help="Min question complexity")] = 4,
    similarity_threshold: Annotated[
        float, typer.Option("--similarity-threshold", help="Cosine threshold for QA-pair dedup")
    ] = 0.9,
    artifact_extraction_model: Annotated[
        str, typer.Option("--artifact-extraction-model", help="Artifact extraction model")
    ] = DEFAULT_CHAT_MODEL,
    artifact_extraction_provider: Annotated[
        str, typer.Option("--artifact-extraction-provider", help="Artifact extraction provider")
    ] = DEFAULT_PROVIDER,
    qa_generation_model: Annotated[str, typer.Option("--qa-generation-model", help="QA generation model")] = (
        DEFAULT_CHAT_MODEL
    ),
    qa_generation_provider: Annotated[str, typer.Option("--qa-generation-provider", help="QA generation provider")] = (
        DEFAULT_PROVIDER
    ),
    quality_judge_model: Annotated[str, typer.Option("--quality-judge-model", help="Quality judge model")] = (
        DEFAULT_CHAT_MODEL
    ),
    quality_judge_provider: Annotated[str, typer.Option("--quality-judge-provider", help="Quality judge provider")] = (
        DEFAULT_PROVIDER
    ),
    embed_model: Annotated[str, typer.Option("--embed-model", help="Embedding model")] = DEFAULT_EMBED_MODEL,
    embed_provider: Annotated[str, typer.Option("--embed-provider", help="Embedding provider")] = DEFAULT_PROVIDER,
    max_parallel_requests_for_gen: Annotated[
        int | None, typer.Option("--max-parallel-requests-for-gen", help="Max parallel generation requests")
    ] = None,
) -> dd.DataDesignerConfigBuilder:
    """Build the retrieval SDG Data Designer workflow.

    Returns:
        A configured Data Designer config builder.
    """
    if end_index < start_index:
        raise click.BadParameter("--end-index must be greater than or equal to --start-index")

    seed_source = DocumentChunkerSeedSource(
        path=str(input_dir),
        file_pattern=file_pattern,
        recursive=recursive,
        file_extensions=file_extensions or [".txt", ".md", ".text"],
        min_text_length=min_text_length,
        sentences_per_chunk=sentences_per_chunk,
        num_sections=num_sections,
        num_files=num_files,
        multi_doc=multi_doc,
        bundle_size=bundle_size,
        bundle_strategy=bundle_strategy,
        max_docs_per_bundle=max_docs_per_bundle,
        multi_doc_manifest=str(multi_doc_manifest) if multi_doc_manifest else None,
    )

    return build_qa_generation_pipeline(
        seed_source=seed_source,
        start_index=start_index,
        end_index=end_index,
        max_artifacts_per_type=max_artifacts_per_type,
        num_pairs=num_pairs,
        min_hops=min_hops,
        max_hops=max_hops,
        min_complexity=min_complexity,
        similarity_threshold=similarity_threshold,
        max_parallel_requests_for_gen=max_parallel_requests_for_gen,
        artifact_extraction_model=artifact_extraction_model,
        artifact_extraction_provider=artifact_extraction_provider,
        qa_generation_model=qa_generation_model,
        qa_generation_provider=qa_generation_provider,
        quality_judge_model=quality_judge_model,
        quality_judge_provider=quality_judge_provider,
        embed_model=embed_model,
        embed_provider=embed_provider,
    )
