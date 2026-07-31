# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry points for the data-designer-retrieval-sdg package.

Provides two subcommands:

- ``generate`` -- run the full SDG pipeline on a directory of text files
- ``convert``  -- convert raw SDG output to Automodel-compatible formats

The ``generate`` subcommand runs the full pipeline through DataDesigner's
native resumable generation support.  The framework owns discovery, chunking,
checkpointing, and async cell scheduling.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import data_designer.config as dd
from data_designer.engine.resources.seed_reader import SeedReaderError
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.logging import LoggerConfig, LoggingConfig, OutputConfig, configure_logging

from data_designer_retrieval_sdg.convert import run_conversion
from data_designer_retrieval_sdg.generation import (
    _count_seed_records,
    _resolve_dataset_name,
    preview_generation,
    run_generation,
)
from data_designer_retrieval_sdg.pipeline import build_model_providers
from data_designer_retrieval_sdg.run_config import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_BUFFER_SIZE,
    GenerationPipelineConfig,
    GenerationRunConfig,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

logger = logging.getLogger(__name__)


def _parse_count_entry(value: str) -> tuple[str, int]:
    """Parse one ``NAME=COUNT`` question-distribution entry."""
    name, separator, raw_count = value.partition("=")
    if not separator or not name or not raw_count:
        raise argparse.ArgumentTypeError("expected NAME=COUNT")
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"count must be an integer, got {raw_count!r}") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("count must be non-negative")
    return name, count


def _build_seed_source(args: argparse.Namespace) -> DocumentChunkerSeedSource:
    """Construct a :class:`DocumentChunkerSeedSource` from CLI arguments."""
    return DocumentChunkerSeedSource(
        path=str(args.input_dir),
        file_pattern=args.file_pattern,
        recursive=args.recursive,
        file_extensions=args.file_extensions,
        min_text_length=args.min_text_length,
        sentences_per_chunk=args.sentences_per_chunk,
        num_sections=args.num_sections,
        num_files=args.num_files,
        multi_doc=args.multi_doc,
        bundle_size=args.bundle_size,
        bundle_strategy=args.bundle_strategy,
        max_docs_per_bundle=args.max_docs_per_bundle,
        multi_doc_manifest=str(args.multi_doc_manifest) if args.multi_doc_manifest else None,
    )


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``generate`` subcommand."""
    defaults = GenerationPipelineConfig()
    p = subparsers.add_parser(
        "generate",
        help="Generate synthetic QA pairs from a directory of text files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing text files")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to save generated output")
    p.add_argument("--file-pattern", default="*", help="Filename glob (basenames only)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursive search")
    p.set_defaults(recursive=True)
    p.add_argument(
        "--file-extensions",
        nargs="+",
        default=[".txt", ".md", ".text"],
        help="Allowed file extensions (use empty string '' to match files without extensions)",
    )
    p.add_argument("--min-text-length", type=int, default=50, help="Minimum document text length")
    p.add_argument("--sentences-per-chunk", type=int, default=5, help="Sentences per chunk")
    p.add_argument("--num-sections", type=int, default=1, help="Sections to divide chunks into")
    p.add_argument("--num-files", type=int, default=None, help="Max files to process")
    p.add_argument(
        "--max-artifacts-per-type",
        type=int,
        default=defaults.max_artifacts_per_type,
        help="Max artifacts per type",
    )
    p.add_argument("--num-pairs", type=int, default=defaults.num_pairs, help="QA pairs per document")
    p.add_argument(
        "--query-counts",
        nargs="+",
        type=_parse_count_entry,
        default=list(defaults.query_counts.items()),
        metavar="NAME=COUNT",
        help="Exact query-type counts; values must sum to --num-pairs",
    )
    p.add_argument(
        "--reasoning-counts",
        nargs="+",
        type=_parse_count_entry,
        default=list(defaults.reasoning_counts.items()),
        metavar="NAME=COUNT",
        help="Exact reasoning-type counts; values must sum to --num-pairs",
    )
    p.add_argument("--min-hops", type=int, default=defaults.min_hops, help="Min hops for multi-hop questions")
    p.add_argument("--max-hops", type=int, default=defaults.max_hops, help="Max hops for multi-hop questions")
    p.add_argument(
        "--min-complexity",
        type=int,
        default=defaults.min_complexity,
        help="Min question complexity",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=defaults.similarity_threshold,
        help="Cosine threshold for QA-pair dedup",
    )
    p.add_argument("--preview", action="store_true", help="Preview without full generation")
    p.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH, help="DD artifact path")
    p.add_argument("--dataset-name", default=None, help="Stable DD dataset name for artifacts and resume")
    p.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="DataDesigner checkpoint buffer size")
    p.add_argument(
        "--resume",
        "-r",
        choices=[mode.value for mode in ResumeMode],
        default=ResumeMode.NEVER.value,
        help="Resume behavior for interrupted generation runs",
    )

    g = p.add_argument_group("multi-document bundling")
    g.add_argument("--multi-doc", action="store_true", help="Enable multi-doc bundling")
    g.add_argument("--bundle-size", type=int, default=2, help="Docs per bundle")
    g.add_argument(
        "--bundle-strategy",
        choices=["sequential", "doc_balanced", "interleaved"],
        default="sequential",
        help="Section splitting strategy",
    )
    g.add_argument("--max-docs-per-bundle", type=int, default=3, help="Max docs per bundle")
    g.add_argument("--multi-doc-manifest", type=Path, default=None, help="Manifest for explicit bundles")

    g = p.add_argument_group("logging")
    g.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    g = p.add_argument_group("model configuration")
    g.add_argument("--artifact-extraction-model", default=defaults.artifact_extraction_model)
    g.add_argument("--artifact-extraction-provider", default=defaults.artifact_extraction_provider)
    g.add_argument("--qa-generation-model", default=defaults.qa_generation_model)
    g.add_argument("--qa-generation-provider", default=defaults.qa_generation_provider)
    g.add_argument("--quality-judge-model", default=defaults.quality_judge_model)
    g.add_argument("--quality-judge-provider", default=defaults.quality_judge_provider)
    g.add_argument("--embed-model", default=defaults.embed_model)
    g.add_argument("--embed-provider", default=defaults.embed_provider)
    g.add_argument("--max-parallel-requests-for-gen", type=int, default=defaults.max_parallel_requests_for_gen)

    g = p.add_argument_group("custom provider")
    g.add_argument("--custom-provider-endpoint", default=None, help="Base URL for custom provider")
    g.add_argument("--custom-provider-name", default="custom")
    g.add_argument("--custom-provider-type", default="openai")
    g.add_argument("--custom-provider-api-key", default=None)
    g.add_argument("--model-providers-file", type=Path, default=None, help="YAML/JSON providers file")

    p.set_defaults(func=_run_generate)


def _run_generate(args: argparse.Namespace) -> None:
    """Execute the ``generate`` subcommand."""
    configure_logging(
        LoggingConfig(
            logger_configs=[LoggerConfig(name="data_designer", level=args.log_level)],
            output_configs=[OutputConfig(destination=sys.stderr, structured=(args.log_level == "DEBUG"))],
            root_level=args.log_level,
        )
    )

    seed_source = _build_seed_source(args)
    try:
        total_records = _count_seed_records(seed_source)
    except SeedReaderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    row_type = "bundles" if args.multi_doc else "text files"
    print(f"Discovered {total_records} {row_type} under {args.input_dir}")

    try:
        args.dataset_name = _resolve_dataset_name(seed_source, args.artifact_path, args.dataset_name)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    model_providers, custom_providers = build_model_providers(
        custom_provider_endpoint=args.custom_provider_endpoint,
        custom_provider_name=args.custom_provider_name,
        custom_provider_type=args.custom_provider_type,
        custom_provider_api_key=args.custom_provider_api_key,
        model_providers_file=args.model_providers_file,
    )

    pipeline_config = _pipeline_config(args)
    _print_model_config(pipeline_config, custom_providers)

    if args.preview:
        _run_preview(seed_source, total_records, args, pipeline_config, model_providers)
        return

    _run_create(seed_source, total_records, args, pipeline_config, model_providers)


def _pipeline_config(args: argparse.Namespace) -> GenerationPipelineConfig:
    """Build validated pipeline settings shared by preview and create runs."""
    return GenerationPipelineConfig(
        max_artifacts_per_type=args.max_artifacts_per_type,
        num_pairs=args.num_pairs,
        query_counts=dict(args.query_counts),
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        reasoning_counts=dict(args.reasoning_counts),
        min_complexity=args.min_complexity,
        similarity_threshold=args.similarity_threshold,
        max_parallel_requests_for_gen=args.max_parallel_requests_for_gen,
        artifact_extraction_model=args.artifact_extraction_model,
        artifact_extraction_provider=args.artifact_extraction_provider,
        qa_generation_model=args.qa_generation_model,
        qa_generation_provider=args.qa_generation_provider,
        quality_judge_model=args.quality_judge_model,
        quality_judge_provider=args.quality_judge_provider,
        embed_model=args.embed_model,
        embed_provider=args.embed_provider,
    )


def _print_model_config(config: GenerationPipelineConfig, custom_providers: list[dd.ModelProvider]) -> None:
    """Print model configuration to stdout."""
    print("\nModel configuration:")
    print(f"  Artifact extraction: {config.artifact_extraction_model} ({config.artifact_extraction_provider})")
    print(f"  QA generation:       {config.qa_generation_model} ({config.qa_generation_provider})")
    print(f"  Quality judge:       {config.quality_judge_model} ({config.quality_judge_provider})")
    print(f"  Embedding:           {config.embed_model} ({config.embed_provider})")
    if custom_providers:
        print("\nCustom model providers:")
        for provider in custom_providers:
            credential_status = "configured" if provider.api_key else "none"
            print(
                f"  {provider.name}: {provider.endpoint} "
                f"(type={provider.provider_type}, credential={credential_status})"
            )


def _run_preview(
    seed_source: DocumentChunkerSeedSource,
    total_records: int,
    args: argparse.Namespace,
    pipeline_config: GenerationPipelineConfig,
    model_providers: list[dd.ModelProvider] | None,
) -> None:
    """Run a single-record preview of the pipeline."""
    print("\nPreviewing generation...")
    try:
        preview_generation(
            GenerationRunConfig(
                seed_source=seed_source,
                output_dir=args.output_dir,
                artifact_path=args.artifact_path,
                dataset_name=args.dataset_name,
                buffer_size=args.buffer_size,
                model_providers=model_providers,
                pipeline=pipeline_config,
                num_records=total_records,
            )
        )
    except Exception as e:  # noqa: BLE001 - preview is best-effort UX
        logger.warning("Preview error: %s", e)


def _run_create(
    seed_source: DocumentChunkerSeedSource,
    total_records: int,
    args: argparse.Namespace,
    pipeline_config: GenerationPipelineConfig,
    model_providers: list[dd.ModelProvider] | None,
) -> None:
    """Run full generation once and export the resulting dataset as JSONL."""
    print(f"\nTotal records: {total_records}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Resume mode: {args.resume}")

    print(f"Dataset name: {args.dataset_name}")
    print("\nGenerating dataset...")
    result = run_generation(
        GenerationRunConfig(
            seed_source=seed_source,
            output_dir=args.output_dir,
            artifact_path=args.artifact_path,
            dataset_name=args.dataset_name,
            buffer_size=args.buffer_size,
            resume=args.resume,
            model_providers=model_providers,
            pipeline=pipeline_config,
            num_records=total_records,
        )
    )

    print(f"\nGeneration complete! Artifacts saved to {result.dataset_path}")
    print(f"Exported JSONL to {result.output_path}")


def _add_convert_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``convert`` subcommand."""
    p = subparsers.add_parser(
        "convert",
        help="Convert SDG output to retriever training/evaluation formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("input_path", help="Path to generated JSONL/JSON/parquet file or output directory")
    p.add_argument("--corpus-id", required=True, help="Corpus identifier")
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--eval-only", action="store_true", help="BEIR eval only (no train/val)")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--quality-threshold", type=float, default=7.0, help="Min quality score")
    p.add_argument("--max-pos-docs", type=int, default=5, help="Max positive docs per query")
    p.add_argument("--use-group-id-in-eval", action="store_true", help="Use group_id in qrels")
    p.add_argument("--split-strategy", choices=["random", "dedupped", "cluster"], default="random")
    p.add_argument("--groups-json", nargs="+", default=None, help="Dedup groups JSON paths")

    p.set_defaults(func=_run_convert)


def _run_convert(args: argparse.Namespace) -> None:
    """Execute the ``convert`` subcommand."""
    run_conversion(
        input_path=args.input_path,
        corpus_id=args.corpus_id,
        output_dir=args.output_dir,
        eval_only=args.eval_only,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        quality_threshold=args.quality_threshold,
        max_pos_docs=args.max_pos_docs,
        use_group_id_in_eval=args.use_group_id_in_eval,
        split_strategy=args.split_strategy,
        groups_json=args.groups_json,
    )


def main() -> None:
    """CLI entry point for ``data-designer-retrieval-sdg``."""
    parser = argparse.ArgumentParser(
        prog="data-designer-retrieval-sdg",
        description="SDG Pipeline for Retriever Evaluation Dataset Generation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_generate_parser(subparsers)
    _add_convert_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
