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
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface import DataDesigner
from data_designer.logging import LoggerConfig, LoggingConfig, OutputConfig, configure_logging

from data_designer_retrieval_sdg.convert import run_conversion
from data_designer_retrieval_sdg.pipeline import build_model_providers, build_qa_generation_pipeline
from data_designer_retrieval_sdg.seed_reader import DocumentChunkerSeedReader
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

logger = logging.getLogger(__name__)


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


def _count_seed_records(seed_source: DocumentChunkerSeedSource) -> int:
    """Probe the seed reader for the total number of records it will produce.

    Builds and attaches a temporary reader so the manifest is materialised
    once for batch math without reading any file contents.
    """
    reader = DocumentChunkerSeedReader()
    reader.attach(seed_source, PlaintextResolver())
    return reader.get_seed_dataset_size()


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``generate`` subcommand."""
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
    p.add_argument("--max-artifacts-per-type", type=int, default=2, help="Max artifacts per type")
    p.add_argument("--num-pairs", type=int, default=7, help="QA pairs per document")
    p.add_argument("--min-hops", type=int, default=2, help="Min hops for multi-hop questions")
    p.add_argument("--max-hops", type=int, default=4, help="Max hops for multi-hop questions")
    p.add_argument("--min-complexity", type=int, default=4, help="Min question complexity")
    p.add_argument("--similarity-threshold", type=float, default=0.9, help="Cosine threshold for QA-pair dedup")
    p.add_argument("--preview", action="store_true", help="Preview without full generation")
    p.add_argument("--artifact-path", type=Path, default=Path("./artifacts"), help="DD artifact path")
    p.add_argument("--dataset-name", default=None, help="Stable DD dataset name for artifacts and resume")
    p.add_argument("--buffer-size", type=int, default=200, help="DataDesigner checkpoint buffer size")
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
    g.add_argument("--artifact-extraction-model", default="nvidia/nemotron-3-nano-30b-a3b")
    g.add_argument("--artifact-extraction-provider", default="nvidia")
    g.add_argument("--qa-generation-model", default="nvidia/nemotron-3-nano-30b-a3b")
    g.add_argument("--qa-generation-provider", default="nvidia")
    g.add_argument("--quality-judge-model", default="nvidia/nemotron-3-nano-30b-a3b")
    g.add_argument("--quality-judge-provider", default="nvidia")
    g.add_argument("--embed-model", default="nvidia/llama-3.2-nv-embedqa-1b-v2")
    g.add_argument("--embed-provider", default="nvidia")
    g.add_argument("--max-parallel-requests-for-gen", type=int, default=None)

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

    model_providers, custom_providers = build_model_providers(
        custom_provider_endpoint=args.custom_provider_endpoint,
        custom_provider_name=args.custom_provider_name,
        custom_provider_type=args.custom_provider_type,
        custom_provider_api_key=args.custom_provider_api_key,
        model_providers_file=args.model_providers_file,
    )

    data_designer = DataDesigner(artifact_path=args.artifact_path, model_providers=model_providers)
    data_designer.set_run_config(dd.RunConfig(disable_early_shutdown=True, buffer_size=args.buffer_size))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs = _pipeline_kwargs(args)
    _print_model_config(args, custom_providers)

    if args.preview:
        _run_preview(data_designer, seed_source, total_records, args, pipeline_kwargs)
        return

    _run_create(data_designer, seed_source, total_records, args, pipeline_kwargs)


def _pipeline_kwargs(args: argparse.Namespace) -> dict:
    """Collect pipeline-builder keyword arguments shared between preview and create runs."""
    return {
        "max_artifacts_per_type": args.max_artifacts_per_type,
        "num_pairs": args.num_pairs,
        "min_hops": args.min_hops,
        "max_hops": args.max_hops,
        "min_complexity": args.min_complexity,
        "similarity_threshold": args.similarity_threshold,
        "max_parallel_requests_for_gen": args.max_parallel_requests_for_gen,
        "artifact_extraction_model": args.artifact_extraction_model,
        "artifact_extraction_provider": args.artifact_extraction_provider,
        "qa_generation_model": args.qa_generation_model,
        "qa_generation_provider": args.qa_generation_provider,
        "quality_judge_model": args.quality_judge_model,
        "quality_judge_provider": args.quality_judge_provider,
        "embed_model": args.embed_model,
        "embed_provider": args.embed_provider,
    }


def _print_model_config(args: argparse.Namespace, custom_providers: list) -> None:
    """Print model configuration to stdout."""
    print("\nModel configuration:")
    print(f"  Artifact extraction: {args.artifact_extraction_model} ({args.artifact_extraction_provider})")
    print(f"  QA generation:       {args.qa_generation_model} ({args.qa_generation_provider})")
    print(f"  Quality judge:       {args.quality_judge_model} ({args.quality_judge_provider})")
    print(f"  Embedding:           {args.embed_model} ({args.embed_provider})")
    if custom_providers:
        print("\nCustom model providers:")
        for p in custom_providers:
            print(f"  {p.name}: {p.endpoint} (type={p.provider_type}, api_key={p.api_key or 'none'})")


def _run_preview(
    data_designer: DataDesigner,
    seed_source: DocumentChunkerSeedSource,
    total_records: int,
    args: argparse.Namespace,
    pipeline_kwargs: dict,
) -> None:
    """Run a single-record preview of the pipeline."""
    config_builder = build_qa_generation_pipeline(
        seed_source=seed_source,
        start_index=0,
        end_index=min(args.buffer_size - 1, total_records - 1),
        **pipeline_kwargs,
    )
    print("\nPreviewing generation...")
    try:
        preview_result = data_designer.preview(config_builder, num_records=1)
        preview_result.display_sample_record()
    except Exception as e:  # noqa: BLE001 - preview is best-effort UX
        logger.warning("Preview error: %s", e)


def _run_create(
    data_designer: DataDesigner,
    seed_source: DocumentChunkerSeedSource,
    total_records: int,
    args: argparse.Namespace,
    pipeline_kwargs: dict,
) -> None:
    """Run full generation once and export the resulting dataset as JSONL."""
    print(f"\nTotal records: {total_records}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Resume mode: {args.resume}")

    config_builder = build_qa_generation_pipeline(
        seed_source=seed_source,
        start_index=0,
        end_index=total_records - 1,
        **pipeline_kwargs,
    )

    dataset_name = args.dataset_name or args.input_dir.name or "retrieval_sdg"
    print(f"Dataset name: {dataset_name}")
    print("\nGenerating dataset...")
    result = data_designer.create(
        config_builder,
        num_records=total_records,
        dataset_name=dataset_name,
        resume=ResumeMode(args.resume),
    )

    output_path = args.output_dir / f"{result.artifact_storage.resolved_dataset_name}.jsonl"
    result.export(output_path, format="jsonl")

    print(f"\nGeneration complete! Artifacts saved to {result.artifact_storage.base_dataset_path}")
    print(f"Exported JSONL to {output_path}")


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
