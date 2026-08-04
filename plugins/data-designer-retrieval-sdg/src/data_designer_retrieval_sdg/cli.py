# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry points for retrieval generation and data conversion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import data_designer.config as dd
from data_designer.engine.resources.seed_reader import SeedReaderError
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.logging import LoggerConfig, LoggingConfig, OutputConfig, configure_logging

from data_designer_retrieval_sdg.convert import run_conversion_with_config
from data_designer_retrieval_sdg.generation import (
    _count_seed_records,
    _resolve_dataset_name,
    preview_generation,
    run_generation,
)
from data_designer_retrieval_sdg.pipeline import build_model_providers
from data_designer_retrieval_sdg.run_config import (
    LoadedRunConfig,
    config_source_from_path,
    dump_resolved_config,
    load_conversion_config,
    load_generation_config,
)

logger = logging.getLogger(__name__)
_SUPPRESS = argparse.SUPPRESS


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


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add file loading and final-override flags shared by both commands."""
    parser.add_argument("--config", type=Path, help="User YAML/JSON config layered over the packaged default")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Final dotted config override; repeat as needed",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Validate and print the redacted effective config without running",
    )


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``generate`` subcommand."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic QA pairs from a directory of text files",
    )
    _add_config_arguments(parser)

    parser.add_argument("--input-dir", type=Path, default=_SUPPRESS, help="Directory containing text files")
    parser.add_argument("--output-dir", type=Path, default=_SUPPRESS, help="Directory for generated JSONL")
    parser.add_argument("--file-pattern", default=_SUPPRESS, help="Filename glob applied to basenames")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=_SUPPRESS,
        help="Enable or disable recursive search",
    )
    parser.add_argument(
        "--file-extensions",
        nargs="+",
        default=_SUPPRESS,
        help="Allowed extensions; use an empty string to match extensionless files",
    )
    parser.add_argument("--min-text-length", type=int, default=_SUPPRESS)
    parser.add_argument("--sentences-per-chunk", type=int, default=_SUPPRESS)
    parser.add_argument("--num-sections", type=int, default=_SUPPRESS)
    parser.add_argument("--num-files", type=int, default=_SUPPRESS)
    parser.add_argument(
        "--num-records",
        type=int,
        default=_SUPPRESS,
        help="Maximum seed records to process; defaults to all discovered records",
    )
    parser.add_argument("--max-artifacts-per-type", type=int, default=_SUPPRESS)
    parser.add_argument("--num-pairs", type=int, default=_SUPPRESS)
    parser.add_argument(
        "--query-counts",
        nargs="+",
        type=_parse_count_entry,
        default=_SUPPRESS,
        metavar="NAME=COUNT",
        help="Exact query-type counts; values must sum to --num-pairs",
    )
    parser.add_argument(
        "--reasoning-counts",
        nargs="+",
        type=_parse_count_entry,
        default=_SUPPRESS,
        metavar="NAME=COUNT",
        help="Exact reasoning-type counts; values must sum to --num-pairs",
    )
    parser.add_argument("--min-hops", type=int, default=_SUPPRESS)
    parser.add_argument("--max-hops", type=int, default=_SUPPRESS)
    parser.add_argument("--min-complexity", type=int, default=_SUPPRESS)
    parser.add_argument("--similarity-threshold", type=float, default=_SUPPRESS)
    parser.add_argument("--preview", action="store_true", help="Preview without full generation")
    parser.add_argument("--artifact-path", type=Path, default=_SUPPRESS)
    parser.add_argument("--dataset-name", default=_SUPPRESS)
    parser.add_argument("--buffer-size", type=int, default=_SUPPRESS)
    parser.add_argument(
        "--resume",
        "-r",
        choices=[mode.value for mode in ResumeMode],
        default=_SUPPRESS,
    )

    group = parser.add_argument_group("multi-document bundling")
    group.add_argument("--multi-doc", action=argparse.BooleanOptionalAction, default=_SUPPRESS)
    group.add_argument("--bundle-size", type=int, default=_SUPPRESS)
    group.add_argument(
        "--bundle-strategy",
        choices=["sequential", "doc_balanced", "interleaved"],
        default=_SUPPRESS,
    )
    group.add_argument("--max-docs-per-bundle", type=int, default=_SUPPRESS)
    group.add_argument("--multi-doc-manifest", type=Path, default=_SUPPRESS)

    group = parser.add_argument_group("logging")
    group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=_SUPPRESS)

    group = parser.add_argument_group("model configuration")
    group.add_argument("--artifact-extraction-model", default=_SUPPRESS)
    group.add_argument("--artifact-extraction-provider", default=_SUPPRESS)
    group.add_argument("--qa-generation-model", default=_SUPPRESS)
    group.add_argument("--qa-generation-provider", default=_SUPPRESS)
    group.add_argument("--quality-judge-model", default=_SUPPRESS)
    group.add_argument("--quality-judge-provider", default=_SUPPRESS)
    group.add_argument("--embed-model", default=_SUPPRESS)
    group.add_argument("--embed-provider", default=_SUPPRESS)
    group.add_argument("--max-parallel-requests-for-gen", type=int, default=_SUPPRESS)

    group = parser.add_argument_group("custom provider")
    group.add_argument("--custom-provider-endpoint", default=_SUPPRESS)
    group.add_argument("--custom-provider-name", default=_SUPPRESS)
    group.add_argument("--custom-provider-type", default=_SUPPRESS)
    group.add_argument("--custom-provider-api-key", default=_SUPPRESS)
    group.add_argument("--model-providers-file", type=Path, default=_SUPPRESS)

    parser.set_defaults(func=_run_generate)


def _assign_if_present(
    args: argparse.Namespace,
    target: dict[str, Any],
    argument: str,
    *,
    config_key: str | None = None,
    transform: Any = None,
) -> None:
    """Copy an explicitly supplied CLI argument into one override mapping."""
    if not hasattr(args, argument):
        return
    value = getattr(args, argument)
    if transform is not None:
        value = transform(value)
    target[config_key or argument] = value


def _generation_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate only explicitly supplied legacy generation flags."""
    overrides: dict[str, Any] = {}
    seed_source: dict[str, Any] = {}
    pipeline: dict[str, Any] = {}

    _assign_if_present(args, seed_source, "input_dir", config_key="path", transform=str)
    for argument in (
        "file_pattern",
        "recursive",
        "file_extensions",
        "min_text_length",
        "sentences_per_chunk",
        "num_sections",
        "num_files",
        "multi_doc",
        "bundle_size",
        "bundle_strategy",
        "max_docs_per_bundle",
    ):
        _assign_if_present(args, seed_source, argument)
    _assign_if_present(args, seed_source, "multi_doc_manifest", transform=str)
    if seed_source:
        overrides["seed_source"] = seed_source

    for argument in (
        "output_dir",
        "artifact_path",
        "dataset_name",
        "buffer_size",
        "resume",
        "num_records",
        "log_level",
    ):
        _assign_if_present(args, overrides, argument)

    for argument in (
        "max_artifacts_per_type",
        "num_pairs",
        "min_hops",
        "max_hops",
        "min_complexity",
        "similarity_threshold",
        "max_parallel_requests_for_gen",
        "artifact_extraction_model",
        "artifact_extraction_provider",
        "qa_generation_model",
        "qa_generation_provider",
        "quality_judge_model",
        "quality_judge_provider",
        "embed_model",
        "embed_provider",
    ):
        _assign_if_present(args, pipeline, argument)
    _assign_if_present(args, pipeline, "query_counts", transform=dict)
    _assign_if_present(args, pipeline, "reasoning_counts", transform=dict)
    if pipeline:
        overrides["pipeline"] = pipeline
    return overrides


def _load_generation_from_args(args: argparse.Namespace) -> tuple[LoadedRunConfig, list[dd.ModelProvider]]:
    """Resolve generation files, ordinary flags, dotted overrides, and providers."""
    loaded = load_generation_config(
        args.config,
        cli_overrides=_generation_cli_overrides(args),
        set_overrides=args.set_overrides,
    )
    provider_file = getattr(args, "model_providers_file", None)
    provider_flags_present = any(
        hasattr(args, argument)
        for argument in (
            "custom_provider_endpoint",
            "custom_provider_name",
            "custom_provider_type",
            "custom_provider_api_key",
            "model_providers_file",
        )
    )
    model_providers, custom_providers = build_model_providers(
        model_providers=loaded.config.model_providers,
        custom_provider_endpoint=getattr(args, "custom_provider_endpoint", None),
        custom_provider_name=getattr(args, "custom_provider_name", "custom"),
        custom_provider_type=getattr(args, "custom_provider_type", "openai"),
        custom_provider_api_key=getattr(args, "custom_provider_api_key", None),
        model_providers_file=provider_file,
    )
    sources = list(loaded.sources)
    override_paths = list(loaded.override_paths)
    if provider_file is not None:
        sources.append(config_source_from_path(provider_file))
    if provider_flags_present:
        override_paths.append("model_providers")
    return (
        LoadedRunConfig(
            config=loaded.config.model_copy(update={"model_providers": model_providers}),
            sources=tuple(sources),
            override_paths=tuple(dict.fromkeys(override_paths)),
            environment_variables=loaded.environment_variables,
        ),
        custom_providers,
    )


def _configure_logging(log_level: str) -> None:
    """Configure Data Designer and root logging from the resolved run config."""
    configure_logging(
        LoggingConfig(
            logger_configs=[LoggerConfig(name="data_designer", level=log_level)],
            output_configs=[OutputConfig(destination=sys.stderr, structured=(log_level == "DEBUG"))],
            root_level=log_level,
        )
    )


def _print_model_config(config: Any, custom_providers: list[dd.ModelProvider]) -> None:
    """Print model selection and credential presence without exposing secrets."""
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


def _run_generate(args: argparse.Namespace) -> None:
    """Execute generation from one fully resolved typed config."""
    explicit_seed_path = (
        args.config is not None
        or hasattr(args, "input_dir")
        or any(expression.startswith("seed_source.path=") for expression in args.set_overrides)
    )
    if not args.print_resolved_config and not explicit_seed_path:
        print("Error: provide --config, --input-dir, or --set seed_source.path=... for generation", file=sys.stderr)
        raise SystemExit(2)

    try:
        loaded, custom_providers = _load_generation_from_args(args)
        dataset_name = _resolve_dataset_name(
            loaded.config.seed_source,
            loaded.config.artifact_path,
            loaded.config.dataset_name,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    config = loaded.config.model_copy(update={"dataset_name": dataset_name})
    if args.print_resolved_config:
        print(dump_resolved_config(config), end="")
        return

    _configure_logging(config.log_level)
    try:
        available_records = _count_seed_records(config.seed_source)
    except SeedReaderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    requested_records = config.num_records if config.num_records is not None else available_records
    if requested_records > available_records:
        print(
            f"Error: num_records={requested_records} exceeds the {available_records} available seed records",
            file=sys.stderr,
        )
        raise SystemExit(2)

    row_type = "bundles" if config.seed_source.multi_doc else "text files"
    print(f"Discovered {available_records} {row_type} under {config.seed_source.path}")
    if requested_records < available_records:
        print(f"Selected the first {requested_records} seed records")
    _print_model_config(config.pipeline, custom_providers)
    effective_config = config.model_copy(update={"num_records": requested_records})

    if args.preview:
        print("\nPreviewing generation...")
        try:
            preview_generation(effective_config)
        except Exception as exc:  # noqa: BLE001 - preview is best-effort UX
            logger.warning("Preview error: %s", exc)
        return

    print(f"\nRecords to process: {requested_records}")
    print(f"Buffer size: {effective_config.buffer_size}")
    print(f"Resume mode: {ResumeMode(effective_config.resume).value}")
    print(f"Dataset name: {effective_config.dataset_name}")
    print("\nGenerating dataset...")
    result = run_generation(
        effective_config,
        config_sources=loaded.sources,
        override_paths=loaded.override_paths,
        environment_variables=loaded.environment_variables,
    )
    print(f"\nGeneration complete! Artifacts saved to {result.dataset_path}")
    print(f"Exported JSONL to {result.output_path}")
    if result.resolved_config_path is not None:
        print(f"Resolved run config: {result.resolved_config_path}")


def _add_convert_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``convert`` subcommand."""
    parser = subparsers.add_parser(
        "convert",
        help="Convert SDG output to retriever training/evaluation formats",
    )
    _add_config_arguments(parser)
    parser.add_argument(
        "input_path",
        nargs="?",
        default=_SUPPRESS,
        help="Generated JSONL/JSON/parquet file or an unambiguous directory",
    )
    parser.add_argument("--corpus-id", default=_SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=_SUPPRESS)
    parser.add_argument("--eval-only", action=argparse.BooleanOptionalAction, default=_SUPPRESS)
    parser.add_argument("--train-ratio", type=float, default=_SUPPRESS)
    parser.add_argument("--val-ratio", type=float, default=_SUPPRESS)
    parser.add_argument("--seed", type=int, default=_SUPPRESS)
    parser.add_argument("--quality-threshold", type=float, default=_SUPPRESS)
    parser.add_argument("--max-pos-docs", type=int, default=_SUPPRESS)
    parser.add_argument("--use-group-id-in-eval", action=argparse.BooleanOptionalAction, default=_SUPPRESS)
    parser.add_argument("--split-strategy", choices=["random", "dedupped", "cluster"], default=_SUPPRESS)
    parser.add_argument("--groups-json", nargs="+", default=_SUPPRESS)
    parser.set_defaults(func=_run_convert)


def _conversion_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate only explicitly supplied conversion CLI values."""
    overrides: dict[str, Any] = {}
    for argument in (
        "input_path",
        "corpus_id",
        "output_dir",
        "eval_only",
        "train_ratio",
        "val_ratio",
        "seed",
        "quality_threshold",
        "max_pos_docs",
        "use_group_id_in_eval",
        "split_strategy",
        "groups_json",
    ):
        _assign_if_present(args, overrides, argument)
    return overrides


def _run_convert(args: argparse.Namespace) -> None:
    """Execute conversion from one fully resolved typed config."""
    explicit_input_path = (
        args.config is not None
        or hasattr(args, "input_path")
        or any(expression.startswith("input_path=") for expression in args.set_overrides)
    )
    if not args.print_resolved_config and not explicit_input_path:
        print("Error: provide --config, input_path, or --set input_path=... for conversion", file=sys.stderr)
        raise SystemExit(2)

    try:
        loaded = load_conversion_config(
            args.config,
            cli_overrides=_conversion_cli_overrides(args),
            set_overrides=args.set_overrides,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.print_resolved_config:
        print(dump_resolved_config(loaded.config), end="")
        return
    run_conversion_with_config(
        loaded.config,
        config_sources=loaded.sources,
        override_paths=loaded.override_paths,
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
