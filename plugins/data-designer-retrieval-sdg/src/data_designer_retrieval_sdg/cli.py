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
from pydantic_settings import CliSettingsSource

from data_designer_retrieval_sdg.convert import run_conversion_with_config
from data_designer_retrieval_sdg.generation import (
    _count_seed_records,
    _resolve_dataset_name,
    preview_generation,
    run_generation,
)
from data_designer_retrieval_sdg.pipeline import build_model_providers
from data_designer_retrieval_sdg.run_config import (
    ConversionRunConfig,
    GenerationRunConfig,
    LoadedRunConfig,
    config_source_from_path,
    dump_resolved_config,
    load_conversion_config,
    load_generation_config,
    resolve_generation_provider_environment,
)

logger = logging.getLogger(__name__)
_SUPPRESS = argparse.SUPPRESS

_GENERATION_CLI_SHORTCUTS = {
    "seed-source.path": "input-dir",
    "seed-source.file-pattern": "file-pattern",
    "seed-source.recursive": "recursive",
    "seed-source.min-text-length": "min-text-length",
    "seed-source.sentences-per-chunk": "sentences-per-chunk",
    "seed-source.num-sections": "num-sections",
    "seed-source.num-files": "num-files",
    "seed-source.multi-doc": "multi-doc",
    "seed-source.bundle-size": "bundle-size",
    "seed-source.bundle-strategy": "bundle-strategy",
    "seed-source.max-docs-per-bundle": "max-docs-per-bundle",
    "seed-source.multi-doc-manifest": "multi-doc-manifest",
    "pipeline.max-artifacts-per-type": "max-artifacts-per-type",
    "pipeline.num-pairs": "num-pairs",
    "pipeline.min-hops": "min-hops",
    "pipeline.max-hops": "max-hops",
    "pipeline.min-complexity": "min-complexity",
    "pipeline.similarity-threshold": "similarity-threshold",
    "pipeline.max-parallel-requests-for-gen": "max-parallel-requests-for-gen",
    "pipeline.artifact-extraction-model": "artifact-extraction-model",
    "pipeline.artifact-extraction-provider": "artifact-extraction-provider",
    "pipeline.qa-generation-model": "qa-generation-model",
    "pipeline.qa-generation-provider": "qa-generation-provider",
    "pipeline.quality-judge-model": "quality-judge-model",
    "pipeline.quality-judge-provider": "quality-judge-provider",
    "pipeline.embed-model": "embed-model",
    "pipeline.embed-provider": "embed-provider",
    "custom-provider.endpoint": "custom-provider-endpoint",
    "custom-provider.name": "custom-provider-name",
    "custom-provider.provider-type": "custom-provider-type",
    "custom-provider.api-key": "custom-provider-api-key",
    "resume": "r",
}


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
    """Add run-config controls shared by both commands."""
    parser.add_argument("--config", type=Path, help="User YAML/JSON config layered over the packaged default")
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
    parser.add_argument("--preview", action="store_true", help="Preview without full generation")

    cli_settings_source = CliSettingsSource(
        GenerationRunConfig,
        root_parser=parser,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_shortcuts=_GENERATION_CLI_SHORTCUTS,
    )

    # These established forms accept multiple values after one flag. Pydantic's
    # canonical nested flags remain available for JSON/list input.
    parser.add_argument(
        "--file-extensions",
        dest="legacy_file_extensions",
        nargs="+",
        default=_SUPPRESS,
        help="Allowed extensions; use an empty string to match extensionless files",
    )
    parser.add_argument(
        "--query-counts",
        dest="legacy_query_counts",
        nargs="+",
        type=_parse_count_entry,
        default=_SUPPRESS,
        metavar="NAME=COUNT",
        help="Exact query-type counts; values must sum to --num-pairs",
    )
    parser.add_argument(
        "--reasoning-counts",
        dest="legacy_reasoning_counts",
        nargs="+",
        type=_parse_count_entry,
        default=_SUPPRESS,
        metavar="NAME=COUNT",
        help="Exact reasoning-type counts; values must sum to --num-pairs",
    )
    parser.set_defaults(func=_run_generate, cli_settings_source=cli_settings_source)


def _generation_legacy_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate only legacy multi-value generation arguments."""
    overrides: dict[str, Any] = {}
    seed_source: dict[str, Any] = {}
    pipeline: dict[str, Any] = {}
    if hasattr(args, "legacy_file_extensions"):
        seed_source["file_extensions"] = args.legacy_file_extensions
    if hasattr(args, "legacy_query_counts"):
        pipeline["query_counts"] = dict(args.legacy_query_counts)
    if hasattr(args, "legacy_reasoning_counts"):
        pipeline["reasoning_counts"] = dict(args.legacy_reasoning_counts)
    if seed_source:
        overrides["seed_source"] = seed_source
    if pipeline:
        overrides["pipeline"] = pipeline
    return overrides


def _load_generation_from_args(args: argparse.Namespace) -> tuple[LoadedRunConfig, list[dd.ModelProvider]]:
    """Resolve generation files, typed CLI values, and provider shorthand."""
    loaded = load_generation_config(
        args.config,
        cli_overrides=_generation_legacy_overrides(args),
        cli_args=args,
        cli_settings_source=args.cli_settings_source,
    )
    config = loaded.config
    custom_provider = config.custom_provider
    model_providers, custom_providers = build_model_providers(
        model_providers=config.model_providers,
        custom_provider_endpoint=custom_provider.endpoint if custom_provider is not None else None,
        custom_provider_name=custom_provider.name if custom_provider is not None else "custom",
        custom_provider_type=custom_provider.provider_type if custom_provider is not None else "openai",
        custom_provider_api_key=custom_provider.api_key if custom_provider is not None else None,
        custom_provider_fields=set(custom_provider.model_fields_set) if custom_provider is not None else None,
        model_providers_file=config.model_providers_file,
    )

    sources = list(loaded.sources)
    if config.model_providers_file is not None:
        sources.append(config_source_from_path(config.model_providers_file))

    config, provider_environment_variables = resolve_generation_provider_environment(
        config.model_copy(update={"model_providers": model_providers})
    )
    environment_variables = tuple(dict.fromkeys((*loaded.environment_variables, *provider_environment_variables)))
    return (
        LoadedRunConfig(
            config=config,
            sources=tuple(sources),
            override_paths=loaded.override_paths,
            environment_variables=environment_variables,
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
    try:
        loaded, custom_providers = _load_generation_from_args(args)
        if not args.print_resolved_config and args.config is None and "seed_source.path" not in loaded.override_paths:
            raise ValueError("provide --config, --input-dir, or --seed-source.path for generation")
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
        conflict_handler="resolve",
    )
    _add_config_arguments(parser)
    cli_settings_source = CliSettingsSource(
        ConversionRunConfig,
        root_parser=parser,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )
    parser.add_argument(
        "legacy_input_path",
        nargs="?",
        default=_SUPPRESS,
        metavar="input_path",
        help="Generated JSONL/JSON/parquet file or an unambiguous directory",
    )
    parser.add_argument(
        "--groups-json",
        dest="legacy_groups_json",
        nargs="+",
        default=_SUPPRESS,
        help="Dedup group JSON paths",
    )
    parser.set_defaults(func=_run_convert, cli_settings_source=cli_settings_source)


def _conversion_legacy_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate positional input and multi-value group paths."""
    overrides: dict[str, Any] = {}
    if hasattr(args, "legacy_input_path"):
        overrides["input_path"] = args.legacy_input_path
    if hasattr(args, "legacy_groups_json"):
        overrides["groups_json"] = args.legacy_groups_json
    return overrides


def _run_convert(args: argparse.Namespace) -> None:
    """Execute conversion from one fully resolved typed config."""
    try:
        loaded = load_conversion_config(
            args.config,
            cli_overrides=_conversion_legacy_overrides(args),
            cli_args=args,
            cli_settings_source=args.cli_settings_source,
        )
        if not args.print_resolved_config and args.config is None and "input_path" not in loaded.override_paths:
            raise ValueError("provide --config, input_path, or --input-path for conversion")
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
