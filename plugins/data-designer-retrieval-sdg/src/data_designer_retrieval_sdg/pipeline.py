# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline builders for retrieval SDG from text and optional images."""

from __future__ import annotations

import json
from pathlib import Path

import data_designer.config as dd
from data_designer.config.default_model_settings import get_builtin_model_providers, get_default_providers
from data_designer.config.seed_source import SeedSource

from data_designer_retrieval_sdg.config import EmbeddingDedupColumnConfig, RetrievalStructuredColumnConfig
from data_designer_retrieval_sdg.models import (
    DocumentArtifacts,
    QAPairEvaluations,
    QueryQualityEvaluations,
    RetrievalAnswers,
    RetrievalQueries,
    SourceAssessments,
)
from data_designer_retrieval_sdg.prompts import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    ANSWER_GENERATION_USER_PROMPT,
    ARTIFACT_EXTRACTION_SYSTEM_PROMPT,
    ARTIFACT_EXTRACTION_USER_PROMPT,
    QA_EVALUATION_SYSTEM_PROMPT,
    QA_EVALUATION_USER_PROMPT,
    QUERY_GENERATION_SYSTEM_PROMPT,
    QUERY_GENERATION_USER_PROMPT,
    QUERY_QUALITY_SYSTEM_PROMPT,
    QUERY_QUALITY_USER_PROMPT,
    SOURCE_ASSESSMENT_SYSTEM_PROMPT,
    SOURCE_ASSESSMENT_USER_PROMPT,
)
from data_designer_retrieval_sdg.run_config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MAX_ARTIFACTS_PER_TYPE,
    DEFAULT_MAX_HOPS,
    DEFAULT_MIN_COMPLEXITY,
    DEFAULT_MIN_HOPS,
    DEFAULT_NUM_PAIRS,
    DEFAULT_PROVIDER,
    DEFAULT_QUERY_COUNTS,
    DEFAULT_REASONING_COUNTS,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from data_designer_retrieval_sdg.stages import assemble_retrieval_pairs, select_retrieval_queries


def custom_model_config(
    artifact_extraction_model: str = DEFAULT_CHAT_MODEL,
    artifact_extraction_provider: str = DEFAULT_PROVIDER,
    qa_generation_model: str = DEFAULT_CHAT_MODEL,
    qa_generation_provider: str = DEFAULT_PROVIDER,
    quality_judge_model: str = DEFAULT_CHAT_MODEL,
    quality_judge_provider: str = DEFAULT_PROVIDER,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_provider: str = DEFAULT_PROVIDER,
    max_parallel_requests_for_gen: int | None = None,
) -> tuple[list[dd.ModelConfig], dict[str, str]]:
    """Configure the model suite for a generation job.

    Each pipeline role (artifact extraction, QA generation, quality judge,
    embedding) can point at a different model+provider.  When multiple roles
    share the same ``(model, provider)`` pair a single ``ModelConfig`` is
    created and the roles share its alias.

    Args:
        artifact_extraction_model: Model name for artifact extraction.
        artifact_extraction_provider: Provider for artifact extraction.
        qa_generation_model: Model name for QA generation.
        qa_generation_provider: Provider for QA generation.
        quality_judge_model: Model name for quality judge.
        quality_judge_provider: Provider for quality judge.
        embed_model: Model name for embeddings.
        embed_provider: Provider for embeddings.
        max_parallel_requests_for_gen: Optional cap on parallel requests
            for chat-completion models.

    Returns:
        Tuple of ``(model_configs, role_aliases)`` where ``role_aliases``
        maps each role name to the ``ModelConfig`` alias it should reference.
    """
    configs: list[dd.ModelConfig] = [
        dd.ModelConfig(
            alias="embed",
            model=embed_model,
            inference_parameters=dd.EmbeddingInferenceParams(
                max_parallel_requests=8,
                extra_body={"input_type": "query", "truncate": "NONE"},
            ),
            provider=embed_provider,
        ),
    ]
    role_aliases: dict[str, str] = {"embed": "embed"}

    chat_roles = [
        ("artifact_extraction", artifact_extraction_model, artifact_extraction_provider),
        ("qa_generation", qa_generation_model, qa_generation_provider),
        ("quality_judge", quality_judge_model, quality_judge_provider),
    ]

    seen: dict[tuple[str, str], str] = {}
    for role_name, model, provider in chat_roles:
        key = (model, provider)
        if key not in seen:
            seen[key] = role_name
            inference_kwargs: dict = {
                "temperature": 0.6,
                "top_p": 0.95,
                "timeout": 120,
            }
            if max_parallel_requests_for_gen is not None:
                inference_kwargs["max_parallel_requests"] = max_parallel_requests_for_gen
            configs.append(
                dd.ModelConfig(
                    alias=role_name,
                    model=model,
                    provider=provider,
                    inference_parameters=dd.ChatCompletionInferenceParams(**inference_kwargs),
                )
            )
        role_aliases[role_name] = seen[key]

    return configs, role_aliases


def build_model_providers(
    custom_provider_endpoint: str | None = None,
    custom_provider_name: str = "custom",
    custom_provider_type: str = "openai",
    custom_provider_api_key: str | None = None,
    custom_provider_fields: set[str] | None = None,
    model_providers_file: Path | None = None,
    model_providers: list[dd.ModelProvider] | None = None,
) -> tuple[list[dd.ModelProvider] | None, list[dd.ModelProvider]]:
    """Build a list of custom ``ModelProvider`` objects from CLI flags / config.

    Inline flags define a single provider; the config file can define
    multiple. Later sources override an earlier provider with the same alias:
    run config, provider file, then inline provider. Custom providers are
    merged with Data Designer defaults so that built-in providers remain
    available.

    Args:
        custom_provider_endpoint: Base URL for an inline custom provider.
        custom_provider_name: Name for the inline provider.
        custom_provider_type: API format (default ``"openai"``).
        custom_provider_api_key: API key or env-var name.
        custom_provider_fields: Inline fields explicitly supplied by the user.
        model_providers_file: Path to a YAML/JSON file with provider entries.
        model_providers: Providers already loaded from a run configuration.

    Returns:
        Tuple of ``(all_providers, custom_only_providers)``.  ``all_providers``
        is ``None`` when no custom providers exist.
    """
    import yaml

    custom_by_name: dict[str, dd.ModelProvider] = {}

    def add_provider(provider: dd.ModelProvider, source: str, *, replace: bool = False) -> None:
        existing = custom_by_name.get(provider.name)
        if existing is not None and existing.model_dump() != provider.model_dump() and not replace:
            raise ValueError(
                f"Conflicting model provider alias {provider.name!r} from {source}; "
                "use distinct aliases or make the endpoint, provider type, and credential identical."
            )
        custom_by_name[provider.name] = provider

    for provider in model_providers or []:
        add_provider(provider, "run configuration")

    if model_providers_file is not None:
        raw = model_providers_file.read_text(encoding="utf-8")
        if model_providers_file.suffix in (".yaml", ".yml"):
            entries = yaml.safe_load(raw)
        else:
            entries = json.loads(raw)

        if not isinstance(entries, list):
            raise ValueError(f"model-providers-file must contain a YAML/JSON list, got {type(entries).__name__}")
        file_providers: dict[str, dd.ModelProvider] = {}
        for entry in entries:
            provider = dd.ModelProvider(**entry)
            existing = file_providers.get(provider.name)
            if existing is not None and existing.model_dump() != provider.model_dump():
                raise ValueError(f"Conflicting model provider alias {provider.name!r} within {model_providers_file}")
            file_providers[provider.name] = provider
        for provider in file_providers.values():
            add_provider(provider, str(model_providers_file), replace=True)

    if custom_provider_endpoint is not None:
        inline_values = {
            "name": custom_provider_name,
            "endpoint": custom_provider_endpoint,
            "provider_type": custom_provider_type,
            "api_key": custom_provider_api_key,
        }
        existing = custom_by_name.get(custom_provider_name)
        if existing is not None and custom_provider_fields is not None:
            merged_values = existing.model_dump()
            for field in custom_provider_fields - {"name"}:
                merged_values[field] = inline_values[field]
            inline_provider = dd.ModelProvider(**merged_values)
        else:
            inline_provider = dd.ModelProvider(**inline_values)
        add_provider(
            inline_provider,
            "inline provider configuration",
            replace=True,
        )

    custom = list(custom_by_name.values())
    if not custom:
        return None, []

    custom_names = {p.name for p in custom}
    try:
        default_providers = get_default_providers()
    except FileNotFoundError:
        # Data Designer seeds the user-level provider file when its interface is
        # constructed; provider resolution runs earlier in this pipeline.
        default_providers = get_builtin_model_providers()
    defaults = [p for p in default_providers if p.name not in custom_names]
    return defaults + custom, custom


def build_qa_generation_pipeline(
    seed_source: SeedSource,
    start_index: int = 0,
    end_index: int = 199,
    max_artifacts_per_type: int = DEFAULT_MAX_ARTIFACTS_PER_TYPE,
    num_pairs: int = DEFAULT_NUM_PAIRS,
    query_counts: dict[str, int] | None = None,
    min_hops: int = DEFAULT_MIN_HOPS,
    max_hops: int = DEFAULT_MAX_HOPS,
    reasoning_counts: dict[str, int] | None = None,
    min_complexity: int = DEFAULT_MIN_COMPLEXITY,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_parallel_requests_for_gen: int | None = None,
    artifact_extraction_model: str = DEFAULT_CHAT_MODEL,
    artifact_extraction_provider: str = DEFAULT_PROVIDER,
    qa_generation_model: str = DEFAULT_CHAT_MODEL,
    qa_generation_provider: str = DEFAULT_PROVIDER,
    quality_judge_model: str = DEFAULT_CHAT_MODEL,
    quality_judge_provider: str = DEFAULT_PROVIDER,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_provider: str = DEFAULT_PROVIDER,
) -> dd.DataDesignerConfigBuilder:
    """Build the retrieval pipeline for the document-chunker source.

    This compatibility entry point preserves the established text workflow.
    It delegates to the same staged pipeline used when seed rows also
    contain images.

    Args:
        seed_source: Configured source of canonical retrieval seed records whose
            output schema includes ``file_name``, ``text``, ``chunks``,
            ``sections_structured``.
        start_index: Start index (inclusive) for ordered index-range selection.
        end_index: End index (inclusive) for ordered index-range selection.
        max_artifacts_per_type: Max artifacts extracted per type.
        num_pairs: QA pairs to generate per document.
        query_counts: Distribution of query types.
        min_hops: Minimum hops for multi-hop questions.
        max_hops: Maximum hops for multi-hop questions.
        reasoning_counts: Distribution of reasoning types.
        min_complexity: Minimum complexity score.
        similarity_threshold: Cosine similarity threshold for QA-pair dedup.
        max_parallel_requests_for_gen: Cap on parallel requests for chat models.
        artifact_extraction_model: Model for artifact extraction.
        artifact_extraction_provider: Provider for artifact extraction.
        qa_generation_model: Model for QA generation.
        qa_generation_provider: Provider for QA generation.
        quality_judge_model: Model for quality judge.
        quality_judge_provider: Provider for quality judge.
        embed_model: Model for embeddings.
        embed_provider: Provider for embeddings.

    Returns:
        Configured ``DataDesignerConfigBuilder`` ready for
        ``DataDesigner.create()`` or ``.preview()``.
    """
    model_configs, role_aliases = custom_model_config(
        artifact_extraction_model=artifact_extraction_model,
        artifact_extraction_provider=artifact_extraction_provider,
        qa_generation_model=qa_generation_model,
        qa_generation_provider=qa_generation_provider,
        quality_judge_model=quality_judge_model,
        quality_judge_provider=quality_judge_provider,
        embed_model=embed_model,
        embed_provider=embed_provider,
        max_parallel_requests_for_gen=max_parallel_requests_for_gen,
    )

    return build_retrieval_pipeline(
        seed_source,
        artifact_model_alias=role_aliases["artifact_extraction"],
        generator_model_alias=role_aliases["qa_generation"],
        judge_model_alias=role_aliases["quality_judge"],
        embedding_model_alias=role_aliases["embed"],
        model_configs=model_configs,
        start_index=start_index,
        end_index=end_index,
        max_artifacts_per_type=max_artifacts_per_type,
        num_pairs=num_pairs,
        query_counts=query_counts,
        min_hops=min_hops,
        max_hops=max_hops,
        reasoning_counts=reasoning_counts,
        min_complexity=min_complexity,
        similarity_threshold=similarity_threshold,
    )


def build_retrieval_pipeline(
    seed_source: SeedSource,
    *,
    artifact_model_alias: str,
    generator_model_alias: str,
    judge_model_alias: str,
    embedding_model_alias: str,
    model_configs: list[dd.ModelConfig] | str | Path | None = None,
    answer_model_alias: str | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    max_artifacts_per_type: int = DEFAULT_MAX_ARTIFACTS_PER_TYPE,
    num_pairs: int = DEFAULT_NUM_PAIRS,
    query_counts: dict[str, int] | None = None,
    min_hops: int = DEFAULT_MIN_HOPS,
    max_hops: int = DEFAULT_MAX_HOPS,
    reasoning_counts: dict[str, int] | None = None,
    min_complexity: int = DEFAULT_MIN_COMPLEXITY,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dd.DataDesignerConfigBuilder:
    """Build one retrieval-SDG pipeline for text and optional image inputs.

    Each seed row must expose retrieval_units and images. Text-only rows use an
    empty images list, which Data Designer treats as zero image contexts.
    Image-bearing rows require vision-capable artifact, generator, and
    grounding-judge models. The source-blind query judge never receives source
    text or images.

    Args:
        seed_source: Seed whose rows follow the canonical retrieval contract.
        artifact_model_alias: Model alias for structured artifact extraction.
        generator_model_alias: Model alias for retrieval-query generation.
        judge_model_alias: Model alias for both independent quality stages.
        embedding_model_alias: Model alias for semantic deduplication.
        model_configs: Data Designer model configuration list or path.
        answer_model_alias: Optional answer model; defaults to the query generator.
        start_index: Optional inclusive ordered-selection start.
        end_index: Optional inclusive ordered-selection end.
        max_artifacts_per_type: Maximum extracted artifacts of each type.
        num_pairs: Number of candidates requested per seed row.
        query_counts: Requested distribution of retrieval-query types.
        min_hops: Minimum reasoning hops.
        max_hops: Maximum reasoning hops.
        reasoning_counts: Requested distribution of reasoning types.
        min_complexity: Minimum requested complexity.
        similarity_threshold: Cosine threshold used for query deduplication.

    Returns:
        Configured retrieval pipeline with query selection before answer generation.

    Raises:
        ValueError: If aliases, selection bounds, or thresholds are invalid.
    """
    _validate_retrieval_pipeline_arguments(
        artifact_model_alias=artifact_model_alias,
        generator_model_alias=generator_model_alias,
        judge_model_alias=judge_model_alias,
        embedding_model_alias=embedding_model_alias,
        start_index=start_index,
        end_index=end_index,
        similarity_threshold=similarity_threshold,
    )
    if answer_model_alias is not None and not answer_model_alias.strip():
        raise ValueError("answer_model_alias must be non-empty when supplied")
    effective_query_counts = dict(DEFAULT_QUERY_COUNTS if query_counts is None else query_counts)
    effective_reasoning_counts = dict(DEFAULT_REASONING_COUNTS if reasoning_counts is None else reasoning_counts)
    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    if start_index is None:
        builder.with_seed_dataset(seed_source, sampling_strategy=dd.SamplingStrategy.ORDERED)
    else:
        builder.with_seed_dataset(
            seed_source,
            sampling_strategy=dd.SamplingStrategy.ORDERED,
            selection_strategy=dd.IndexRange(start=start_index, end=end_index),
        )

    image_context = [dd.ImageContext(column_name="images")]
    builder.add_column(
        RetrievalStructuredColumnConfig(
            name="document_artifacts",
            system_prompt=ARTIFACT_EXTRACTION_SYSTEM_PROMPT,
            prompt=ARTIFACT_EXTRACTION_USER_PROMPT.format(
                max_artifacts_per_type=max_artifacts_per_type,
            ),
            output_format=DocumentArtifacts,
            model_alias=artifact_model_alias,
            multi_modal_context=image_context,
        )
    )
    builder.add_column(
        RetrievalStructuredColumnConfig(
            name="query_generation",
            system_prompt=QUERY_GENERATION_SYSTEM_PROMPT,
            prompt=QUERY_GENERATION_USER_PROMPT.format(
                query_counts_multi_hop=effective_query_counts.get("multi_hop", 0),
                query_counts_structural=effective_query_counts.get("structural", 0),
                query_counts_contextual=effective_query_counts.get("contextual", 0),
                reasoning_counts_factual=effective_reasoning_counts.get("factual", 0),
                reasoning_counts_relational=effective_reasoning_counts.get("relational", 0),
                reasoning_counts_inferential=effective_reasoning_counts.get("inferential", 0),
                reasoning_counts_temporal=effective_reasoning_counts.get("temporal", 0),
                reasoning_counts_procedural=effective_reasoning_counts.get("procedural", 0),
                reasoning_counts_visual=effective_reasoning_counts.get("visual", 0),
                reasoning_counts_causal=effective_reasoning_counts.get("causal", 0),
                min_hops=min_hops,
                max_hops=max_hops,
                min_complexity=min_complexity,
                num_pairs=num_pairs,
                query_surface_schedule=", ".join(
                    f"candidate {index}: {('keyword', 'instruction', 'question')[index % 3]}"
                    for index in range(num_pairs)
                ),
            ),
            output_format=RetrievalQueries,
            model_alias=generator_model_alias,
            multi_modal_context=image_context,
        )
    )
    builder.add_column(
        EmbeddingDedupColumnConfig(
            name="deduplicated_queries",
            source_column="query_generation",
            items_key="queries",
            text_field="question",
            model_alias=embedding_model_alias,
            similarity_threshold=similarity_threshold,
        )
    )
    builder.add_column(
        RetrievalStructuredColumnConfig(
            name="source_blind_evaluations",
            system_prompt=QUERY_QUALITY_SYSTEM_PROMPT,
            prompt=QUERY_QUALITY_USER_PROMPT,
            output_format=QueryQualityEvaluations,
            model_alias=judge_model_alias,
            skip=dd.SkipConfig(when="{{ deduplicated_queries | length == 0 }}"),
        )
    )
    builder.add_column(
        dd.CustomColumnConfig(
            name="query_selection",
            generator_function=select_retrieval_queries,
            propagate_skip=False,
        )
    )
    builder.add_column(
        RetrievalStructuredColumnConfig(
            name="answer_generation",
            system_prompt=ANSWER_GENERATION_SYSTEM_PROMPT,
            prompt=ANSWER_GENERATION_USER_PROMPT,
            output_format=RetrievalAnswers,
            model_alias=answer_model_alias or generator_model_alias,
            multi_modal_context=image_context,
            skip=dd.SkipConfig(when="{{ query_selection.selected | length == 0 }}"),
        )
    )
    builder.add_column(
        dd.CustomColumnConfig(
            name="deduplicated_qa_pairs",
            generator_function=assemble_retrieval_pairs,
            propagate_skip=False,
        )
    )
    builder.add_column(
        RetrievalStructuredColumnConfig(
            name="source_assessments",
            system_prompt=SOURCE_ASSESSMENT_SYSTEM_PROMPT,
            prompt=SOURCE_ASSESSMENT_USER_PROMPT,
            output_format=SourceAssessments,
            model_alias=judge_model_alias,
            multi_modal_context=image_context,
            skip=dd.SkipConfig(when="{{ deduplicated_qa_pairs | length == 0 }}"),
        )
    )
    builder.add_column(
        RetrievalStructuredColumnConfig(
            name="qa_evaluations",
            system_prompt=QA_EVALUATION_SYSTEM_PROMPT,
            prompt=QA_EVALUATION_USER_PROMPT,
            output_format=QAPairEvaluations,
            model_alias=judge_model_alias,
            skip=dd.SkipConfig(when="{{ deduplicated_qa_pairs | length == 0 }}"),
        )
    )
    return builder


def _validate_retrieval_pipeline_arguments(
    *,
    artifact_model_alias: str,
    generator_model_alias: str,
    judge_model_alias: str,
    embedding_model_alias: str,
    start_index: int | None,
    end_index: int | None,
    similarity_threshold: float,
) -> None:
    """Validate the canonical pipeline construction arguments."""
    aliases = (
        artifact_model_alias,
        generator_model_alias,
        judge_model_alias,
        embedding_model_alias,
    )
    if any(not alias for alias in aliases):
        raise ValueError("artifact, generator, judge, and embedding model aliases must be non-empty")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1 and 1")
    if (start_index is None) != (end_index is None):
        raise ValueError("start_index and end_index must be supplied together")
    if start_index is not None and end_index is not None and start_index > end_index:
        raise ValueError("start_index must be less than or equal to end_index")
