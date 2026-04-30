# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline builder for the retriever SDG workflow.

Assembles a four-column DataDesigner pipeline:

1. ``document_artifacts`` -- LLM-based artifact extraction
2. ``qa_generation`` -- LLM-based QA pair generation
3. ``deduplicated_qa_pairs`` -- embedding-based deduplication (plugin column)
4. ``qa_evaluations`` -- LLM-based quality evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

import data_designer.config as dd
from data_designer.config.default_model_settings import get_default_providers

from data_designer_retrieval_sdg.config import EmbeddingDedupColumnConfig
from data_designer_retrieval_sdg.models import (
    DocumentArtifacts,
    QAPairEvaluations,
    QuestionAnswerPairs,
)
from data_designer_retrieval_sdg.prompts import (
    ARTIFACT_EXTRACTION_SYSTEM_PROMPT,
    ARTIFACT_EXTRACTION_USER_PROMPT,
    QA_EVALUATION_SYSTEM_PROMPT,
    QA_EVALUATION_USER_PROMPT,
    QA_GENERATION_SYSTEM_PROMPT,
    QA_GENERATION_USER_PROMPT,
)
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

DEFAULT_CHAT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
DEFAULT_EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
DEFAULT_PROVIDER = "nvidia"


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
    model_providers_file: Path | None = None,
) -> tuple[list[dd.ModelProvider] | None, list[dd.ModelProvider]]:
    """Build a list of custom ``ModelProvider`` objects from CLI flags / config.

    Inline flags define a single provider; the config file can define
    multiple.  When both are supplied the inline provider overwrites any
    file entry with the same name.  Custom providers are merged with Data
    Designer defaults so that built-in providers remain available.

    Args:
        custom_provider_endpoint: Base URL for an inline custom provider.
        custom_provider_name: Name for the inline provider.
        custom_provider_type: API format (default ``"openai"``).
        custom_provider_api_key: API key or env-var name.
        model_providers_file: Path to a YAML/JSON file with provider entries.

    Returns:
        Tuple of ``(all_providers, custom_only_providers)``.  ``all_providers``
        is ``None`` when no custom providers exist.
    """
    import yaml

    custom: list[dd.ModelProvider] = []

    if model_providers_file is not None:
        raw = model_providers_file.read_text(encoding="utf-8")
        if model_providers_file.suffix in (".yaml", ".yml"):
            entries = yaml.safe_load(raw)
        else:
            entries = json.loads(raw)

        if not isinstance(entries, list):
            raise ValueError(f"model-providers-file must contain a YAML/JSON list, got {type(entries).__name__}")
        for entry in entries:
            custom.append(dd.ModelProvider(**entry))

    if custom_provider_endpoint is not None:
        custom = [p for p in custom if p.name != custom_provider_name]
        custom.append(
            dd.ModelProvider(
                name=custom_provider_name,
                endpoint=custom_provider_endpoint,
                provider_type=custom_provider_type,
                api_key=custom_provider_api_key,
            )
        )

    if not custom:
        return None, []

    custom_names = {p.name for p in custom}
    defaults = [p for p in get_default_providers() if p.name not in custom_names]
    return defaults + custom, custom


DEFAULT_QUERY_COUNTS: dict[str, int] = {"multi_hop": 3, "structural": 2, "contextual": 2}
DEFAULT_REASONING_COUNTS: dict[str, int] = {
    "factual": 1,
    "relational": 1,
    "inferential": 1,
    "temporal": 1,
    "procedural": 1,
    "causal": 1,
    "visual": 1,
}


def build_qa_generation_pipeline(
    seed_source: DocumentChunkerSeedSource,
    start_index: int = 0,
    end_index: int = 199,
    max_artifacts_per_type: int = 2,
    num_pairs: int = 5,
    query_counts: dict[str, int] | None = None,
    min_hops: int = 2,
    max_hops: int = 3,
    reasoning_counts: dict[str, int] | None = None,
    min_complexity: int = 4,
    similarity_threshold: float = 0.9,
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
    """Build a four-column QA generation pipeline.

    The pipeline adds columns in order:

    1. ``document_artifacts`` -- structured artifact extraction
    2. ``qa_generation`` -- QA pair generation from artifacts + sections
    3. ``deduplicated_qa_pairs`` -- embedding dedup (plugin)
    4. ``qa_evaluations`` -- quality scoring

    Args:
        seed_source: Configured :class:`DocumentChunkerSeedSource` whose
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
    if query_counts is None:
        query_counts = dict(DEFAULT_QUERY_COUNTS)
    if reasoning_counts is None:
        reasoning_counts = dict(DEFAULT_REASONING_COUNTS)

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

    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    config_builder.with_seed_dataset(
        seed_source,
        sampling_strategy=dd.SamplingStrategy.ORDERED,
        selection_strategy=dd.IndexRange(start=start_index, end=end_index),
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="document_artifacts",
            system_prompt=ARTIFACT_EXTRACTION_SYSTEM_PROMPT,
            prompt=ARTIFACT_EXTRACTION_USER_PROMPT.format(
                max_artifacts_per_type=max_artifacts_per_type,
            ),
            output_format=DocumentArtifacts,
            model_alias=role_aliases["artifact_extraction"],
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="qa_generation",
            system_prompt=QA_GENERATION_SYSTEM_PROMPT,
            prompt=QA_GENERATION_USER_PROMPT.format(
                query_counts_multi_hop=query_counts.get("multi_hop", 0),
                query_counts_structural=query_counts.get("structural", 0),
                query_counts_contextual=query_counts.get("contextual", 0),
                reasoning_counts_factual=reasoning_counts.get("factual", 0),
                reasoning_counts_relational=reasoning_counts.get("relational", 0),
                reasoning_counts_inferential=reasoning_counts.get("inferential", 0),
                reasoning_counts_temporal=reasoning_counts.get("temporal", 0),
                reasoning_counts_procedural=reasoning_counts.get("procedural", 0),
                reasoning_counts_visual=reasoning_counts.get("visual", 0),
                reasoning_counts_causal=reasoning_counts.get("causal", 0),
                min_hops=min_hops,
                max_hops=max_hops,
                min_complexity=min_complexity,
                num_pairs=num_pairs,
            ),
            output_format=QuestionAnswerPairs,
            model_alias=role_aliases["qa_generation"],
        )
    )

    config_builder.add_column(
        EmbeddingDedupColumnConfig(
            name="deduplicated_qa_pairs",
            source_column="qa_generation",
            items_key="pairs",
            text_field="question",
            model_alias="embed",
            similarity_threshold=similarity_threshold,
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="qa_evaluations",
            system_prompt=QA_EVALUATION_SYSTEM_PROMPT,
            prompt=QA_EVALUATION_USER_PROMPT,
            output_format=QAPairEvaluations,
            model_alias=role_aliases["quality_judge"],
        )
    )

    return config_builder
