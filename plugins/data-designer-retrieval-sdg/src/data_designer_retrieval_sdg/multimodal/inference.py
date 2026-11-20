# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed prompt payloads for Data Designer structured columns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import data_designer.config as dd
from pydantic import BaseModel

from data_designer_retrieval_sdg.multimodal.models import GenerationContext, JudgeStage, ModelRole, ModelSettings
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource


def model_configuration(
    model: ModelSettings, concurrency: int, *, skip_health_check: bool
) -> tuple[dd.ModelConfig, dd.ModelProvider]:
    """Build the same native deployment settings for readiness and generation.

    Args:
        model: Serving deployment and inference options.
        concurrency: Maximum parallel requests to this deployment.
        skip_health_check: Whether this configuration has already been checked.

    Returns:
        Native model and provider configurations sharing a provider name.
    """
    return (
        dd.ModelConfig(
            alias="retrieval",
            model=model.model,
            provider="retrieval-provider",
            skip_health_check=skip_health_check,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=model.temperature,
                max_tokens=model.max_tokens,
                timeout=model.timeout,
                max_parallel_requests=concurrency,
                extra_body=model.extra_body,
            ),
        ),
        dd.ModelProvider(
            name="retrieval-provider",
            endpoint=model.endpoint,
            provider_type="openai",
            api_key=model.credential_env,
        ),
    )


@dataclass(frozen=True)
class Request:
    """A typed request with ordered image attachments and explicit model routing.

    Attributes:
        text: Fully rendered prompt text.
        schema: Required structured response schema.
        role: Generator or default judge model role.
        images: Ordered paths to actual image attachments.
        judge_stage: Optional judgment stage selecting a configured override.
        context: Original source-unit membership for checkpoint provenance.
        image_unit_ids: Source-unit IDs aligned with the image attachments.
        query: Generated query retained with any failure diagnostic.
        instructions: Requested query slots retained with the seed record.
    """

    text: str
    schema: type[BaseModel]
    role: ModelRole
    images: tuple[Path, ...] = ()
    judge_stage: JudgeStage | None = None
    context: GenerationContext | None = None
    image_unit_ids: tuple[str, ...] = ()
    query: str | None = None
    instructions: tuple[dict, ...] = ()


def source_request(
    instruction: str,
    schema: type[BaseModel],
    role: ModelRole,
    context: GenerationContext,
    sources: dict[str, RetrievalSource],
    *,
    judge_stage: JudgeStage | None = None,
    **payload,
) -> Request:
    """Attach exact source text and aligned image pixels without truncation.

    Localization additionally identifies each source's document, canonical page,
    and one-based image attachment position. Unknown pages and absent images
    retain null metadata rather than inferred identifiers.

    Args:
        instruction: Rendered task instructions preceding the source payload.
        schema: Required structured response schema.
        role: Generator or default judge model role.
        context: Source-unit membership for the request.
        sources: Canonical sources indexed by unit ID.
        judge_stage: Optional judgment stage selecting a configured override.
        **payload: Additional task fields serialized with the original sources.

    Returns:
        A request with original source text, ordered images, and checkpoint provenance.
    """
    selected = [sources[key] for key in context.unit_ids]
    data = {
        "sources": [{"unit_id": s.unit_id, "text": s.text} for s in selected],
        "image_unit_ids": [s.unit_id for s in selected for _ in s.images],
        **payload,
    }
    if judge_stage == "localization":
        image_numbers = {unit_id: number for number, unit_id in enumerate(data["image_unit_ids"], start=1)}
        for entry, source in zip(data["sources"], selected):
            entry.update(
                document_id=source.document_id,
                page_number=source.page_number,
                image_number=image_numbers.get(source.unit_id),
            )
    return Request(
        instruction + "\n" + json.dumps(data, ensure_ascii=False),
        schema,
        role,
        tuple(Path(image) for s in selected for image in s.images),
        judge_stage=judge_stage,
        context=context,
        image_unit_ids=tuple(data["image_unit_ids"]),
        query=payload.get("query"),
        instructions=tuple(payload.get("instructions", [])),
    )
