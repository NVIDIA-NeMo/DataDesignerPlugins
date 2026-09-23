# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual rendered-request bounds, independent of canonical source text length."""

from functools import cache

from data_designer.engine.processing.ginja import environment

from data_designer_retrieval_sdg.multimodal.models import GenerationContext
from data_designer_retrieval_sdg.multimodal.storage import digest, fingerprint
from data_designer_retrieval_sdg.structured import RetrievalStructuredResponseRecipe


class RequestTooLarge(ValueError):
    """A request must be split before generation; never truncate its evidence."""


@cache
def tokenizer(path: str, sha256: str):
    """Load the deployment's local tokenizer, with its content hash in cache identity."""
    from tokenizers import Tokenizer

    encoder = Tokenizer.from_file(path)
    encoder.no_truncation()
    encoder.no_padding()
    return encoder


def validate_model_budget(model) -> None:
    """Fail closed for live requests when deployment-specific limits are unknown."""
    if model.context_window_tokens is None or model.tokenizer_file is None or model.request_overhead_tokens is None:
        raise ValueError(
            "Configure context_window_tokens, tokenizer_file and request_overhead_tokens for each model; "
            "DD's rendering limit is not a model context window"
        )
    if model.max_tokens + model.request_overhead_tokens >= model.context_window_tokens:
        raise ValueError("Model output and request overhead leave no input context space")


def validate_deployment(config, has_images: bool) -> None:
    """Validate model budgets/assets before creating an immutable run directory."""
    for role in ("generator", "judge"):
        model = getattr(config, role)
        validate_model_budget(model)
        if has_images and model.image_tokens_per_image is None:
            raise ValueError(f"Configure image_tokens_per_image for {role} before running image SDG")
        tokenizer(str(model.tokenizer_file), digest(model.tokenizer_file))


def check_request(request, config, *, require_model_budget=False) -> None:
    """Check DD's actual rendered text and the deployment's independent token budget.

    The native structured recipe appends schema instructions after rendering.
    Count them for model input, plus configured chat/system overhead, image-token
    upper bounds and max_tokens output space. Local tokenizer assets and all
    deployment bounds must match the serving configuration. No universal image
    token formula or character-to-token ratio is assumed.
    """
    if len(request.text) > environment.MAX_RENDERED_LEN:
        raise RequestTooLarge(f"Rendered prompt exceeds DD limit ({environment.MAX_RENDERED_LEN} characters)")
    model = getattr(config, request.role)
    if require_model_budget or model.context_window_tokens is not None:
        validate_model_budget(model)
        if request.images and model.image_tokens_per_image is None:
            raise ValueError("Configure image_tokens_per_image as a serving-model upper bound for image requests")
        recipe = RetrievalStructuredResponseRecipe(json_schema=request.schema.model_json_schema(), pruning=False)
        text = recipe.apply_recipe_to_user_prompt(request.text)
        encoder = tokenizer(str(model.tokenizer_file), digest(model.tokenizer_file))
        tokens = len(encoder.encode(text, add_special_tokens=False).ids)
        required = (
            tokens
            + model.request_overhead_tokens
            + model.max_tokens
            + len(request.images) * (model.image_tokens_per_image or 0)
        )
        if required > model.context_window_tokens:
            raise RequestTooLarge(
                f"{request.role} request needs {required} tokens including images, overhead and output; "
                f"configured context window is {model.context_window_tokens}"
            )


def split_context(context):
    """Split at whole-unit boundaries; a single oversized unit fails explicitly."""
    if len(context.unit_ids) == 1:
        raise RequestTooLarge(
            f"Source unit {context.unit_ids[0]!r} cannot fit its rendered request; "
            "prepare smaller canonical units or correct the deployment budget"
        )
    midpoint = len(context.unit_ids) // 2
    return [
        GenerationContext(
            context_id="ctx_" + fingerprint([context.context_id, ids])[:32],
            unit_ids=ids,
            language=context.language,
        )
        for ids in (context.unit_ids[:midpoint], context.unit_ids[midpoint:])
    ]


def fit_contexts(contexts, request_builder, config):
    """Keep fitting memberships intact; split oversized requests without losing units.

    Args:
        contexts: Ordered whole-unit memberships.
        request_builder: Callable returning every concrete request to check for a context.
        config: DD/model request limits.

    Returns:
        Contexts whose concrete requests fit. Only RequestTooLarge triggers splitting;
        missing model metadata and other errors propagate immediately.
    """
    result, pending = [], list(reversed(contexts))
    while pending:
        context = pending.pop()
        try:
            for request in request_builder(context):
                check_request(request, config)
        except RequestTooLarge:
            pending.extend(reversed(split_context(context)))
        else:
            result.append(context)
    return result
