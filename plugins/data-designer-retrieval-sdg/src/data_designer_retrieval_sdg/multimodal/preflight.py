# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local deployment checks before freezing a run or making model requests."""

import os

from data_designer_retrieval_sdg.multimodal import prompts
from data_designer_retrieval_sdg.multimodal.bounds import RequestTooLarge, check_request
from data_designer_retrieval_sdg.multimodal.inference import Request
from data_designer_retrieval_sdg.multimodal.models import (
    Localization,
    MultimodalSDGConfig,
    QueryMetadata,
    RelevanceJudgment,
    SelfSufficiencyJudgment,
    SummaryJudgment,
    TOCConfirmation,
)


def validate_credentials(config: MultimodalSDGConfig) -> None:
    """Check nonempty credential variables for every potentially active deployment.

    Args:
        config: Generator, enabled judges, and optional hosted embedding settings.

    Raises:
        ValueError: Required variables are missing or blank. Diagnostics contain
            stage, model, and variable names, never credential values. This local
            check does not authenticate to the serving deployment.
    """
    deployments = [
        ("toc_confirmation" if stage == "judge" else stage, model.model, model.credential_env)
        for stage, model in {"generator": config.generator, **config.active_judges()}.items()
    ]
    if config.combination_iterations and config.summary_embedding_endpoint:
        deployments.append(
            ("summary_embedding", config.summary_embedding_model, config.summary_embedding_credential_env)
        )
    missing = [
        f"{stage} model {model!r}: set the credential environment variable {variable}"
        for stage, model, variable in deployments
        if not os.environ.get(variable, "").strip()
    ]
    if missing:
        raise ValueError("Model preflight failed:\n" + "\n".join(missing))


def validate_judge_prompts(config: MultimodalSDGConfig) -> None:
    """Check fixed judge instructions and schemas with dynamic content omitted.

    Args:
        config: Active judge settings and the summary-judgment persona.

    Raises:
        RequestTooLarge: The fixed prompt exceeds DD's rendering bound or the
            configured model budget, including schema, overhead and output space.

    Empty query/source/summary fields and no images keep this check independent
    of generated content. Passing does not establish that a populated request
    will fit; the existing per-request checks still validate actual content.
    """
    templates = {
        "summary": (
            prompts.render("judgment", summary="", persona=config.persona, language=""),
            SummaryJudgment,
        ),
        "query_metadata": (prompts.render("query_metadata", query=""), QueryMetadata),
        "self_sufficiency": (prompts.render("self_sufficiency", summary="", query=""), SelfSufficiencyJudgment),
        "relevance": (prompts.render("relevancy", summary="", query=""), RelevanceJudgment),
        "localization": (prompts.LOCALIZE, Localization),
        "judge": (prompts.render("check_toc", page="", language=""), TOCConfirmation),
    }
    for stage, model in config.active_judges().items():
        text, schema = templates[stage]
        request = Request(text, schema, "judge", judge_stage=None if stage == "judge" else stage)
        try:
            check_request(request, config)
        except RequestTooLarge as exc:
            name = "toc_confirmation" if stage == "judge" else stage
            raise RequestTooLarge(
                f"{name} model {model.model!r}: fixed judge prompt does not fit. {exc}. "
                "Check context_window_tokens, max_tokens, request_overhead_tokens and the stage's prompt settings."
            ) from exc
