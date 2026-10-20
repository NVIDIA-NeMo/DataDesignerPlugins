# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential checks before freezing a run or making model requests."""

import os

from data_designer_retrieval_sdg.multimodal.models import MultimodalSDGConfig


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
        raise ValueError("Model credential preflight failed:\n" + "\n".join(missing))
