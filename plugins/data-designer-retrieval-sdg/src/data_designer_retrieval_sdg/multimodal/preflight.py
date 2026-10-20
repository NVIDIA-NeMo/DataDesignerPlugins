# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Data Designer readiness checks before freezing a multimodal run."""

from tempfile import TemporaryDirectory

import data_designer.config as dd
from data_designer.engine.secret_resolver import EnvironmentResolver
from data_designer.interface import DataDesigner

from data_designer_retrieval_sdg.multimodal.inference import model_configuration
from data_designer_retrieval_sdg.multimodal.models import MultimodalSDGConfig
from data_designer_retrieval_sdg.multimodal.storage import fingerprint


def check_models(config: MultimodalSDGConfig) -> None:
    """Probe distinct active deployments with Data Designer's readiness API.

    Args:
        config: Generator, enabled judges, and optional hosted embedding settings.

    Raises:
        DataDesignerError: Native credential or model readiness failure. These
            exceptions propagate unchanged so provider diagnostics are preserved.
        TimeoutError: Data Designer's readiness timeout expires.
    """
    models, providers, columns, seen = [], [], [], set()
    for stage, settings in {"generator": config.generator, **config.active_judges()}.items():
        model, provider = model_configuration(settings, config.concurrency, skip_health_check=False)
        key = fingerprint([model.model_dump(mode="json"), provider.model_dump(mode="json")])
        if key in seen:
            continue
        seen.add(key)
        alias = "toc_confirmation" if stage == "judge" else stage
        model.alias = alias
        model.provider = provider.name = alias
        models.append(model)
        providers.append(provider)
        # check_models probes aliases referenced by columns, not every declared model.
        columns.append(dd.LLMTextColumnConfig(name=alias, model_alias=alias, prompt="Hello!"))
    if config.combination_iterations and config.summary_embedding_endpoint:
        alias = "summary_embedding"
        models.append(
            dd.ModelConfig(
                alias=alias,
                model=config.summary_embedding_model,
                provider=alias,
                inference_parameters=dd.EmbeddingInferenceParams(
                    timeout=600, extra_body=config.summary_embedding_extra_body
                ),
            )
        )
        providers.append(
            dd.ModelProvider(
                name=alias,
                endpoint=config.summary_embedding_endpoint,
                provider_type="openai",
                api_key=config.summary_embedding_credential_env,
            )
        )
        columns.append(dd.EmbeddingColumnConfig(name=alias, model_alias=alias, target_column="generator"))
    builder = dd.DataDesignerConfigBuilder(model_configs=models)
    for column in columns:
        builder.add_column(column)
    # Native readiness creates artifacts; keep them outside the immutable run.
    with TemporaryDirectory(prefix="retrieval-readiness-") as directory:
        designer = DataDesigner(
            artifact_path=directory,
            model_providers=providers,
            secret_resolver=EnvironmentResolver(),
            auto_configure_logging=False,
        )
        designer.set_run_config(dd.RunConfig(display_tui=False, otel_metrics_port=None))
        designer.check_models(builder)
