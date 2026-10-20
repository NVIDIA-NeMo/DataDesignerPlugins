# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed requests using registered Data Designer structured columns."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import data_designer.config as dd
import pyarrow as pa
import pyarrow.parquet as pq
from data_designer.engine.secret_resolver import EnvironmentResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage, ResumeMode
from data_designer.interface import DataDesigner
from pydantic import BaseModel

from data_designer_retrieval_sdg.multimodal.bounds import check_request
from data_designer_retrieval_sdg.multimodal.config import RetrievalStructuredColumnConfig
from data_designer_retrieval_sdg.multimodal.models import (
    FailedGeneration,
    GenerationContext,
    JudgeStage,
    ModelRole,
    ModelSettings,
    MultimodalSDGConfig,
)
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource
from data_designer_retrieval_sdg.multimodal.storage import digest, fingerprint, write_json
from data_designer_retrieval_sdg.multimodal.structured import FAILURE_KEY


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

    Args:
        text: Fully rendered prompt text.
        schema: Required structured response schema.
        role: Generator or default judge model role.
        images: Ordered paths to actual image attachments.
        judge_stage: Optional judgment stage selecting a configured override.
    """

    text: str
    schema: type[BaseModel]
    role: ModelRole
    images: tuple[Path, ...] = ()
    judge_stage: JudgeStage | None = None


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

    Args:
        instruction: Rendered task instructions preceding the source payload.
        schema: Required structured response schema.
        role: Generator or default judge model role.
        context: Source-unit membership for the request.
        sources: Canonical sources indexed by unit ID.
        judge_stage: Optional judgment stage selecting a configured override.
        **payload: Additional task fields serialized with the original sources.

    Returns:
        A request with original source text and ordered image attachments.
    """
    selected = [sources[key] for key in context.unit_ids]
    data = {
        "sources": [{"unit_id": s.unit_id, "text": s.text} for s in selected],
        "image_unit_ids": [s.unit_id for s in selected for _ in s.images],
        **payload,
    }
    return Request(
        instruction + "\n" + json.dumps(data, ensure_ascii=False),
        schema,
        role,
        tuple(Path(image) for s in selected for image in s.images),
        judge_stage=judge_stage,
    )


def request_identity(context: GenerationContext, **details) -> dict:
    """Attach source membership and optional query/slot details to a failure record.

    Args:
        context: Original context owning the request.
        **details: Additional query, slot, or instruction provenance.

    Returns:
        JSON-compatible identity retained with any generation failure.
    """
    return {"context_id": context.context_id, "source_unit_ids": context.unit_ids, **details}


def generate_records(
    inference: DataDesignerInference, requests: list[Request], stage: str, records: list[dict]
) -> list[BaseModel]:
    """Generate aligned responses and retain contextual diagnostics for failed ones.

    Args:
        inference: Run-owned response cache and native Data Designer transport.
        requests: Ordered requests, including any identical requests shared by records.
        stage: Workflow stage responsible for these requests.
        records: One context/source/query identity per request.

    Returns:
        Validated responses or explicit failures in the original record order.
        Consumers decide the stage-specific disposition without shifting alignment.

    Raises:
        ValueError: Request, identity, or response counts disagree.
    """
    if len(requests) != len(records):
        raise ValueError("Every generation request requires a record identity")
    responses = inference.generate(requests)
    for request, record, response in zip(requests, records, responses, strict=True):
        if isinstance(response, FailedGeneration):
            key = inference.request_key(request)
            identity = fingerprint([stage, key, record])
            inference.failures[identity] = {
                **record,
                "stage": stage,
                "request_id": key,
                "failure": response.model_dump(mode="json"),
            }
    return responses


def parse_failure(value: object, policy: str) -> FailedGeneration | None:
    """Accept only a well-formed diagnostic under the explicit skip policy.

    Args:
        value: Decoded native response or cached payload.
        policy: Configured error/skip policy.

    Returns:
        The typed failure, or None for an ordinary response.

    Raises:
        ValueError: A failure marker is malformed or the skip policy is disabled.
    """
    if not isinstance(value, dict) or FAILURE_KEY not in value:
        return None
    if policy != "skip" or set(value) != {FAILURE_KEY}:
        raise ValueError("Saved failure requires the skip policy and a valid diagnostic")
    return FailedGeneration.model_validate(value[FAILURE_KEY])


class DataDesignerInference:
    """Run schema-homogeneous batches with immutable per-response caches.

    Args:
        root: Run-owned inference directory, protected by the workflow lock.
        config: Explicit model settings and request concurrency/batch bounds.
    """

    def __init__(self, root: Path, config: MultimodalSDGConfig):
        self.root = root
        self.config = config
        self.failures: dict[str, dict] = {}

    def request_key(self, request: Request) -> str:
        """Bind responses to exact prompts, schema, pixels and model settings."""
        model = self.config.model_for(request.role, request.judge_stage)
        return fingerprint(
            {
                "transport": "retrieval-structured-v1",
                "prompt": request.text,
                "schema": request.schema.model_json_schema(),
                "model": model.model_dump(mode="json"),
                "images": [digest(path) for path in request.images],
                "tokenizer_sha256": digest(model.tokenizer_file)
                if model.context_window_tokens is not None and model.tokenizer_file
                else None,
            }
        )

    def cached(self, request: Request) -> BaseModel | None:
        """Read a schema-validated cache entry; detect edits instead of regenerating it."""
        key = self.request_key(request)
        path = self.root / "responses" / f"{key}.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value["request_id"] != key or fingerprint(value["response"]) != value["response_sha256"]:
            raise ValueError(f"Response cache integrity failure: {key}")
        failure = parse_failure(value["response"], self.config.failed_record_policy)
        return failure if failure is not None else request.schema.model_validate(value["response"])

    def generate(self, requests: list[Request]) -> list[BaseModel]:
        """Retry only missing rows within the configured finite attempt budget.

        Missing outputs fail the stage after preserving successful responses and
        attempt evidence. Exceptions stop immediately; only normally completed
        batches with missing/invalid rows get another attempt. Native DD's own
        schema corrections remain bounded inside each attempt. In skip mode,
        explicit exhausted validation failures are cached terminal results;
        unexplained missing rows still fail the stage.
        """
        groups: dict[str, dict[str, Request]] = {}
        for request in requests:
            check_request(request, self.config)
            if self.cached(request) is None:
                group = fingerprint(
                    [
                        self.config.model_for(request.role, request.judge_stage).model_dump(mode="json"),
                        request.schema.model_json_schema(),
                        bool(request.images),
                    ]
                )
                groups.setdefault(group, {})[self.request_key(request)] = request
        for group in groups.values():
            items = list(group.items())
            for start in range(0, len(items), self.config.batch_size):
                pending = items[start : start + self.config.batch_size]
                for _ in range(self.config.missing_response_attempts):
                    if not pending:
                        break
                    self.run_batch(pending)
                    pending = [(key, request) for key, request in pending if self.cached(request) is None]
        results = [self.cached(request) for request in requests]
        if any(value is None for value in results):
            raise RuntimeError("Missing structured responses; inspect attempt evidence before explicitly resuming")
        return results

    def run_batch(self, items: list[tuple[str, Request]]) -> None:
        """Execute a homogeneous native DD batch and preserve every valid response."""
        for _, item in items:
            check_request(item, self.config)
        request = items[0][1]
        model = self.config.model_for(request.role, request.judge_stage)
        attempt = self.root / "attempts" / uuid4().hex
        attempt.mkdir(parents=True)
        rows = [
            {
                "request_id": key,
                "request_text": item.text,
                "request_images": [base64.b64encode(path.read_bytes()).decode() for path in item.images],
            }
            for key, item in items
        ]
        seed = attempt / "seed.parquet"
        pq.write_table(pa.Table.from_pylist(rows), seed)
        model_config, provider = model_configuration(model, self.config.concurrency, skip_health_check=True)
        builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
        builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed)))
        builder.add_column(
            RetrievalStructuredColumnConfig(
                name="response",
                model_alias="retrieval",
                prompt="{{ request_text }}",
                output_format=request.schema,
                failed_record_policy=self.config.failed_record_policy,
                multi_modal_context=[dd.ImageContext(column_name="request_images")] if request.images else None,
            )
        )
        designer = DataDesigner(
            artifact_path=attempt / "dd",
            secret_resolver=EnvironmentResolver(),
            auto_configure_logging=False,
            model_providers=[provider],
        )
        designer.set_run_config(
            dd.RunConfig(
                disable_early_shutdown=True,
                max_conversation_correction_steps=0,
                buffer_size=self.config.batch_size,
                max_concurrent_row_groups=1,
                display_tui=False,
                otel_metrics_port=None,
            )
        )
        write_json(attempt / "requested.json", {"request_ids": [key for key, _ in items]})
        try:
            result = designer.create(builder, num_records=len(items), dataset_name="requests")
        except Exception as exc:
            # Do not serialize exception messages: provider errors can contain credentials.
            write_json(attempt / "failure.json", {"exception_type": type(exc).__name__})
            storage = ArtifactStorage(
                artifact_path=attempt / "dd", dataset_name="requests", resume=ResumeMode.IF_POSSIBLE
            )
            self.collect_responses(items, storage.final_dataset_path, attempt)
            raise
        self.collect_responses(items, result.artifact_storage.final_dataset_path, attempt)

    def collect_responses(self, items: list[tuple[str, Request]], dataset: Path, attempt: Path) -> None:
        """Cache completed rows even when a later provider failure stops the batch.

        Args:
            items: Exact requests owned by this attempt.
            dataset: Native DD final-row shards, never arbitrary partial artifacts.
            attempt: Directory for immutable completion evidence.
        """
        expected = {key: item for key, item in items}
        batches = sorted(dataset.glob("batch_*.parquet"))
        seen: set[str] = set()
        # Reject ambiguous identities before publishing any response from this attempt.
        for batch in batches:
            for key in pq.read_table(batch, columns=["request_id"]).column("request_id").to_pylist():
                if key not in expected or key in seen:
                    raise ValueError("Unexpected or duplicate Data Designer response identity")
                seen.add(key)
        found: set[str] = set()
        failed: list[str] = []
        invalid: list[str] = []
        for batch in batches:
            for row in pq.read_table(batch).to_pylist():
                key = row["request_id"]
                if key not in expected or key in found:
                    raise ValueError("Unexpected or duplicate Data Designer response identity")
                raw = row["response"]
                request = expected[key]
                try:
                    value = json.loads(raw) if isinstance(raw, str) else raw
                except ValueError:
                    invalid.append(key)
                    continue
                failure = parse_failure(value, self.config.failed_record_policy)
                if failure is not None:
                    payload = {FAILURE_KEY: failure.model_dump(mode="json")}
                    failed.append(key)
                else:
                    try:
                        payload = request.schema.model_validate(value).model_dump(mode="json")
                    except ValueError:
                        invalid.append(key)
                        continue
                write_json(
                    self.root / "responses" / f"{key}.json",
                    {
                        "request_id": key,
                        "response": payload,
                        "response_sha256": fingerprint(payload),
                    },
                )
                found.add(key)
        write_json(
            attempt / "outcome.json",
            {
                "requested": len(items),
                "returned": len(found) - len(failed),
                "missing": sorted(expected.keys() - found),
                "invalid": invalid,
                **({"failed": failed} if self.config.failed_record_policy == "skip" else {}),
            },
        )
