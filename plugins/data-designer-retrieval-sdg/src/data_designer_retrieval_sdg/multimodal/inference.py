# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed requests using registered Data Designer structured columns."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import data_designer.config as dd
import pyarrow as pa
import pyarrow.parquet as pq
from data_designer.engine.secret_resolver import EnvironmentResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage, ResumeMode
from data_designer.interface import DataDesigner
from pydantic import BaseModel

from data_designer_retrieval_sdg.config import RetrievalStructuredColumnConfig
from data_designer_retrieval_sdg.multimodal.bounds import check_request
from data_designer_retrieval_sdg.multimodal.models import GenerationContext, MultimodalSDGConfig
from data_designer_retrieval_sdg.multimodal.storage import digest, fingerprint, write_json
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource


@dataclass(frozen=True)
class Request:
    """A single typed request; images are real ordered attachments, never path-only prompts."""

    text: str
    schema: type[BaseModel]
    role: Literal["generator", "judge"]
    images: tuple[Path, ...] = ()


def source_request(
    instruction: str,
    schema: type,
    role: str,
    context: GenerationContext,
    sources: dict[str, RetrievalSource],
    **payload,
) -> Request:
    """Attach exact source text and aligned image pixels without truncation or templating source content."""
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
    )


class DataDesignerInference:
    """Run schema-homogeneous batches with immutable per-response caches.

    Args:
        root: Run-owned inference directory, protected by the workflow lock.
        config: Explicit model settings and request concurrency/batch bounds.
    """

    def __init__(self, root: Path, config: MultimodalSDGConfig):
        self.root = root
        self.config = config

    def request_key(self, request: Request) -> str:
        """Bind responses to exact prompts, schema, pixels and model settings."""
        return fingerprint(
            {
                "transport": "retrieval-structured-v1",
                "prompt": request.text,
                "schema": request.schema.model_json_schema(),
                "model": getattr(self.config, request.role).model_dump(mode="json"),
                "images": [digest(path) for path in request.images],
                "tokenizer_sha256": digest(getattr(self.config, request.role).tokenizer_file)
                if getattr(self.config, request.role).tokenizer_file
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
        return request.schema.model_validate(value["response"])

    def generate(self, requests: list[Request]) -> list[BaseModel]:
        """Retry only missing rows within the configured finite attempt budget.

        Missing outputs fail the stage after preserving successful responses and
        attempt evidence. Exceptions stop immediately; only normally completed
        batches with missing/invalid rows get another attempt. Native DD's own
        schema corrections remain bounded inside each attempt.
        """
        groups: dict[str, dict[str, Request]] = {}
        for request in requests:
            check_request(request, self.config, require_model_budget=True)
            if self.cached(request) is None:
                group = fingerprint([request.role, request.schema.model_json_schema(), bool(request.images)])
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
            check_request(item, self.config, require_model_budget=True)
        request = items[0][1]
        model = getattr(self.config, request.role)
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
        builder = dd.DataDesignerConfigBuilder(
            model_configs=[
                dd.ModelConfig(
                    alias="retrieval",
                    model=model.model,
                    provider="retrieval-provider",
                    skip_health_check=True,
                    inference_parameters=dd.ChatCompletionInferenceParams(
                        temperature=model.temperature,
                        max_tokens=model.max_tokens,
                        timeout=model.timeout,
                        max_parallel_requests=self.config.concurrency,
                        extra_body=model.extra_body,
                    ),
                )
            ]
        )
        builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed)))
        builder.add_column(
            RetrievalStructuredColumnConfig(
                name="response",
                model_alias="retrieval",
                prompt="{{ request_text }}",
                output_format=request.schema,
                multi_modal_context=[dd.ImageContext(column_name="request_images")] if request.images else None,
            )
        )
        designer = DataDesigner(
            artifact_path=attempt / "dd",
            secret_resolver=EnvironmentResolver(),
            auto_configure_logging=False,
            model_providers=[
                dd.ModelProvider(
                    name="retrieval-provider",
                    endpoint=model.endpoint,
                    provider_type="openai",
                    api_key=model.credential_env,
                )
            ],
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
        found: set[str] = set()
        invalid: list[str] = []
        for batch in sorted(dataset.glob("batch_*.parquet")):
            for row in pq.read_table(batch).to_pylist():
                key = row["request_id"]
                if key not in expected or key in found:
                    raise ValueError("Unexpected or duplicate Data Designer response identity")
                raw = row["response"]
                request = expected[key]
                try:
                    response = (
                        request.schema.model_validate_json(raw)
                        if isinstance(raw, str)
                        else request.schema.model_validate(raw)
                    )
                except ValueError:
                    invalid.append(key)
                    continue
                payload = response.model_dump(mode="json")
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
                "returned": len(found),
                "missing": sorted(expected.keys() - found),
                "invalid": invalid,
            },
        )
