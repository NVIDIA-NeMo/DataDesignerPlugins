# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from threading import BoundedSemaphore
from typing import Any
from uuid import uuid4

import httpx
from data_designer.config.scheduling import SchedulingMetadata
from data_designer.engine.column_generators.generators.base import ColumnGenerator, ColumnGeneratorCellByCell

from data_designer_gym.config import GymColumnConfig, GymHarnessColumnConfig, GymRequestConfig, GymResourceColumnConfig
from data_designer_gym.conversion import _json_compatible, _json_value, _source_value

logger = logging.getLogger(__name__)


class GymCallError(ValueError):
    """A request, transport, protocol, or incomplete harness failure."""

    def __init__(self, kind: str, reason: str, *, response: Any = None, http_status: int | None = None):
        super().__init__(reason)
        self.kind = kind
        self.response = response
        self.http_status = http_status


def gym_request_from_row(data: dict, config: GymRequestConfig) -> dict:
    """Assemble and validate a JSON object at the request boundary."""
    if config.request_column is not None:
        request = _json_value(_source_value(data, config.request_column), config.request_column)
    else:
        request = _json_compatible(config.request_values, "request_values")
        request.update(
            {key: _json_compatible(_source_value(data, path), path) for key, path in config.request_columns.items()}
        )
    if not isinstance(request, dict):
        raise ValueError("Gym request must be a JSON object")
    # HTTP JSON cannot carry NaN or Infinity, including values loaded from JSON strings.
    json.dumps(request, allow_nan=False)
    return request


def final_assistant_answer(response: dict) -> str:
    """Extract a terminal assistant message, rejecting unfinished tool loops."""
    if (
        response.get("status") not in (None, "completed")
        or response.get("incomplete_details")
        or response.get("error")
        or response.get("required_action")
    ):
        raise GymCallError("incomplete", "harness response is not completed")
    output = response.get("output")
    if not isinstance(output, list) or not output or not isinstance(output[-1], dict):
        raise GymCallError("incomplete", "harness response has no terminal assistant message")
    if not all(isinstance(item, dict) for item in output):
        raise GymCallError("response", "harness output must contain JSON objects")
    completed_calls = {
        item.get("call_id")
        for item in output
        if item.get("type") in ("function_call_output", "custom_tool_call_output")
    }
    if any(
        item.get("type") in ("function_call", "custom_tool_call")
        and (not item.get("call_id") or item["call_id"] not in completed_calls)
        for item in output
    ):
        raise GymCallError("incomplete", "harness response contains a pending tool call")
    message = output[-1]
    if (
        message.get("type") != "message"
        or message.get("role") != "assistant"
        or message.get("status") != "completed"
        or message.get("phase") == "commentary"
    ):
        raise GymCallError("incomplete", "harness response ends in an intermediate item")
    content = message.get("content")
    if not isinstance(content, list) or not all(isinstance(part, dict) for part in content):
        raise GymCallError("response", "assistant message content must be a list of objects")
    text = "".join(
        part["text"] for part in content if part.get("type") == "output_text" and isinstance(part.get("text"), str)
    )
    if not text.strip():
        raise GymCallError("incomplete", "harness has no final assistant text")
    return text


def project_response(response: Any, fields: dict[str, str] | None) -> Any:
    """Select named JSON object paths, retaining null for missing optional fields."""
    if response is None or fields is None:
        return response
    projected = {}
    for name, path in fields.items():
        value = response
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        projected[name] = value
    return projected


def gym_episode_outcome(response: dict) -> dict:
    """Validate Gym outcome fields separately from transport and task quality."""
    result, failure = response.get("result"), response.get("failure")
    if (result is None) == (failure is None) or not isinstance(result if failure is None else failure, dict):
        raise GymCallError("response", "Gym episode must contain exactly one result or failure object")
    outcome = result if failure is None else failure
    if failure is not None and (
        not isinstance(failure.get("failure_reason"), str) or not isinstance(failure.get("terminal"), bool)
    ):
        raise GymCallError("response", "Gym episode failure requires failure_reason and boolean terminal")
    metadata_fields = ("failure_kind", "stage") if failure is not None else ("failure_kind",)
    if any(outcome.get(field) is not None and not isinstance(outcome[field], str) for field in metadata_fields):
        raise GymCallError("response", "Gym episode failure metadata must be strings or null")
    reward = result.get("reward") if result is not None else None
    mask_sample = result.get("mask_sample") if result is not None else None
    if reward is not None and (isinstance(reward, bool) or not isinstance(reward, (int, float))):
        raise GymCallError("response", "Gym episode reward must be a number or null")
    if mask_sample is not None and not isinstance(mask_sample, bool):
        raise GymCallError("response", "Gym episode mask_sample must be a boolean or null")
    return {
        "episode_status": "failed" if failure is not None else "completed",
        "episode_failure_kind": outcome.get("failure_kind"),
        "episode_failure_reason": failure["failure_reason"] if failure is not None else None,
        "episode_failure_stage": failure.get("stage") if failure is not None else None,
        "episode_terminal": failure["terminal"] if failure is not None else None,
        "mask_sample": mask_sample,
        "reward": reward,
    }


class GymColumnGenerator(ColumnGeneratorCellByCell[GymColumnConfig]):
    """Execute one bounded, isolated Gym request per row without replaying it."""

    def _initialize(self) -> None:
        self._slots = BoundedSemaphore(self.config.max_parallel_requests)
        steps = [self.config]
        if self.config.session is not None:
            steps.extend((*self.config.session.initialize, *self.config.session.cleanup))
        self._tokens = {}
        for step in steps:
            if step.bearer_token_env is None:
                continue
            token = os.environ.get(step.bearer_token_env)
            if not token:
                raise ValueError(f"missing Gym credential environment variable: {step.bearer_token_env}")
            if not token.isascii() or "\r" in token or "\n" in token:
                raise ValueError("Gym credentials must contain valid HTTP header values")
            self._tokens[step.bearer_token_env] = token
        column_key = hashlib.sha256(self.config.name.encode()).hexdigest()[:16]
        self._artifact_dir = self.base_dataset_path / "gym" / column_key
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        """Declare service-backed work for DD admission control."""
        endpoint_key = hashlib.sha256(self.config.endpoint.encode()).hexdigest()[:16]
        return SchedulingMetadata.custom_model(
            "data_designer_gym", endpoint_key, "v1", weight=self.config.max_parallel_requests
        )

    def generate(self, data: dict) -> dict:
        """Use DD's supported bridge when called synchronously."""
        return ColumnGenerator.generate(self, data)

    async def agenerate(self, data: dict) -> dict:
        """Retain the original row, evidence, and row-local lifecycle outcome."""
        attempt_id = uuid4().hex
        evidence = {
            "attempt_id": attempt_id,
            "column": self.config.name,
            "status": "failed",
            "row_id": None,
            "provenance": {},
        }
        request = None
        response = None
        answer = None
        acquired = False
        cookies: dict[str, str] = {}
        headers: dict[str, str] = {}
        cleanup_requests = []
        try:
            row_id = _json_compatible(_source_value(data, self.config.row_id_column), self.config.row_id_column)
            if not isinstance(row_id, (str, int)) or isinstance(row_id, bool) or not str(row_id).strip():
                raise ValueError("row_id_column must contain a non-empty string or integer")
            evidence["row_id"] = str(row_id)
            provenance = {
                path: _json_compatible(_source_value(data, path), path) for path in self.config.provenance_columns
            }
            json.dumps(provenance, allow_nan=False)
            evidence["provenance"] = provenance
            request = gym_request_from_row(data, self.config)
            if isinstance(self.config, GymHarnessColumnConfig):
                if "input" not in request or request.get("stream") or request.get("background"):
                    raise ValueError("harness requests require input and foreground, non-streaming responses")
            setup_requests = []
            if self.config.session is not None:
                setup_requests = [(step, gym_request_from_row(data, step)) for step in self.config.session.initialize]
                cleanup_requests = [(step, gym_request_from_row(data, step)) for step in self.config.session.cleanup]
            try:
                await asyncio.wait_for(self._acquire(), timeout=self.config.queue_timeout)
            except asyncio.TimeoutError as exc:
                raise GymCallError("queue_timeout", "Gym admission deadline exceeded before sending a request") from exc
            acquired = True
            async with httpx.AsyncClient(
                timeout=self.config.timeout, follow_redirects=False, trust_env=False
            ) as client:
                try:
                    # Cleanup must also run after a partially successful initialization.
                    response = await asyncio.wait_for(
                        self._execute(client, setup_requests, request, cookies, headers), timeout=self.config.timeout
                    )
                    if isinstance(self.config, GymHarnessColumnConfig):
                        answer = final_assistant_answer(response)
                    elif (
                        isinstance(self.config, GymResourceColumnConfig)
                        and self.config.response_protocol == "gym-episode"
                    ):
                        evidence.update(gym_episode_outcome(response))
                    evidence["status"] = "completed"
                finally:
                    if cleanup_requests:
                        errors, cancelled = await self._cleanup_shielded(client, cleanup_requests, cookies, headers)
                        if errors:
                            evidence["cleanup_errors"] = errors
                            evidence["status"] = "failed"
                            evidence["failure_kind"] = "cleanup"
                            evidence["failure_reason"] = "Gym session cleanup failed"
                            answer = None
                        if cancelled:
                            raise asyncio.CancelledError
        except asyncio.CancelledError:
            evidence.update(status="cancelled", failure_kind="cancelled", failure_reason="Gym call was cancelled")
            raise
        except GymCallError as exc:
            evidence.update(failure_kind=exc.kind, failure_reason=str(exc), http_status=exc.http_status)
            if exc.response is not None:
                response = exc.response
        except (asyncio.TimeoutError, httpx.TimeoutException):
            evidence.update(failure_kind="timeout", failure_reason="Gym execution deadline exceeded")
        except httpx.HTTPError:
            evidence.update(failure_kind="transport", failure_reason="Gym transport failed; request was not retried")
        except (ValueError, TypeError) as exc:
            evidence.update(failure_kind="request", failure_reason=str(exc))
        finally:
            if acquired:
                self._slots.release()
            artifact = self._artifact_dir / f"{attempt_id}.json"
            evidence["artifact"] = str(artifact.relative_to(self.base_dataset_path))
            try:
                artifact.write_text(
                    json.dumps({**evidence, "request": request, "response": response}, allow_nan=False),
                    encoding="utf-8",
                )
            except OSError:
                evidence.update(
                    status="failed",
                    failure_kind="storage",
                    failure_reason="Could not persist Gym evidence",
                    artifact=None,
                )
                answer = None
                logger.error("Could not persist Gym evidence for attempt %s", attempt_id)
        # JSON payloads stay lossless and Parquet-safe even with mixed SQL values or empty objects.
        cell = {
            key: evidence.get(key)
            for key in (
                "attempt_id",
                "column",
                "status",
                "row_id",
                "failure_kind",
                "failure_reason",
                "http_status",
                "artifact",
            )
        }
        cell["provenance_json"] = json.dumps(evidence["provenance"])
        cell["cleanup_errors_json"] = json.dumps(evidence.get("cleanup_errors", []))
        if isinstance(self.config, GymResourceColumnConfig) and self.config.response_protocol == "gym-episode":
            cell.update(
                {
                    key: evidence.get(key)
                    for key in (
                        "episode_status",
                        "episode_failure_kind",
                        "episode_failure_reason",
                        "episode_failure_stage",
                        "episode_terminal",
                        "mask_sample",
                        "reward",
                    )
                }
            )
        if isinstance(self.config, GymResourceColumnConfig) or self.config.response_fields is not None:
            cell["response_json"] = json.dumps(project_response(response, self.config.response_fields))
        if isinstance(self.config, GymHarnessColumnConfig):
            return {**data, self.config.name: answer, self.config.evidence_column: cell}
        return {**data, self.config.name: cell}

    async def _acquire(self) -> None:
        """Wait for a worker-local lifecycle slot with cancellable backoff."""
        delay = 0.01
        while not self._slots.acquire(blocking=False):
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.25)

    async def _execute(self, client, setup_requests, request, cookies, headers) -> dict:
        """Initialize row-local services, then execute the main request once."""
        for step, body in setup_requests:
            try:
                result = await self._post(client, step, body, cookies, headers)
            except GymCallError as exc:
                exc.response = None
                raise
            for header, path in step.response_headers.items():
                try:
                    value = _source_value(result, path)
                except ValueError as exc:
                    raise GymCallError(
                        "response", "initialization did not return the configured session token"
                    ) from exc
                if not isinstance(value, str) or not value or not value.isascii() or "\r" in value or "\n" in value:
                    raise GymCallError("response", "initialization returned an invalid session token")
                headers[header] = value
        return await self._post(client, self.config, request, cookies, headers)

    async def _post(self, client, step, body, cookies, headers, *, allow_empty=False) -> dict:
        """Call a configured route and validate its JSON response."""
        # Explicitly carry Gym cookies between the configured resource and agent hosts.
        client.cookies.clear()
        client.cookies.update(cookies)
        endpoint = step.endpoint or self.config.endpoint
        token_env = step.bearer_token_env
        if token_env is None and endpoint == self.config.endpoint:
            token_env = self.config.bearer_token_env
        auth = {"Authorization": f"Bearer {self._tokens[token_env]}"} if token_env else {}
        response = await client.request(step.method, endpoint + step.route, json=body, headers=headers | auth)
        cookies.update({cookie.name: cookie.value for cookie in response.cookies.jar})
        try:
            value = response.json()
            json.dumps(value, allow_nan=False)
            valid_json = True
        except ValueError:
            value = {"body_text": response.text}
            valid_json = False
        if not 200 <= response.status_code < 300:
            kind = "server" if response.status_code >= 500 else "http"
            raise GymCallError(
                kind, f"Gym returned HTTP {response.status_code}", response=value, http_status=response.status_code
            )
        if allow_empty and response.status_code == 204:
            return {}
        if not valid_json or not isinstance(value, dict):
            raise GymCallError(
                "response", "Gym must return a JSON object", response=value, http_status=response.status_code
            )
        return value

    async def _cleanup(self, client, cleanup_requests, cookies, headers) -> list[dict]:
        """Attempt every cleanup step with its own deadline."""
        errors = []
        for step, body in cleanup_requests:
            try:
                await asyncio.wait_for(
                    self._post(client, step, body, cookies, headers, allow_empty=True),
                    timeout=self.config.session.cleanup_timeout,
                )
            except (ValueError, httpx.HTTPError, asyncio.TimeoutError) as exc:
                errors.append({"route": step.route, "failure_kind": getattr(exc, "kind", "cleanup")})
        return errors

    async def _cleanup_shielded(self, client, cleanup_requests, cookies, headers) -> tuple[list[dict], bool]:
        """Finish bounded cleanup before propagating repeated cancellation."""
        cleanup = asyncio.create_task(self._cleanup(client, cleanup_requests, cookies, headers))
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        return cleanup.result(), cancelled


class GymResourceColumnGenerator(GymColumnGenerator, ColumnGeneratorCellByCell[GymResourceColumnConfig]):
    """Expose resource responses as structured DD evidence."""


class GymHarnessColumnGenerator(GymColumnGenerator, ColumnGeneratorCellByCell[GymHarnessColumnConfig]):
    """Expose final harness text and a declared evidence side-effect column."""
