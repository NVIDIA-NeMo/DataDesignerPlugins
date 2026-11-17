# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep native DD rendering, parsing and conversation policy around Gym calls."""

import asyncio
import copy
import hashlib
import json
from contextvars import ContextVar

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)
from data_designer.engine.column_generators.generators.llm_completion import (
    LLMJudgeCellGenerator,
    LLMStructuredCellGenerator,
)

from data_designer_gym.columns import GymResourceColumnGenerator
from data_designer_gym.native_config import GymNativeLLMColumnConfig

_CALL = ContextVar("data_designer_gym_native_call")


class _RoutedJudge(LLMJudgeCellGenerator):
    @property
    def model(self):
        return self.gym_model


class _RoutedStructured(LLMStructuredCellGenerator):
    @property
    def model(self):
        return self.gym_model


class _NativeTransport(GymResourceColumnGenerator):
    async def _execute(self, client, setup_requests, request, cookies, headers, evidence):
        identity = f"{self.base_dataset_path}\0{self.config.name}\0{evidence['row_id']}"
        session = hashlib.sha256(identity.encode()).hexdigest()
        step = self.config.model_copy(update={"route": f"/ng-rollout/dd-{session}{self.config.route}"})
        evidence.update(failure_stage="execute", failure_route=step.route)
        return await self._post(client, step, request, cookies, headers)


class _GymCompletionClient:
    """Implement DD's completion protocol using the existing Gym resource column."""

    def __init__(self, owner, provider_name):
        self.owner = owner
        self.provider_name = provider_name

    def supports_chat_completion(self):
        return True

    def supports_embeddings(self):
        return False

    def supports_image_generation(self):
        return False

    def close(self):
        pass  # The resource column owns a bounded HTTP client per call.

    async def aclose(self):
        pass

    def completion(self, request):
        return asyncio.run(self.acompletion(request))

    async def acompletion(self, request):
        """Use native DD serialization/parsing and persist Gym evidence per attempt."""
        from data_designer.engine.models.clients.adapters.openai_compatible import translate_openai_compatible_messages
        from data_designer.engine.models.clients.errors import (
            ProviderError,
            ProviderErrorKind,
            map_http_status_to_provider_error_kind,
        )
        from data_designer.engine.models.clients.parsing import aparse_chat_completion_response
        from data_designer.engine.models.clients.types import TransportKwargs

        state = _CALL.get()
        transport = TransportKwargs.from_request(request)
        # Framework attribution is observational. Other per-call upstream headers
        # cannot be silently passed to Gym itself instead of the model endpoint.
        if set(transport.headers) - {"X-Title"}:
            raise ValueError("native Gym transport does not support per-call upstream header overrides")
        payload = {
            "model": request.model,
            "messages": translate_openai_compatible_messages(
                request.messages, provider_name=self.provider_name, model_name=request.model
            ),
            **transport.body,
        }
        extensions = {key: payload.pop(key) for key in self.owner.config.metadata_extra_body_keys if key in payload}
        if extensions:
            if "metadata" in payload:
                raise ValueError("source metadata conflicts with the explicit Gym extra_body envelope")
            payload["metadata"] = {"extra_body": json.dumps(extensions, sort_keys=True, allow_nan=False)}
        data = {**state["row"], "_gym_native_request": json.dumps(payload, allow_nan=False)}
        result = await self.owner.transport.agenerate(data)
        cell = result[self.owner.transport.config.name]
        state["calls"].append(cell)
        if cell["status"] != "completed":
            status = cell.get("http_status")
            kind = (
                map_http_status_to_provider_error_kind(status)
                if status
                else {
                    "timeout": ProviderErrorKind.TIMEOUT,
                    "transport": ProviderErrorKind.API_CONNECTION,
                }.get(cell.get("failure_kind"), ProviderErrorKind.CLIENT_ERROR)
            )
            raise ProviderError(
                kind=kind,
                message="Gym model call failed; inspect retained attempt artifact",
                status_code=status,
                provider_name=self.provider_name,
                model_name=request.model,
            )
        return await aparse_chat_completion_response(json.loads(cell["response_json"]))


class GymNativeLLMColumnGenerator(
    ColumnGeneratorCellByCell[GymNativeLLMColumnConfig],
    ColumnGeneratorWithModelRegistry[GymNativeLLMColumnConfig],
):
    """Delegate to native DD generators with an isolated Gym-backed model facade.

    DD 1.0.0rc2 exposes no facade transport replacement method. The two isolated
    shallow copies below replace only ModelFacade._client and executor._inner;
    they never mutate the registry or shared provider/client. Usage accounting,
    admission, provider retry and native conversation policy remain upstream.
    Unsupported facade layouts fail before sending requests.
    """

    def _initialize(self):
        from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
        from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor

        original = self.get_model(self.config.model_alias)
        if not isinstance(original._client, ModelRequestExecutor) or not isinstance(
            original._client._inner, OpenAICompatibleClient
        ):
            raise ValueError("native Gym columns require the tested DD 1.0.0rc2 OpenAI-compatible facade layout")
        self.transport = _NativeTransport(self.config.resource, self.resource_provider)
        facade = copy.copy(original)
        facade._client = copy.copy(original._client)
        facade._client._inner = _GymCompletionClient(self, original.model_provider_name)
        kind = _RoutedJudge if self.config.native_column.column_type == "llm-judge" else _RoutedStructured
        self.native = kind(self.config.native_column, self.resource_provider)
        self.native.gym_model = facade

    def get_scheduling_metadata(self):
        """Keep the original provider/model concurrency bucket."""
        return ColumnGeneratorWithModelRegistry.get_scheduling_metadata(self)

    def generate(self, data):
        """Preserve native synchronous generation when invoked from a worker thread."""
        state = {"row": data, "calls": []}
        token = _CALL.set(state)
        try:
            return self.native.generate(data)
        finally:
            data[self.config.name + "__gym_calls"] = json.dumps(state["calls"], allow_nan=False)
            _CALL.reset(token)

    async def agenerate(self, data):
        """Preserve native parsing/restarts with row-local call context."""
        state = {"row": data, "calls": []}
        token = _CALL.set(state)
        try:
            return await self.native.agenerate(data)
        finally:
            data[self.config.name + "__gym_calls"] = json.dumps(state["calls"], allow_nan=False)
            _CALL.reset(token)
