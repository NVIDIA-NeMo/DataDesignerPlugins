# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the real Data Designer engine with controlled model transport only."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import create_autospec, patch

from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.testing import make_stub_completion_response

from data_designer_retrieval_sdg.multimodal import models
from data_designer_retrieval_sdg.multimodal.execution import generate_candidates as execute_candidates
from data_designer_retrieval_sdg.multimodal.inference import Request

# Resume tests may replace transport more than once within one monkeypatch scope.
COMPLETION_SIGNATURE = ModelFacade.acompletion


class ScriptedTransport:
    """Translate Data Designer model messages into deterministic fixture requests."""

    def __init__(self, inference):
        self.inference = inference
        self.messages = []

    async def complete(self, facade: ModelFacade, messages, **kwargs):
        self.messages.append(deepcopy([message.to_dict() for message in messages]))
        content = messages[0].to_dict()["content"]
        images = []
        if isinstance(content, list):
            text = "\n".join(item["text"] for item in content if item["type"] == "text")
            images = [Path("test-attachment") for item in content if item["type"] in {"image", "image_url"}]
        else:
            text = content
        schema = json.loads(text.split("<response_schema>\n")[1].split("\n</response_schema>")[0])
        title = schema["title"]
        model = (
            models.QueryBatch.for_slot_count(schema["properties"]["queries"]["minItems"])
            if title.startswith("QueryBatch")
            else getattr(models, title)
        )
        request = Request(
            text.split("\n\n* Your response must be in JSON format.")[0],
            model,
            "judge" if facade.model_alias in models.JudgeModels.model_fields else facade.model_alias,
            tuple(images),
            judge_stage=facade.model_alias if facade.model_alias in models.JudgeModels.model_fields else None,
        )
        response = self.inference.generate([request])[0]
        return make_stub_completion_response(content=response.model_dump_json())


def install_inference(monkeypatch, inference):
    """Replace only hosted inference; parsing, DAG execution and artifacts remain real."""
    monkeypatch.setenv("TEST_SDG_KEY", "test-placeholder")
    transport = ScriptedTransport(inference)
    completion = create_autospec(COMPLETION_SIGNATURE, side_effect=transport.complete)
    completion.request_messages = transport.messages
    monkeypatch.setattr(ModelFacade, "acompletion", completion)
    return completion


def generate_candidates(config, sources, contexts, inference):
    """Run the workflow with deterministic model responses."""
    transport = ScriptedTransport(inference)
    with (
        patch.dict("os.environ", {"TEST_SDG_KEY": "test-placeholder"}),
        patch.object(ModelFacade, "acompletion", autospec=COMPLETION_SIGNATURE, side_effect=transport.complete),
    ):
        return execute_candidates(config, sources, contexts)
