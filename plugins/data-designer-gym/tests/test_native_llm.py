# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare unchanged native DD behavior with Gym-routed native columns."""

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from data_designer import config as dd
from data_designer.config.run_config import RunConfig
from data_designer.interface.data_designer import DataDesigner
from pydantic import ValidationError

pytest.importorskip("data_designer.engine.models.clients.model_request_executor")

from data_designer_gym.config import GymResourceColumnConfig
from data_designer_gym.native_config import GymNativeLLMColumnConfig


class Backend(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append(body)
        content = self.server.answer
        if self.server.malformed_first and len(self.server.calls) == 1:
            content = "not JSON"
        value = {
            "id": f"chatcmpl-{len(self.server.calls)}",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content, "reasoning_content": "same reasoning"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(value).encode())

    def log_message(self, *args):
        pass


class GymProxy(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path.startswith("/ng-rollout/dd-")
        self.server.routes.append(self.path)
        body.update(json.loads(body.pop("metadata", {}).get("extra_body", "{}")))
        req = Request(
            self.server.backend + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req) as response:
                value = response.read()
                self.send_response(response.status)
        except HTTPError as error:
            value = error.read()
            self.send_response(error.code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(value)

    def log_message(self, *args):
        pass


@contextmanager
def server(handler):
    instance = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    instance.calls, instance.routes = [], []
    instance.answer = ""
    instance.malformed_first = False
    thread = Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance, f"http://127.0.0.1:{instance.server_port}"
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join()


def native_column(kind, skip=False):
    common = dict(
        name="assessment",
        model_alias="judge",
        prompt="Check {{ question }}",
        system_prompt="Original system",
        with_trace="all_messages",
        extract_reasoning_content=True,
        skip={"when": "{{ not evaluate }}", "value": None} if skip else None,
        propagate_skip=False,
    )
    if kind == "judge":
        return dd.LLMJudgeColumnConfig(
            **common,
            scores=[
                dd.Score(
                    name="correctness",
                    description="Check correctness",
                    options={"pass": "Correct", "fail": "Incorrect"},
                )
            ],
        )
    return dd.LLMStructuredColumnConfig(
        **common,
        output_format={
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ready", "hold"]}, "reasoning": {"type": "string"}},
            "required": ["status", "reasoning"],
            "additionalProperties": False,
        },
    )


def wrapped(column, endpoint):
    return GymNativeLLMColumnConfig(
        name=column.name,
        native_column=column,
        model_alias=column.model_alias,
        skip=column.skip,
        propagate_skip=column.propagate_skip,
        drop=column.drop,
        resource=GymResourceColumnConfig(
            name=column.name + "_transport",
            endpoint=endpoint,
            route="/v1/chat/completions",
            row_id_column="seed_id",
            request_column="_gym_native_request",
            timeout=10,
            max_parallel_requests=2,
        ),
        metadata_extra_body_keys=["top_k", "min_p", "repetition_penalty"],
    )


def generate(root, backend, column, rows):
    seed = root / "seed.jsonl"
    root.mkdir()
    seed.write_text("".join(json.dumps(row) + "\n" for row in rows))
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="judge",
                model="original-judge",
                provider="original-provider",
                skip_health_check=True,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=512,
                    timeout=10,
                    max_parallel_requests=2,
                    extra_body={"reasoning_effort": "medium", "top_k": 20},
                ),
            )
        ]
    )
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed)), sampling_strategy="ordered")
    for item in column if isinstance(column, list) else [column]:
        builder.add_column(item)
    designer = DataDesigner(
        artifact_path=root / "artifacts",
        model_providers=[dd.ModelProvider(name="original-provider", endpoint=backend + "/v1", provider_type="openai")],
    )
    designer.set_run_config(RunConfig(max_conversation_restarts=2, max_conversation_correction_steps=0))
    result = designer.create(builder, num_records=len(rows), dataset_name="fixture")
    return result.load_dataset().to_dict(orient="records")


@pytest.mark.parametrize("kind", ["judge", "structured"])
@pytest.mark.parametrize("malformed_first", [False, True])
def test_native_request_parser_restart_and_trace_parity(tmp_path, kind, malformed_first):
    column = native_column(kind)
    answer = json.dumps(
        {"correctness": {"score": "pass", "reasoning": "Grounded"}}
        if kind == "judge"
        else {"status": "ready", "reasoning": "Keep original reasoning"}
    )
    rows = [{"seed_id": "row-1", "question": "Literal {braces}, café", "evaluate": True}]
    with server(Backend) as (backend, backend_url), server(GymProxy) as (gym, gym_url):
        backend.answer, backend.malformed_first, gym.backend = answer, malformed_first, backend_url
        direct = generate(tmp_path / "direct", backend_url, column, rows)
        expected_requests = list(backend.calls)
        backend.calls.clear()
        routed = generate(tmp_path / "routed", backend_url, wrapped(column, gym_url), rows)
        assert backend.calls == expected_requests
        assert len(backend.calls) == (2 if malformed_first else 1)
        assert direct == [{k: v for k, v in row.items() if k != "assessment__gym_calls"} for row in routed]
        calls = json.loads(routed[0]["assessment__gym_calls"])
        assert len(calls) == len(backend.calls)
        assert all(call["status"] == "completed" for call in calls)
        assert len(set(gym.routes)) == 1
        for call in calls:
            path = tmp_path / "routed/artifacts/fixture" / call["artifact"]
            assert json.loads(path.read_text())["status"] == "completed"


@pytest.mark.parametrize("kind", ["judge", "structured"])
def test_skip_has_no_model_calls_and_preserves_null(tmp_path, kind):
    with server(Backend) as (backend, url), server(GymProxy) as (gym, gym_url):
        gym.backend = url
        rows = [{"seed_id": "skip-1", "question": "Do not call", "evaluate": False}]
        result = generate(tmp_path / "skipped", url, wrapped(native_column(kind, skip=True), gym_url), rows)
        assert result[0]["assessment"] is None
        assert backend.calls == [] and gym.routes == []


def test_configuration_does_not_silently_change_native_policy():
    column = native_column("judge")
    config = wrapped(column, "http://localhost:1234")
    with pytest.raises(ValidationError, match="native name"):
        GymNativeLLMColumnConfig.model_validate({**config.model_dump(), "name": "different"})
    with pytest.raises(ValidationError, match="propagation"):
        GymNativeLLMColumnConfig.model_validate({**config.model_dump(), "propagate_skip": True})


@pytest.mark.parametrize("kind", ["judge", "structured"])
def test_parallel_rows_have_independent_native_evidence(tmp_path, kind):
    answer = json.dumps(
        {"correctness": {"score": "fail", "reasoning": "Same verdict"}}
        if kind == "judge"
        else {"status": "hold", "reasoning": "Original issue"}
    )
    rows = [{"seed_id": f"row-{i}", "question": f"Question {i}", "evaluate": True} for i in range(4)]
    with server(Backend) as (backend, url), server(GymProxy) as (gym, gym_url):
        backend.answer, gym.backend = answer, url
        result = generate(tmp_path / "parallel", url, wrapped(native_column(kind), gym_url), rows)
        assert len(result) == len(backend.calls) == len(set(gym.routes)) == 4
        for row in result:
            calls = json.loads(row["assessment__gym_calls"])
            assert len(calls) == 1 and calls[0]["row_id"] == row["seed_id"]
            artifact = tmp_path / "parallel/artifacts/fixture" / calls[0]["artifact"]
            body = json.loads(artifact.read_text())["request"]
            assert row["question"] in json.dumps(body["messages"][-1]["content"])


def test_downstream_judge_early_stop_preserves_failure_and_null(tmp_path):
    first = native_column("judge")
    second = dd.LLMJudgeColumnConfig.model_validate(
        {
            **first.model_dump(),
            "name": "followup",
            "skip": {"when": "{{ assessment.correctness.score != 'pass' }}", "value": None},
        }
    )
    with server(Backend) as (backend, url), server(GymProxy) as (gym, gym_url):
        backend.answer = json.dumps({"correctness": {"score": "fail", "reasoning": "Concrete defect"}})
        gym.backend = url
        result = generate(
            tmp_path / "chain",
            url,
            [wrapped(first, gym_url), wrapped(second, gym_url)],
            [{"seed_id": "row-1", "question": "Original question", "evaluate": True}],
        )
        assert result[0]["assessment"]["correctness"]["score"] == "fail"
        assert result[0]["followup"] is None
        assert len(backend.calls) == 1
