# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from functools import partial
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.storage.artifact_storage import ArtifactStorage
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner
from pydantic import ValidationError

from data_designer_gym.columns import (
    GymCallError,
    GymHarnessColumnGenerator,
    GymResourceColumnGenerator,
    final_assistant_answer,
    gym_request_from_row,
)
from data_designer_gym.config import GymHarnessColumnConfig, GymResourceColumnConfig
from data_designer_gym.plugin import harness_plugin, resource_plugin


def assistant_message(text: str, **kwargs) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text}],
        **kwargs,
    }


@dataclass
class GymHTTP:
    status: int = 200
    response: object = field(default_factory=lambda: {"rows": [[3]], "execution_status": "success"})
    delay: float = 0
    cleanup_status: int = 200
    cleanup_delay: float = 0
    seed_status: int = 200
    seed_delay: float = 0
    seed_response: dict = field(default_factory=lambda: {"token": "row-token"})
    response_headers: dict = field(default_factory=dict)
    responses_by_sql: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    active: int = 0
    peak: int = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((request.url.path, body, dict(request.headers)))
        headers = self.response_headers.get(request.url.path, {})
        if request.url.path == "/seed_session":
            await asyncio.sleep(self.seed_delay)
            return httpx.Response(
                self.seed_status,
                json=self.seed_response,
                headers={"set-cookie": f"episode={body['id']}; Path=/"} | headers,
            )
        if request.url.path == "/cleanup":
            await asyncio.sleep(self.cleanup_delay)
            return httpx.Response(self.cleanup_status, json={})
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.response == "transport":
                raise httpx.ConnectError("secret URL should not be recorded", request=request)
            if self.response == "invalid-json":
                return httpx.Response(self.status, text="not JSON")
            if self.response == "nonfinite":
                return httpx.Response(200, text='{"rows": [[NaN]]}')
            if body.get("sql") in self.responses_by_sql:
                status, response = self.responses_by_sql[body["sql"]]
                return httpx.Response(status, json=response)
            return httpx.Response(self.status, json=self.response, headers=headers)
        finally:
            self.active -= 1


@pytest.fixture()
def service(monkeypatch) -> GymHTTP:
    service = GymHTTP()
    monkeypatch.setattr(httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(service)))
    return service


@pytest.fixture()
def resources(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(artifact_storage=ArtifactStorage(artifact_path=tmp_path, dataset_name="test"))


@pytest.fixture()
def config() -> GymResourceColumnConfig:
    return GymResourceColumnConfig(
        name="execution",
        endpoint="http://resource:8000",
        route="/execute",
        row_id_column="id",
        request_columns={"sql": "candidate.sql", "database_path": "database"},
        provenance_columns=["candidate.id", "variant"],
    )


@pytest.fixture()
def row() -> dict:
    return {
        "id": "seed-1",
        "candidate": {"id": "candidate-2", "sql": "select 3"},
        "database": "toy.db",
        "variant": "original",
        "keep": "upstream",
    }


@pytest.fixture()
def session() -> dict:
    return {
        "initialize": [
            {
                "endpoint": "http://initializer:9000",
                "route": "/seed_session",
                "request_columns": {"id": "id"},
                "response_headers": {"X-Session": "token"},
            }
        ],
        "cleanup": [{"endpoint": "http://initializer:9000", "route": "/cleanup"}],
        "cleanup_timeout": 0.1,
    }


@pytest.mark.parametrize("plugin", [resource_plugin, harness_plugin])
def test_valid_column_plugin(plugin) -> None:
    assert_valid_plugin(plugin)


def test_config_round_trip_and_dependencies(config, session) -> None:
    config = GymResourceColumnConfig.model_validate(
        config.model_dump()
        | {
            "session": session,
            "response_protocol": "gym-episode",
            "response_fields": {"score": "result.reward"},
        }
    )
    config = GymResourceColumnConfig.model_validate_json(config.model_dump_json())
    assert config.required_columns == ["candidate", "database", "id", "variant"]
    assert config.side_effect_columns == []


@pytest.mark.parametrize(
    "updates",
    [
        {"endpoint": "file:///tmp/db"},
        {"endpoint": "http://user:password@host"},
        {"endpoint": "http://host?key=secret"},
        {"route": "//host/execute"},
        {"route": "/../execute"},
        {"row_id_column": ""},
        {"request_columns": {"sql": "a..b"}},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"max_parallel_requests": 0},
        {"request_column": "request"},
        {"name": "candidate"},
        {"bearer_token_env": "bad key"},
        {"session": {"initialize": [{"route": "/seed_session"}]}},
        {"response_fields": {"": "result.reward"}},
        {"response_fields": {" score": "result.reward"}},
        {"response_fields": {"score": "result..reward"}},
        {"response_fields": {"score": ""}},
        {"response_protocol": "unknown"},
    ],
)
def test_invalid_config(config, updates) -> None:
    with pytest.raises(ValidationError):
        GymResourceColumnConfig.model_validate(config.model_dump() | updates)


def test_resource_preserves_evidence_and_row(service, resources, config, row) -> None:
    output = GymResourceColumnGenerator(config, resources).generate(row)
    assert {key: output[key] for key in row} == row
    evidence = output["execution"]
    assert evidence["status"] == "completed"
    assert evidence["row_id"] == "seed-1"
    assert json.loads(evidence["provenance_json"]) == {"candidate.id": "candidate-2", "variant": "original"}
    assert json.loads(evidence["response_json"])["rows"] == [[3]]
    artifact = json.loads((resources.artifact_storage.base_dataset_path / evidence["artifact"]).read_text())
    assert artifact["request"] == {"sql": "select 3", "database_path": "toy.db"}
    assert artifact["response"] == service.response
    assert len(service.calls) == 1


@pytest.mark.parametrize(
    "fields",
    [{"score": "result.reward", "optional": "result.missing", "list_attribute": "result.transcript.count"}, {}],
)
def test_resource_response_projection_keeps_full_artifact(service, resources, config, row, fields) -> None:
    service.response = {"result": {"reward": 0, "transcript": ["large payload"]}}
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"response_fields": fields})
    output = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    expected = {"score": 0, "optional": None, "list_attribute": None} if fields else {}
    assert json.loads(output["response_json"]) == expected
    artifact = json.loads((resources.artifact_storage.base_dataset_path / output["artifact"]).read_text())
    assert artifact["response"] == service.response
    assert service.response["result"]["transcript"] == ["large payload"]


@pytest.mark.parametrize(
    ("payload", "status", "reward", "mask", "kind", "terminal"),
    [
        ({"result": {"reward": 0, "mask_sample": False}}, "completed", 0, False, None, None),
        (
            {"result": {"reward": 0, "mask_sample": True, "failure_kind": "usersim:scorer_inconclusive"}},
            "completed",
            0,
            True,
            "usersim:scorer_inconclusive",
            None,
        ),
        ({"result": {}}, "completed", None, None, None, None),
        (
            {
                "failure": {
                    "failure_reason": "session expired",
                    "terminal": False,
                    "failure_kind": "session_expired",
                    "stage": "actor",
                }
            },
            "failed",
            None,
            None,
            "session_expired",
            False,
        ),
        ({"failure": {"failure_reason": "invalid task", "terminal": True}}, "failed", None, None, None, True),
    ],
)
def test_episode_outcome_is_separate_from_call_status(
    service, resources, config, row, payload, status, reward, mask, kind, terminal
) -> None:
    service.response = {
        "episode_id": {"rollout_id": "run-row-1", "attempt": 0},
        "task_id": {"taskset": "usersim:example", "task_id": "scenario-1"},
        **payload,
    }
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"response_protocol": "gym-episode"})
    output = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    assert output["status"] == "completed"
    assert output["failure_kind"] is None
    assert output["episode_status"] == status
    assert output["reward"] == reward
    assert output["mask_sample"] is mask
    assert output["episode_failure_kind"] == kind
    assert output["episode_terminal"] is terminal
    outcome = payload.get("failure", payload.get("result"))
    assert output["episode_failure_reason"] == outcome.get("failure_reason")
    assert output["episode_failure_stage"] == outcome.get("stage")
    artifact = json.loads((resources.artifact_storage.base_dataset_path / output["artifact"]).read_text())
    assert artifact["episode_status"] == status
    assert artifact["response"] == service.response


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"result": {}, "failure": {}},
        {"result": []},
        {"failure": []},
        {"failure": {"terminal": False}},
        {"failure": {"failure_reason": "bad", "terminal": "false"}},
        {"result": {"reward": True}},
        {"result": {"reward": "0"}},
        {"result": {"mask_sample": "false"}},
        {"failure": {"failure_reason": "bad", "terminal": False, "failure_kind": 3}},
    ],
)
def test_malformed_episode_retains_response_and_cleans_up(service, resources, config, row, session, payload) -> None:
    service.response = payload
    config = GymResourceColumnConfig.model_validate(
        config.model_dump() | {"response_protocol": "gym-episode", "session": session}
    )
    output = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    assert output["status"] == "failed"
    assert output["failure_kind"] == "response"
    assert output["episode_status"] is None
    assert [call[0] for call in service.calls] == ["/seed_session", "/execute", "/cleanup"]
    artifact = json.loads((resources.artifact_storage.base_dataset_path / output["artifact"]).read_text())
    assert artifact["response"] == payload


def test_generic_json_does_not_interpret_episode_fields(service, resources, config, row) -> None:
    service.response = {"failure": {"description": "domain-specific JSON"}}
    output = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    assert output["status"] == "completed"
    assert "episode_status" not in output
    assert json.loads(output["response_json"]) == service.response


def test_episode_cleanup_failure_preserves_outcome(service, resources, config, row, session) -> None:
    service.response = {"result": {"reward": 0, "mask_sample": False}}
    service.cleanup_status = 500
    config = GymResourceColumnConfig.model_validate(
        config.model_dump() | {"response_protocol": "gym-episode", "session": session}
    )
    output = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    assert output["status"] == "failed"
    assert output["failure_kind"] == "cleanup"
    assert output["episode_status"] == "completed"
    assert output["reward"] == 0


def test_episode_projection_in_dd_workflow(service, tmp_path) -> None:
    service.responses_by_sql = {
        "zero": (200, {"result": {"reward": 0, "mask_sample": False, "transcript": ["large"]}}),
        "lost": (200, {"failure": {"failure_reason": "session lost", "terminal": False}}),
    }
    builder = DataDesignerConfigBuilder().with_seed_dataset(
        DataFrameSeedSource(df=pd.DataFrame([{"id": 1, "sql": "zero"}, {"id": 2, "sql": "lost"}]))
    )
    builder.add_column(
        GymResourceColumnConfig(
            name="episode",
            endpoint="http://host",
            route="/run",
            row_id_column="id",
            request_columns={"sql": "sql"},
            response_protocol="gym-episode",
            response_fields={"score": "result.reward"},
        )
    )
    builder.add_column(
        ExpressionColumnConfig(name="returned_result", expr="{{ episode.episode_status == 'completed' }}")
    )
    dataset = DataDesigner(artifact_path=tmp_path / "artifacts").create(builder, num_records=2).load_dataset()
    assert dataset["returned_result"].tolist() == ["True", "False"]
    assert [row["status"] for row in dataset["episode"]] == ["completed", "completed"]
    assert [json.loads(row["response_json"]) for row in dataset["episode"]] == [{"score": 0}, {"score": None}]


@pytest.mark.parametrize(
    ("status", "response", "kind"),
    [
        (500, {"error": "server"}, "server"),
        (422, {"detail": "invalid SQL"}, "http"),
        (200, "transport", "transport"),
        (200, "invalid-json", "response"),
        (200, [], "response"),
        (200, "nonfinite", "response"),
    ],
)
def test_request_failures_are_not_retried(service, resources, config, row, status, response, kind) -> None:
    service.status, service.response = status, response
    output = GymResourceColumnGenerator(config, resources).generate(row)
    assert output["execution"]["status"] == "failed"
    assert output["execution"]["failure_kind"] == kind
    assert output["execution"]["failure_stage"] == "execute"
    assert output["execution"]["failure_route"] == "/execute"
    assert len(service.calls) == 1


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_create_with_mixed_responses_and_empty_provenance(service, tmp_path, suffix) -> None:
    service.responses_by_sql = {
        "select": (200, {"rows": [[3, "Alice", None]], "metadata": {}}),
        "broken": (422, {"detail": ["invalid"]}),
    }
    seed = pd.DataFrame([{"id": 1, "sql": "select"}, {"id": 2, "sql": "broken"}])
    source = LocalFileSeedSource.from_dataframe(seed, str(tmp_path / "seed.parquet"))
    builder = DataDesignerConfigBuilder().with_seed_dataset(source)
    builder.add_column(
        GymResourceColumnConfig(
            name="result", endpoint="http://host", route="/execute", row_id_column="id", request_columns={"sql": "sql"}
        )
    )
    path = tmp_path / f"workflow.{suffix}"
    builder.write_config(path)
    builder = DataDesignerConfigBuilder.from_config(path)
    dataset = DataDesigner(artifact_path=tmp_path / "artifacts").create(builder, num_records=2).load_dataset()
    outputs = dataset["result"].tolist()
    assert [output["status"] for output in outputs] == ["completed", "failed"]
    assert json.loads(outputs[0]["response_json"])["rows"] == [[3, "Alice", None]]
    assert all(output["provenance_json"] == "{}" for output in outputs)


def test_provenance_nan_is_reported_before_http(service, resources, config, row) -> None:
    output = GymResourceColumnGenerator(config, resources).generate({**row, "variant": float("nan")})
    assert output["execution"]["failure_kind"] == "request"
    assert output["execution"]["failure_stage"] == "request"
    assert output["execution"]["failure_route"] is None
    assert not service.calls


def test_bearer_tokens_are_scoped_to_step_endpoint(service, resources, config, row, session, monkeypatch) -> None:
    monkeypatch.setenv("GYM_TEST_TOKEN", "main-token")
    config = GymResourceColumnConfig.model_validate(
        config.model_dump() | {"session": session, "bearer_token_env": "GYM_TEST_TOKEN"}
    )
    GymResourceColumnGenerator(config, resources).generate(row)
    assert "authorization" not in service.calls[0][2]
    assert service.calls[1][2]["authorization"] == "Bearer main-token"
    assert "authorization" not in service.calls[-1][2]


@pytest.mark.parametrize("route", ["/execute", "/session_transition"])
@pytest.mark.parametrize(
    ("cookie", "expected"),
    [
        ("episode=; Max-Age=0; Path=/", None),
        ("episode=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/", None),
        ("episode=; Max-Age=0; Domain=gym.example; Path=/", None),
        ("episode=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Domain=.GYM.EXAMPLE; Path=/", None),
        ('episode="é"; Max-Age=0; Domain=gym.example; Path=/', None),
        ("episode=replaced; Path=/", "replaced"),
        ("episode=replaced; Domain=.GYM.EXAMPLE; Path=/", "replaced"),
        ("other=extra; Path=/", "unchanged"),
        ("episode=replaced; Domain=unrelated.example; Path=/", "unchanged"),
        ("episode=; Max-Age=0; Domain=unrelated.example; Path=/", "unchanged"),
        ("episode=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Domain=unrelated.example; Path=/", "unchanged"),
        ('episode="é"; Max-Age=0; Domain=unrelated.example; Path=/', "unchanged"),
        ("episode=replaced; Max-Age=invalid; Path=/", "unchanged"),
        ("episode=replaced; Max-Age=60; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/", "replaced"),
    ],
)
def test_lifecycle_cookie_changes_and_row_isolation(service, resources, config, row, session, route, cookie, expected):
    service.response_headers[route] = httpx.Headers({"set-cookie": cookie}, encoding="utf-8")
    if route == "/session_transition":
        session["initialize"].append({"endpoint": "http://transition.gym.example:9001", "route": route})
    config = GymResourceColumnConfig.model_validate(
        config.model_dump() | {"endpoint": "http://resource.gym.example:8000", "session": session}
    )
    generator = GymResourceColumnGenerator(config, resources)
    for row_id in ("first", "second"):
        service.calls.clear()
        assert generator.generate({**row, "id": row_id})["execution"]["status"] == "completed"
        assert "cookie" not in service.calls[0][2]
        transition = next(index for index, call in enumerate(service.calls) if call[0] == route)
        assert SimpleCookie(service.calls[transition][2]["cookie"])["episode"].value == row_id
        for _, _, headers in service.calls[transition + 1 :]:
            sent = SimpleCookie(headers.get("cookie", ""))
            if expected is None:
                assert "episode" not in sent
            else:
                assert sent["episode"].value == (row_id if expected == "unchanged" else expected)


def test_queue_wait_does_not_use_execution_timeout(service, resources, config, row) -> None:
    service.delay = 0.03
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"max_parallel_requests": 1, "timeout": 0.06})
    outputs = asyncio.run(concurrent_rows(GymResourceColumnGenerator(config, resources), row))
    assert all(output["execution"]["status"] == "completed" for output in outputs)


def test_queue_deadline_is_distinct(service, resources, config, row) -> None:
    service.delay = 0.04
    config = GymResourceColumnConfig.model_validate(
        config.model_dump() | {"max_parallel_requests": 1, "queue_timeout": 0.01}
    )
    outputs = asyncio.run(concurrent_rows(GymResourceColumnGenerator(config, resources), row))
    assert outputs[0]["execution"]["status"] == "completed"
    assert all(output["execution"]["failure_kind"] == "queue_timeout" for output in outputs[1:])
    assert all(output["execution"]["failure_stage"] == "admission" for output in outputs[1:])
    assert all(output["execution"]["failure_route"] is None for output in outputs[1:])
    assert len(service.calls) == 1


async def repeatedly_cancel_row(generator, row) -> None:
    task = asyncio.create_task(generator.agenerate(row))
    await asyncio.sleep(0.02)
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_repeated_cancellation_finishes_cleanup(service, resources, config, row, session) -> None:
    service.delay, service.cleanup_delay = 1, 0.04
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"session": session})
    asyncio.run(repeatedly_cancel_row(GymResourceColumnGenerator(config, resources), row))
    assert service.calls[-1][0] == "/cleanup"


def test_cleanup_timeout_continues_remaining_steps(service, resources, config, row, session) -> None:
    service.cleanup_delay = 0.03
    session = {**session, "cleanup_timeout": 0.01, "cleanup": session["cleanup"] * 2}
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"session": session})
    output = GymResourceColumnGenerator(config, resources).generate(row)
    assert len(json.loads(output["execution"]["cleanup_errors_json"])) == 2
    assert sum(route == "/cleanup" for route, _, _ in service.calls) == 2


def test_cleanup_accepts_no_content(service, resources, config, row, session) -> None:
    service.cleanup_status = 204
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"session": session})
    evidence = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    assert evidence["status"] == "completed"
    assert evidence["failure_stage"] is None
    assert evidence["failure_route"] is None


@pytest.mark.parametrize("payload", ["[]", "not JSON", '{"x": NaN}', {"x": float("inf")}])
def test_request_boundary(payload) -> None:
    config = GymResourceColumnConfig(
        name="result", endpoint="http://host", route="/execute", row_id_column="id", request_column="request"
    )
    with pytest.raises(ValueError):
        gym_request_from_row({"request": payload}, config)


@pytest.mark.parametrize(
    "output",
    [
        [],
        [{"type": "function_call"}],
        [{"type": "function_call_output", "output": "answer"}],
        [assistant_message("thinking", phase="commentary")],
        [assistant_message("answer", status="incomplete")],
        [assistant_message("answer"), {"type": "function_call"}],
        [{"type": "function_call", "call_id": "pending"}, assistant_message("answer")],
        [{"type": "custom_tool_call", "call_id": "pending"}, assistant_message("answer")],
    ],
)
def test_intermediate_output_is_not_answer(output) -> None:
    with pytest.raises(GymCallError):
        final_assistant_answer({"status": "completed", "output": output})


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "incomplete"},
        {"incomplete_details": {"reason": "max_output_tokens"}},
        {"error": {"message": "failed"}},
        {"required_action": {"type": "submit_tool_outputs"}},
    ],
)
def test_incomplete_response_is_not_answer(updates) -> None:
    with pytest.raises(GymCallError):
        final_assistant_answer({"status": "completed", "output": [assistant_message("partial")], **updates})


@pytest.mark.parametrize("status", [None, "completed"])
def test_completed_gym_adapter_answer(status) -> None:
    assert final_assistant_answer({"status": status, "output": [assistant_message("answer")]}) == "answer"


def test_harness_final_answer_trace_and_cleanup(service, resources, session) -> None:
    service.response = {
        "status": "completed",
        "output": [
            assistant_message("thinking"),
            {"type": "function_call", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "3"},
            assistant_message("3"),
        ],
    }
    config = GymHarnessColumnConfig(
        name="answer",
        endpoint="http://agent:8001",
        row_id_column="id",
        request_columns={"input": "question"},
        evidence_column="trace",
        session=session,
    )
    output = GymHarnessColumnGenerator(config, resources).generate({"id": "seed-2", "question": "count"})
    assert output["answer"] == "3"
    assert output["trace"]["status"] == "completed"
    assert "response_json" not in output["trace"]
    assert [call[0] for call in service.calls] == ["/seed_session", "/v1/responses", "/cleanup"]
    assert "episode=seed-2" in service.calls[1][2]["cookie"]
    assert service.calls[1][2]["x-session"] == "row-token"
    artifact = json.loads((resources.artifact_storage.base_dataset_path / output["trace"]["artifact"]).read_text())
    assert artifact["response"] == service.response
    assert "row-token" not in json.dumps(artifact)


def test_harness_projects_response_into_existing_evidence(service, resources) -> None:
    service.response = {"output": [assistant_message("answer")], "usage": {"total_tokens": 12}}
    config = GymHarnessColumnConfig(
        name="answer",
        evidence_column="trace",
        endpoint="http://agent",
        row_id_column="id",
        request_columns={"input": "question"},
        response_fields={"tokens": "usage.total_tokens"},
    )
    output = GymHarnessColumnGenerator(config, resources).generate({"id": "row-1", "question": "test"})
    assert output["answer"] == "answer"
    assert json.loads(output["trace"]["response_json"]) == {"tokens": 12}
    artifact = json.loads((resources.artifact_storage.base_dataset_path / output["trace"]["artifact"]).read_text())
    assert artifact["response"] == service.response


async def concurrent_rows(generator, row) -> list:
    return await asyncio.gather(*(generator.agenerate({**row, "id": f"row-{index}"}) for index in range(6)))


def test_bounded_concurrency_and_session_isolation(service, resources, config, row, session) -> None:
    service.delay = 0.02
    config = GymResourceColumnConfig.model_validate(
        config.model_dump() | {"max_parallel_requests": 2, "session": session}
    )
    outputs = asyncio.run(concurrent_rows(GymResourceColumnGenerator(config, resources), row))
    assert service.peak == 2
    assert len({output["execution"]["attempt_id"] for output in outputs}) == 6
    cookies = [headers["cookie"] for route, _, headers in service.calls if route == "/execute"]
    assert sorted(cookies) == [f"episode=row-{index}" for index in range(6)]
    assert sum(route == "/cleanup" for route, _, _ in service.calls) == 6


def test_timeout_cleanup_and_no_retry(service, resources, config, row, session) -> None:
    service.delay = 0.1
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"timeout": 0.02, "session": session})
    output = GymResourceColumnGenerator(config, resources).generate(row)
    assert output["execution"]["failure_kind"] == "timeout"
    assert output["execution"]["failure_stage"] == "execute"
    assert output["execution"]["failure_route"] == "/execute"
    assert [call[0] for call in service.calls] == ["/seed_session", "/execute", "/cleanup"]


async def cancel_row(generator, row) -> None:
    task = asyncio.create_task(generator.agenerate(row))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_cancellation_cleans_up_and_retains_artifact(service, resources, config, row, session) -> None:
    service.delay = 1
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"session": session})
    asyncio.run(cancel_row(GymResourceColumnGenerator(config, resources), row))
    assert service.calls[-1][0] == "/cleanup"
    artifacts = list(resources.artifact_storage.base_dataset_path.glob("gym/*/*.json"))
    assert len(artifacts) == 1
    assert json.loads(artifacts[0].read_text())["status"] == "cancelled"


def test_cleanup_failure_is_visible(service, resources, config, row, session) -> None:
    service.cleanup_status = 500
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"session": session})
    evidence = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    assert evidence["status"] == "failed"
    assert evidence["failure_kind"] == "cleanup"
    assert evidence["failure_stage"] == "cleanup"
    assert evidence["failure_route"] == "/cleanup"
    assert json.loads(evidence["cleanup_errors_json"]) == [{"route": "/cleanup", "failure_kind": "server"}]


def test_workflow_dependencies_round_trip_and_downstream_consumption(service, tmp_path, config, row) -> None:
    builder = DataDesignerConfigBuilder().with_seed_dataset(DataFrameSeedSource(df=pd.DataFrame([row])))
    builder.add_column(ExpressionColumnConfig(name="accepted", expr="{{ execution.status == 'completed' }}"))
    builder.add_column(config)
    result = DataDesigner(artifact_path=tmp_path / "workflow").preview(builder, num_records=1)
    output = result.dataset.iloc[0].to_dict()
    assert output["keep"] == "upstream"
    assert output["accepted"] in (True, "True")
    assert len(service.calls) == 1


def test_missing_session_token_is_response_failure(service, resources, config, row, session) -> None:
    service.seed_response = {}
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"session": session})
    output = GymResourceColumnGenerator(config, resources).generate(row)
    assert output["execution"]["failure_kind"] == "response"
    assert output["execution"]["failure_stage"] == "initialize"
    assert output["execution"]["failure_route"] == "/seed_session"
    assert [call[0] for call in service.calls] == ["/seed_session", "/cleanup"]


@pytest.mark.parametrize(("status", "delay", "kind"), [(503, 0, "server"), (200, 0.1, "timeout")])
def test_initialization_failure_is_identified_without_tokens(
    service, resources, config, row, session, status, delay, kind
):
    service.seed_status, service.seed_delay = status, delay
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"session": session, "timeout": 0.02})
    evidence = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    assert evidence["failure_kind"] == kind
    assert evidence["failure_stage"] == "initialize"
    assert evidence["failure_route"] == "/seed_session"
    artifact = json.loads((resources.artifact_storage.base_dataset_path / evidence["artifact"]).read_text())
    assert artifact["failure_stage"] == "initialize"
    assert artifact["failure_route"] == "/seed_session"
    assert artifact["response"] is None
    assert "row-token" not in json.dumps(artifact)
    assert [call[0] for call in service.calls] == ["/seed_session", "/cleanup"]


def test_execution_failure_keeps_stage_when_cleanup_also_fails(service, resources, config, row, session):
    service.status, service.cleanup_status = 422, 500
    config = GymResourceColumnConfig.model_validate(config.model_dump() | {"session": session})
    evidence = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    assert evidence["failure_kind"] == "http"
    assert evidence["failure_stage"] == "execute"
    assert evidence["failure_route"] == "/execute"
    assert json.loads(evidence["cleanup_errors_json"]) == [{"route": "/cleanup", "failure_kind": "server"}]


def fail_write(*args, **kwargs):
    raise OSError("disk full")


def test_artifact_write_failure_is_visible(service, resources, config, row, monkeypatch) -> None:
    monkeypatch.setattr(Path, "write_text", fail_write)
    output = GymResourceColumnGenerator(config, resources).generate(row)["execution"]
    assert output["status"] == "failed"
    assert output["failure_kind"] == "storage"
    assert output["failure_stage"] == "storage"
    assert output["failure_route"] is None
    assert output["artifact"] is None


@pytest.mark.parametrize("field", ["stream", "background"])
def test_dynamic_harness_request_validation(service, resources, field) -> None:
    config = GymHarnessColumnConfig(
        name="answer", evidence_column="evidence", endpoint="http://host", row_id_column="id", request_column="request"
    )
    output = GymHarnessColumnGenerator(config, resources).generate(
        {"id": "row", "request": {"input": "question", field: True}}
    )
    assert output["answer"] is None
    assert output["evidence"]["failure_kind"] == "request"
    assert not service.calls
