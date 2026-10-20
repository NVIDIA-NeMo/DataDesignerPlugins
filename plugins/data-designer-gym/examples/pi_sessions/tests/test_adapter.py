# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import shlex
import sys
from pathlib import Path

import httpx
import pytest

pytest.importorskip("dd_gym_pi_sessions", reason="Run the isolated pi_sessions example tests separately")

import dd_gym_pi_sessions.store as store_module
from dd_gym_pi_sessions.app import PersistentPiAgent, PersistentPiConfig
from dd_gym_pi_sessions.store import SessionStore, save_json
from fastapi import HTTPException
from fastapi.testclient import TestClient
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from omegaconf import OmegaConf
from pydantic import ValidationError
from responses_api_agents.pi_agent.app import PiAgent


@pytest.fixture
def config(tmp_path):
    return PersistentPiConfig(
        name="pi",
        host="127.0.0.1",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        command=shlex.join([sys.executable, str(Path(__file__).with_name("fake_pi.py"))]),
        model="fixture/model",
        session_root=str(tmp_path / "sessions"),
        timeout=1,
    )


def agent(config):
    return PersistentPiAgent(
        config=config,
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host=config.host, port=config.port),
            global_config_dict=OmegaConf.create({}),
        ),
    )


def request(turn=1, text="VALUE = 1", user="run-row-1"):
    return {"user": user, "metadata": {"turn": str(turn)}, "input": text}


def test_upstream_mapping_and_restart(config):
    assert issubclass(PersistentPiAgent, PiAgent)
    first_agent = agent(config)
    with TestClient(first_agent.setup_webserver()) as client:
        first = client.post("/v1/responses", json=request())
        assert first.status_code == 200, first.text
        assert first.json()["status"] == "completed"
        assert first.json()["_ng_agent_observations"]["source"] == "pi"
        assert [item["type"] for item in first.json()["output"]] == ["function_call", "function_call_output", "message"]
        assert first.json()["usage"]["total_tokens"] == 15
        assert client.get("/stats").json()["active_pi_processes"] == 0
        assert client.post("/v1/responses", json=request()).json() == first.json()
    with TestClient(agent(config).setup_webserver()) as client:
        assert client.post("/v1/responses", json=request()).json() == first.json()
        second = client.post("/v1/responses", json=request(2, "VALUE = 2"))
        assert second.status_code == 200, second.text
        assert first.json()["metadata"]["pi_session_id"] == second.json()["metadata"]["pi_session_id"]
        assert first.json()["metadata"]["workspace"] == second.json()["metadata"]["workspace"]
        assert client.get("/stats").json()["active_pi_processes"] == 0
        assert client.post("/v1/responses", json=request(2, "VALUE = 2")).json() == second.json()
    directory = Path(config.session_root) / "run-row-1"
    before = json.loads((directory / "turn-1.json").read_text())
    after = json.loads((directory / "turn-2.json").read_text())
    assert before["native"]["messages"] == after["native"]["restored_messages"]
    assert before["native"]["pid"] != after["native"]["pid"]
    assert (directory / "workspace/solution.py").read_text() == "VALUE = 2"
    assert after["observations"]["source"] == "pi"
    assert json.loads((directory / "manifest.json").read_text())["status"] == "completed"
    assert any(record["kind"] == "tool_call" for record in after["observations"]["records"])


@pytest.mark.parametrize(
    "updates",
    [
        {"user": "../escape"},
        {"metadata": {"turn": "0"}},
        {"metadata": {"turn": "3"}},
        {"metadata": {"turn": "1", "other": "value"}},
        {"stream": True},
        {"background": True},
        {"tools": [{"type": "function", "name": "unknown", "parameters": {}}]},
    ],
)
def test_invalid_request_never_runs(config, updates):
    with TestClient(agent(config).setup_webserver()) as client:
        assert client.post("/v1/responses", json=request() | updates).status_code == 422
    assert not list(Path(config.session_root).glob("*/manifest.json"))


def test_order_identity_and_configuration(config):
    with TestClient(agent(config).setup_webserver()) as client:
        assert client.post("/v1/responses", json=request(2)).status_code == 409
        assert client.post("/v1/responses", json=request()).status_code == 200
        assert client.post("/v1/responses", json=request(text="changed")).status_code == 409
        assert client.post("/v1/responses", json=request(user="run-row-2", text="isolated")).status_code == 200
    changed = config.model_copy(update={"model": "fixture/changed"})
    with TestClient(agent(changed).setup_webserver()) as client:
        assert client.post("/v1/responses", json=request(2)).status_code == 409
    directory = Path(config.session_root)
    assert (directory / "run-row-1/workspace/solution.py").read_text() == "VALUE = 1"
    assert (directory / "run-row-2/workspace/solution.py").read_text() == "isolated"


def test_concurrency_change_preserves_replay_and_followup(config):
    with TestClient(agent(config).setup_webserver()) as client:
        first = client.post("/v1/responses", json=request())
        assert first.status_code == 200
    changed = config.model_copy(update={"concurrency": config.concurrency + 1})
    with TestClient(agent(changed).setup_webserver()) as client:
        assert client.post("/v1/responses", json=request()).json() == first.json()
        second = client.post("/v1/responses", json=request(2, "VALUE = 2"))
        assert second.status_code == 200
        assert second.json()["metadata"]["pi_session_id"] == first.json()["metadata"]["pi_session_id"]
        assert client.get("/stats").json()["active_pi_processes"] == 0
    assert (Path(config.session_root) / "run-row-1/workspace/solution.py").read_text() == "VALUE = 2"


@pytest.mark.parametrize("text", ["crash", "incomplete", "empty", "hang"])
def test_failed_turn_is_not_replayed(config, text):
    with TestClient(agent(config).setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/responses", json=request(text=text)).status_code == 500
        assert client.get("/stats").json()["active_pi_processes"] == 0
        assert client.post("/v1/responses", json=request(text=text)).status_code == 409
    manifest = json.loads((Path(config.session_root) / "run-row-1/manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["turn"] == 0


def test_history_tampering_is_rejected(config):
    with TestClient(agent(config).setup_webserver()) as client:
        assert client.post("/v1/responses", json=request()).status_code == 200
        manifest = json.loads((Path(config.session_root) / "run-row-1/manifest.json").read_text())
        Path(manifest["session_file"]).write_text("[]")
        assert client.post("/v1/responses", json=request(2)).status_code == 409


async def cancel_request(config):
    instance = agent(config)
    app = instance.setup_webserver()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://pi") as client:
            pending = asyncio.create_task(client.post("/v1/responses", json=request(text="hang")))
            for _ in range(200):
                if (await instance.stats())["active_pi_processes"]:
                    break
                await asyncio.sleep(0.01)
            assert (await instance.stats())["active_pi_processes"] == 1
            pending.cancel()
            await asyncio.sleep(0.02)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 15)
            assert (await instance.stats())["active_pi_processes"] == 0


def test_repeated_cancellation_closes_pi_and_marks_turn_interrupted(config):
    asyncio.run(cancel_request(config))
    manifest = json.loads((Path(config.session_root) / "run-row-1/manifest.json").read_text())
    assert manifest["status"] == "interrupted"


def test_session_store_rejects_concurrent_writers_and_running_state(tmp_path):
    store = SessionStore(tmp_path)
    with store.turn("row", 1, {}, "config"):
        with pytest.raises(HTTPException, match="active writer"):
            with SessionStore(tmp_path).turn("row", 1, {}, "config"):
                pytest.fail("Concurrent writer admitted")
    with pytest.raises(HTTPException, match="interrupted turns"):
        with SessionStore(tmp_path).turn("row", 1, {}, "config"):
            pytest.fail("Uncommitted turn admitted")


def test_receipt_without_committed_manifest_is_not_replayed(tmp_path):
    store = SessionStore(tmp_path)
    with store.turn("row", 1, {}, "config") as turn:
        save_json(turn.directory / "turn-1.json", {"request_hash": hashlib.sha256(b"{}").hexdigest(), "response": {}})
    with pytest.raises(HTTPException, match="interrupted turns"):
        with SessionStore(tmp_path).turn("row", 1, {}, "config"):
            pytest.fail("Unpublished receipt replayed")


class PublicationFailure:
    def __init__(self, original):
        self.original = original

    def __call__(self, path, value):
        if path.name == "manifest.json" and value["status"] == "parked":
            raise OSError("Manifest publication failed")
        return self.original(path, value)


def test_failed_manifest_publication_does_not_commit_the_turn(config, monkeypatch):
    original = store_module.save_json
    monkeypatch.setattr(store_module, "save_json", PublicationFailure(original))
    with TestClient(agent(config).setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/responses", json=request()).status_code == 500
    manifest = json.loads((Path(config.session_root) / "run-row-1/manifest.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["turn"] == 0
    monkeypatch.setattr(store_module, "save_json", original)
    with TestClient(agent(config).setup_webserver()) as client:
        assert client.post("/v1/responses", json=request()).status_code == 409


@pytest.mark.parametrize(
    "updates", [{"num_workers": 2}, {"extra_args": ["--no-session"]}, {"sandbox_provider": "local"}, {"max_turns": 0}]
)
def test_invalid_configuration(config, updates):
    with pytest.raises(ValidationError):
        PersistentPiConfig.model_validate(config.model_dump() | updates)
