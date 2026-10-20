# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from uuid import uuid4

import pytest

pytest.importorskip("dd_gym_pi_sessions", reason="Run the isolated pi_sessions example tests separately")

import httpx


class ModelFixture(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(getattr(self.server, "delay", 0))
        messages = body["messages"]
        if messages[-1]["role"] == "user":
            text = str(messages[-1]["content"])
            token = "alpha" if "alpha" in str(messages) else "beta"
            value = 2 if "second turn" in text else 1
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_" + uuid4().hex,
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": json.dumps(
                                {"path": "solution.py", "content": f"TOKEN = '{token}'\nVALUE = {value}\n"}
                            ),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            delta, finish = {"role": "assistant", "content": "Saved."}, "stop"
        common = {
            "id": "chatcmpl_" + uuid4().hex,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body["model"],
        }
        chunks = [
            common | {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            common
            | {
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        ]
        payload = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_adapter(root, backend):
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        address = port.getsockname()[1]
    config = {
        "name": "pi",
        "host": "127.0.0.1",
        "port": address,
        "entrypoint": "app.py",
        "num_workers": 1,
        "command": os.environ["PI_COMMAND"],
        "model": "fixture/local",
        "session_root": str(root / "sessions"),
        "timeout": 30,
        "concurrency": 2,
        "models_config": {
            "providers": {
                "fixture": {
                    "baseUrl": f"http://127.0.0.1:{backend.server_port}/v1",
                    "api": "openai-completions",
                    "apiKey": "unused",
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                        "supportsUsageInStreaming": False,
                        "maxTokensField": "max_tokens",
                        "supportsStore": False,
                        "supportsStrictMode": False,
                    },
                    "models": [
                        {
                            "id": "local",
                            "name": "Local deterministic fixture",
                            "reasoning": False,
                            "input": ["text"],
                            "contextWindow": 16384,
                            "maxTokens": 1024,
                        }
                    ],
                }
            }
        },
    }
    path = root / f"config-{address}.json"
    path.write_text(json.dumps(config))
    log = root / f"server-{address}.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, "-m", "dd_gym_pi_sessions.app", "--config", str(path)],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    endpoint = f"http://127.0.0.1:{address}"
    try:
        for _ in range(300):
            if process.poll() is not None:
                raise AssertionError(log.read_text())
            try:
                if httpx.get(endpoint + "/stats", timeout=0.2, trust_env=False).status_code == 200:
                    return process, endpoint
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise AssertionError("Adapter readiness timed out")
    except BaseException:
        stop_adapter(process)
        raise


def stop_adapter(process):
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise


@pytest.mark.skipif(not os.getenv("PI_COMMAND"), reason="Set PI_COMMAND to the installed Pi 1.2.0 command")
def test_real_pi_and_dd_survive_server_restart(tmp_path):
    backend = ThreadingHTTPServer(("127.0.0.1", 0), ModelFixture)
    thread = Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    first = {}
    pids = []
    try:
        for phase in (1, 2):
            process, endpoint = start_adapter(tmp_path, backend)
            pids.append(process.pid)
            try:
                if os.getenv("DD_PYTHON"):
                    subprocess.run(
                        [
                            os.environ["DD_PYTHON"],
                            str(Path(__file__).with_name("dd_flow.py")),
                            endpoint,
                            str(phase),
                            str(tmp_path),
                        ],
                        check=True,
                        timeout=90,
                    )
                else:
                    for token in ("alpha", "beta"):
                        text = (
                            f"Create solution.py with TOKEN = '{token}' and VALUE = 1"
                            if phase == 1
                            else "For the second turn set VALUE = 2 and preserve TOKEN."
                        )
                        payload = {"user": f"run-{token}", "metadata": {"turn": str(phase)}, "input": text}
                        response = httpx.post(endpoint + "/v1/responses", json=payload, timeout=45, trust_env=False)
                        assert response.status_code == 200, response.text
                        assert (
                            httpx.post(endpoint + "/v1/responses", json=payload, timeout=10, trust_env=False).json()
                            == response.json()
                        )
                assert httpx.get(endpoint + "/stats", trust_env=False).json()["active_pi_processes"] == 0
                for token in ("alpha", "beta"):
                    directory = tmp_path / "sessions" / f"run-{token}"
                    native = json.loads((directory / f"turn-{phase}.json").read_text())["native"]
                    assert (directory / "workspace/solution.py").read_text() == f"TOKEN = '{token}'\nVALUE = {phase}\n"
                    if phase == 1:
                        first[token] = native
                    else:
                        assert native["restored_messages"] == first[token]["messages"]
                        assert native["state"]["sessionId"] == first[token]["state"]["sessionId"]
                        assert native["pid"] != first[token]["pid"]
            finally:
                stop_adapter(process)
        assert pids[0] != pids[1]
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join()


@pytest.mark.skipif(not (os.getenv("PI_COMMAND") and os.getenv("DD_PYTHON")), reason="Set PI_COMMAND and DD_PYTHON")
def test_dd_timeout_skips_reaction_and_saved_stage_followup(tmp_path):
    backend = ThreadingHTTPServer(("127.0.0.1", 0), ModelFixture)
    backend.delay = 0.3
    thread = Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    try:
        process, endpoint = start_adapter(tmp_path, backend)
        try:
            subprocess.run(
                [os.environ["DD_PYTHON"], str(Path(__file__).with_name("dd_timeout.py")), endpoint, str(tmp_path)],
                check=True,
                timeout=90,
            )
            directory = tmp_path / "sessions/run-alpha"
            for _ in range(300):
                if (directory / "turn-1.json").is_file():
                    break
                time.sleep(0.1)
            assert (directory / "turn-1.json").is_file()
            assert not (directory / "turn-2.json").exists()
            assert not list(directory.glob("turn-2-*.rpc.jsonl"))
            assert (directory / "workspace/solution.py").read_text() == "TOKEN = 'alpha'\nVALUE = 1\n"
            assert httpx.get(endpoint + "/stats", trust_env=False).json()["active_pi_processes"] == 0
        finally:
            stop_adapter(process)
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join()
