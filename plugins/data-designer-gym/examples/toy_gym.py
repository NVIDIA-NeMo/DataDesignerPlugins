# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded demo services using real Gym SimpleAgent and resource implementations."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sqlite3
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    global_aiohttp_client_exit,
    set_global_aiohttp_client,
)
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from resources_servers.example_multi_step.app import (
    ExampleMultiStepResourcesServer,
    ExampleMultiStepResourcesServerConfig,
)
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class SQLRequest(BaseModel):
    database: str
    sql: str


class SeedRequest(BaseModel):
    database: str


class ToySQLResources(ExampleMultiStepResourcesServer):
    """Demo-only /execute extension. This route is not an upstream Gym contract."""

    databases: dict[str, Path]
    sessions: dict[str, str] = Field(default_factory=dict)
    initialized: int = 0
    cleaned: int = 0

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/execute")(self.execute)
        app.post("/query_database")(self.query_database)
        app.post("/cleanup")(self.cleanup)
        app.get("/stats")(self.stats)
        return app

    async def seed_session(self, request: Request, body: SeedRequest) -> dict:
        if body.database not in self.databases:
            raise ValueError("unknown demo database")
        self.sessions[request.session[SESSION_ID_KEY]] = body.database
        self.initialized += 1
        return {}

    async def cleanup(self, request: Request) -> dict:
        if self.sessions.pop(request.session[SESSION_ID_KEY], None) is not None:
            self.cleaned += 1
        return {}

    async def stats(self) -> dict:
        return {"active_sessions": len(self.sessions), "initialized": self.initialized, "cleaned": self.cleaned}

    async def query_database(self, request: Request, body: SQLRequest) -> dict:
        if self.sessions.get(request.session[SESSION_ID_KEY]) != body.database:
            raise ValueError("database does not match initialized session")
        return await self.execute(body)

    async def execute(self, body: SQLRequest) -> dict:
        path = self.databases.get(body.database)
        if path is None:
            raise ValueError("unknown demo database")
        try:
            with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=1) as connection:
                connection.set_authorizer(read_only_action)
                connection.set_progress_handler(interrupt_long_query, 100_000)
                cursor = connection.execute(body.sql)
                return {
                    "execution_status": "success",
                    "columns": [field[0] for field in cursor.description or []],
                    "rows": cursor.fetchmany(10),
                }
        except sqlite3.Error as exc:
            return {"execution_status": "sql_error", "error": str(exc)}

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        raise NotImplementedError("The toy execution service has no semantic verifier or gold SQL")


def read_only_action(action, arg1, arg2, database, trigger) -> int:
    return (
        sqlite3.SQLITE_OK
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION)
        else sqlite3.SQLITE_DENY
    )


def interrupt_long_query() -> int:
    return 1


def toy_databases(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    databases = {}
    for name, count in (("original", 3), ("variant", 4)):
        path = (root / f"{name}.db").resolve()
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT)")
            connection.execute("DELETE FROM items")
            connection.executemany(
                "INSERT INTO items VALUES (?, ?)", [(index, f"item-{index}") for index in range(count)]
            )
        databases[name] = path
    return databases


class DemoServer(uvicorn.Server):
    def capture_signals(self):
        return nullcontext()


def stop_servers(servers) -> None:
    for server in servers:
        server.should_exit = True


async def serve(args) -> None:
    host = "127.0.0.1"
    configs = OmegaConf.create(
        {
            "toy": {
                "resources_servers": {"example_multi_step": {"host": host, "port": args.port, "entrypoint": "app.py"}}
            },
            "model": {
                "responses_api_models": {"openai_model": {"host": host, "port": args.port + 1, "entrypoint": "app.py"}}
            },
        }
    )
    client = ServerClient(head_server_config=BaseServerConfig(host=host, port=args.port), global_config_dict=configs)
    resources = ToySQLResources(
        config=ExampleMultiStepResourcesServerConfig(
            name="toy", host=host, port=args.port, entrypoint="app.py", num_workers=1
        ),
        server_client=client,
        databases=toy_databases(args.database_dir),
    )
    model = SimpleModelServer(
        config=SimpleModelServerConfig(
            name="model",
            host=host,
            port=args.port + 1,
            entrypoint="app.py",
            num_workers=1,
            openai_base_url=args.model_endpoint,
            openai_api_key=os.environ[args.api_key_env],
            openai_model=args.model,
        ),
        server_client=client,
    )
    if args.model_adapter == "chat":
        model = VLLMModel(
            config=VLLMModelConfig(
                name="model",
                host=host,
                port=args.port + 1,
                entrypoint="app.py",
                num_workers=1,
                base_url=args.model_endpoint,
                api_key=os.environ[args.api_key_env],
                model=args.model,
                return_token_id_information=False,
                uses_reasoning_parser=False,
            ),
            server_client=client,
        )
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            name="agent",
            host=host,
            port=args.port + 2,
            entrypoint="app.py",
            num_workers=1,
            max_steps=4,
            resources_server={"type": "resources_servers", "name": "toy"},
            model_server={"type": "responses_api_models", "name": "model"},
        ),
        server_client=client,
    )
    http_client = set_global_aiohttp_client(
        GlobalAIOHTTPAsyncClientConfig(global_aiohttp_connector_limit=8, global_aiohttp_connector_limit_per_host=4)
    )
    try:
        servers = [
            DemoServer(
                uvicorn.Config(
                    server.setup_webserver(), host=host, port=args.port + index, workers=1, log_level="warning"
                )
            )
            for index, server in enumerate((resources, model, agent))
        ]
        for signum in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(signum, partial(stop_servers, servers))
        await asyncio.gather(*(server.serve() for server in servers))
    finally:
        await http_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--model-adapter", choices=("responses", "chat"), default="responses")
    try:
        asyncio.run(serve(parser.parse_args()))
    finally:
        global_aiohttp_client_exit()


if __name__ == "__main__":
    main()
