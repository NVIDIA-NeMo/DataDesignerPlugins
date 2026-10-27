# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Literal
from uuid import uuid4

import uvicorn
from fastapi import Body, HTTPException, Request
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import ObservationGap
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from omegaconf import OmegaConf
from pydantic import Field, PrivateAttr, model_validator
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig, parse_pi_events

from dd_gym_pi_sessions.rpc import PiRPC
from dd_gym_pi_sessions.store import SessionStore, Turn, digest, save_json

CURRENT_TURN: ContextVar[Turn | None] = ContextVar("dd_pi_turn", default=None)


class PersistentPiConfig(PiAgentConfig):
    """Local two-turn example; Gym and Pi remain in a separate runtime."""

    session_root: str
    pi_version: Literal["1.2.0"] = "1.2.0"
    auto_compaction: bool = False
    max_turns: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_persistence(self) -> PersistentPiConfig:
        if self.num_workers not in (None, 1):
            raise ValueError("Persistent Pi requires one server worker")
        if self.sandbox_provider or self.extra_args or self.mcp_servers:
            raise ValueError("This local file-tool example does not accept sandbox, extra_args, or MCP overrides")
        if not self.command_parts or self.timeout <= 0 or self.concurrency < 1:
            raise ValueError("Pi command, positive timeout, and positive concurrency are required")
        return self


class PersistentPiAgent(PiAgent):
    """Extend upstream Pi response mapping and observations with durable native sessions."""

    config: PersistentPiConfig
    _store: SessionStore = PrivateAttr()
    _requests: set[asyncio.Task] = PrivateAttr(default_factory=set)
    _processes: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        self._store = SessionStore(Path(self.config.session_root))

    def setup_webserver(self):
        app = super().setup_webserver()
        app.router.lifespan_context = self.lifespan
        app.get("/stats")(self.stats)
        return app

    @asynccontextmanager
    async def lifespan(self, app):
        version = await asyncio.create_subprocess_exec(
            *self.config.command_parts, "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, _ = await asyncio.wait_for(version.communicate(), 10)
        except BaseException:
            if version.returncode is None:
                version.kill()
            await version.wait()
            raise
        if version.returncode != 0 or stdout.decode().strip() != self.config.pi_version:
            raise ValueError(f"Install Pi {self.config.pi_version} before starting this adapter")
        try:
            yield
        finally:
            requests = list(self._requests)
            for task in requests:
                task.cancel()
            await asyncio.gather(*requests, return_exceptions=True)

    def fingerprint(self) -> str:
        configuration = self.config.model_dump(
            mode="json",
            exclude={
                "name",
                "host",
                "port",
                "entrypoint",
                "num_workers",
                "workspace_root",
                "session_root",
                "concurrency",
            },
        )
        return digest({"agent": configuration, "models": self._build_models_config()})

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """Accept the existing DD user/metadata.turn convention, returning committed retries unchanged."""
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "Use conversation IDs, not native single-activation agent sessions")
        metadata = body.metadata or {}
        turn_text = metadata.get("turn", "")
        if not isinstance(body.user, str) or not turn_text.isdecimal() or set(metadata) != {"turn"}:
            raise HTTPException(422, "Expected user=conversation ID and metadata.turn")
        number = int(turn_text)
        if not 1 <= number <= self.config.max_turns:
            raise HTTPException(422, "Turn is outside the configured conversation length")
        validated = self._validate_request(body.model_copy(update={"user": None, "metadata": None}))
        task = asyncio.current_task()
        self._requests.add(task)
        try:
            async with self.sem:
                with self._store.turn(body.user, number, validated.model_dump(mode="json"), self.fingerprint()) as turn:
                    if turn.receipt is not None:
                        return NeMoGymResponse.model_validate(turn.receipt["response"])
                    token = CURRENT_TURN.set(turn)
                    try:
                        episode = await self._create_episode(
                            validated, rollout_id=f"{body.user}-t{number}", collect_observations=True
                        )
                    finally:
                        CURRENT_TURN.reset(token)
                    if number > 1:
                        episode.observations.gaps.append(
                            ObservationGap(
                                code="conversation_prefix_in_native_receipt",
                                detail=str(turn.directory / f"turn-{number}.json"),
                            )
                        )
                    response = episode.response.model_copy(
                        update={
                            "status": "completed",
                            "metadata": {
                                "conversation_id": body.user,
                                "turn": turn_text,
                                "pi_session_id": turn.native["state"]["sessionId"],
                                "workspace": str(turn.workspace),
                            },
                            "_ng_agent_observations": episode.observations.model_dump(mode="json"),
                        }
                    )
                    self._store.commit(
                        turn,
                        response.model_dump(mode="json"),
                        episode.observations.model_dump(mode="json"),
                        final=number == self.config.max_turns,
                    )
                    return response
        finally:
            self._requests.discard(task)

    async def _run_pi(self, instruction, system_prompt, *, rollout_id=None, collect_observations=True):
        turn = CURRENT_TURN.get()
        if turn is None:
            raise RuntimeError("Pi execution requires a durable conversation turn")
        home = turn.directory / "home"
        agent_dir = home / ".pi" / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        save_json(agent_dir / "models.json", self._build_models_config(rollout_id))
        save_json(agent_dir / "settings.json", self._build_settings_config())
        environment = self._env(home) | {"PI_CODING_AGENT_DIR": str(agent_dir)}
        command = [
            *self.config.command_parts,
            "--mode",
            "rpc",
            "--tools",
            "read,write,edit",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
        ]
        if turn.manifest["turn"]:
            command += ["--session", turn.manifest["session_file"]]
        else:
            command += ["--session-dir", str(turn.directory / "sessions"), "--session-id", str(uuid4())]
        provider, _, model = self._effective_model().partition("/")
        command += ["--provider", provider, "--model", model] if model else ["--model", provider]
        if self.config.thinking:
            command += ["--thinking", self.config.thinking]
        if system_prompt:
            command += ["--append-system-prompt", system_prompt]
        trace_path = turn.directory / f"turn-{turn.number}-{uuid4().hex}.rpc.jsonl"
        rpc = await PiRPC.start(command, turn.workspace, environment, trace_path)
        self._processes[turn.directory.name] = rpc.process.pid
        failed = False
        try:
            async with asyncio.timeout(self.config.timeout):
                await rpc.command("set_auto_retry", enabled=False)
                await rpc.command("set_auto_compaction", enabled=False)
                previous = (await rpc.command("get_messages"))["messages"]
                initial = await rpc.command("get_state")
                if turn.manifest["turn"]:
                    receipt = json.loads((turn.directory / f"turn-{turn.manifest['turn']}.json").read_text())
                    if previous != receipt["native"]["messages"] or initial["sessionId"] != turn.manifest["session_id"]:
                        raise RuntimeError("Pi did not restore the committed native conversation")
                messages, state = await rpc.prompt(instruction, previous)
                turn.native = {
                    "state": state,
                    "messages": messages,
                    "restored_messages": previous,
                    "pid": rpc.process.pid,
                    "trace": str(trace_path),
                }
        except BaseException:
            failed = True
            raise
        finally:
            try:
                await rpc.close(abort=failed)
            finally:
                self._processes.pop(turn.directory.name, None)
        output, usage = parse_pi_events("\n".join(json.dumps(event) for _, event in rpc.events))
        return output, usage, self.config.model, rpc.events

    async def seed_agent_session(self, request: Request, body: dict = Body()):
        raise HTTPException(422, "This adapter owns durable conversations through /v1/responses")

    async def run(self, request: Request, body: dict = Body()):
        raise HTTPException(422, "This answer-only adapter uses /v1/responses; verification belongs to the workflow")

    async def stats(self) -> dict:
        return {"active_pi_processes": len(self._processes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gym-config", type=Path)
    args = parser.parse_args()
    config = PersistentPiConfig.model_validate(json.loads(args.config.read_text()))
    agent = PersistentPiAgent(
        config=config,
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host=config.host, port=config.port),
            global_config_dict=OmegaConf.load(args.gym_config) if args.gym_config else OmegaConf.create({}),
        ),
    )
    uvicorn.run(agent.setup_webserver(), host=config.host, port=config.port, workers=1)


if __name__ == "__main__":
    main()
