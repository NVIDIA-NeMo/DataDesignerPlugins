# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path
from time import time
from uuid import uuid4


class PiRPC:
    """Own one Pi process until its native session is flushed and the process exits."""

    def __init__(self, process: asyncio.subprocess.Process, trace, stderr):
        self.process = process
        self.trace = trace
        self.stderr = stderr
        self.events: list[tuple[float, dict]] = []

    @classmethod
    async def start(cls, command: list[str], workspace: Path, environment: dict, trace_path: Path) -> PiRPC:
        trace = trace_path.open("x")
        stderr = trace_path.with_suffix(".stderr").open("xb")
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace,
                env=environment,
                start_new_session=True,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                limit=16 * 1024 * 1024,
            )
        except BaseException:
            trace.close()
            stderr.close()
            raise
        return cls(process, trace, stderr)

    async def read(self) -> dict:
        line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError("Pi RPC exited before completing the turn; inspect its stderr artifact")
        event = json.loads(line)
        self.events.append((time(), event))
        self.trace.write(json.dumps(event) + "\n")
        self.trace.flush()
        return event

    async def command(self, kind: str, **values) -> dict:
        request_id = uuid4().hex
        self.process.stdin.write((json.dumps({"id": request_id, "type": kind, **values}) + "\n").encode())
        await self.process.stdin.drain()
        while True:
            event = await self.read()
            if event.get("type") == "response" and event.get("id") == request_id:
                if not event.get("success"):
                    raise RuntimeError(event.get("error", "Pi command failed"))
                return event.get("data", {})

    async def prompt(self, text: str, previous: list[dict]) -> tuple[list[dict], dict]:
        accepted = await self.command("prompt", message=text)
        if accepted.get("disposition") == "handled":
            raise RuntimeError("Pi did not start an agent run")
        while True:
            event = await self.read()
            if event.get("type") == "agent_settled":
                if event.get("aborted"):
                    raise RuntimeError("Pi turn aborted")
                break
        messages = (await self.command("get_messages"))["messages"]
        state = await self.command("get_state")
        if messages[: len(previous)] != previous:
            raise RuntimeError("Pi changed the restored conversation prefix")
        new = messages[len(previous) :]
        assistants = [message for message in new if message.get("role") == "assistant"]
        if not assistants or assistants[-1].get("stopReason") != "stop":
            raise RuntimeError("Pi did not produce a completed assistant reply")
        if not any(
            block.get("type") == "text" and str(block.get("text", "")).strip()
            for block in assistants[-1].get("content", [])
        ):
            raise RuntimeError("Pi did not produce final assistant text")
        completed = {message.get("toolCallId") for message in new if message.get("role") == "toolResult"}
        if any(
            block.get("id") not in completed
            for message in assistants
            for block in message.get("content", [])
            if block.get("type") == "toolCall"
        ):
            raise RuntimeError("Pi returned an unresolved tool call")
        return messages, state

    async def close(self, *, abort: bool = False) -> None:
        """Finish bounded cleanup even if the caller is cancelled repeatedly."""
        cleanup = asyncio.create_task(self._close(abort))
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self, abort: bool) -> None:
        try:
            if self.process.returncode is None:
                self.process.stdin.close()
                if abort:
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), 5)
                except asyncio.TimeoutError:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(self.process.wait(), 5)
                    except asyncio.TimeoutError:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        await self.process.wait()
                    if not abort:
                        raise RuntimeError("Pi required forced termination; turn was not committed")
            if self.process.returncode != 0 and not abort:
                raise RuntimeError(f"Pi exited with code {self.process.returncode}; turn was not committed")
        finally:
            self.trace.close()
            self.stderr.close()
