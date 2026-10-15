# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from functools import partial

import aiohttp
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from data_designer_sandbox_piston.client import (
    AdaptiveSlotController,
    SandboxOutput,
    execute_code_in_sandbox,
    serialize_sandbox_output,
)

ALLOWED_RESULT_FIELDS = frozenset(SandboxOutput.model_fields)


def env_required(env: Mapping[str, str], key: str) -> str:
    """Read a required environment variable.

    Args:
        env: Environment mapping.
        key: Environment variable name.

    Returns:
        Non-empty environment value.

    Raises:
        RuntimeError: If the variable is missing or empty.
    """
    raw = env.get(key)
    if raw is None or raw == "":
        raise RuntimeError(f"{key} environment variable is required but not set.")
    return raw


def env_int(env: Mapping[str, str], key: str) -> int:
    """Read a required positive integer environment variable."""
    value = int(env_required(env, key))
    if value <= 0:
        raise RuntimeError(f"{key} must be positive")
    return value


def env_result_fields(env: Mapping[str, str]) -> tuple[str, ...]:
    """Read and validate selected MCP result fields."""
    raw = env_required(env, "SANDBOX_RESULT_FIELDS")
    fields = tuple(field.strip() for field in raw.split(",") if field.strip())
    if not fields:
        raise RuntimeError("SANDBOX_RESULT_FIELDS must include at least one field.")
    invalid = sorted(set(fields) - ALLOWED_RESULT_FIELDS)
    if invalid:
        allowed = ", ".join(sorted(ALLOWED_RESULT_FIELDS))
        raise RuntimeError(f"Invalid SANDBOX_RESULT_FIELDS value(s): {', '.join(invalid)}. Allowed fields: {allowed}.")
    return fields


class SandboxMCPState:
    """Runtime state for the Piston sandbox MCP server."""

    def __init__(
        self,
        sandbox_url: str,
        language: str,
        version: str,
        run_timeout: int,
        run_cpu_time: int,
        tool_description: str,
        result_fields: tuple[str, ...],
    ) -> None:
        self.sandbox_url = sandbox_url
        self.language = language
        self.version = version
        self.run_timeout = run_timeout
        self.run_cpu_time = run_cpu_time
        self.tool_description = tool_description
        self.result_fields = result_fields
        self._slots = AdaptiveSlotController()
        self._session: aiohttp.ClientSession | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SandboxMCPState:
        """Create state from environment variables."""
        resolved_env = env or os.environ
        return cls(
            sandbox_url=env_required(resolved_env, "SANDBOX_URL"),
            language=env_required(resolved_env, "SANDBOX_LANGUAGE"),
            version=env_required(resolved_env, "SANDBOX_VERSION"),
            run_timeout=env_int(resolved_env, "SANDBOX_RUN_TIMEOUT"),
            run_cpu_time=env_int(resolved_env, "SANDBOX_RUN_CPU_TIME"),
            tool_description=env_required(resolved_env, "SANDBOX_TOOL_DESCRIPTION"),
            result_fields=env_result_fields(resolved_env),
        )

    async def get_session(self) -> aiohttp.ClientSession:
        """Return a shared HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def execute(self, code: str) -> SandboxOutput:
        """Execute one stateless code block."""
        session = await self.get_session()
        return await execute_code_in_sandbox(
            session=session,
            slots=self._slots,
            sandbox_url=self.sandbox_url,
            code=code,
            language=self.language,
            version=self.version,
            run_timeout=self.run_timeout,
            run_cpu_time=self.run_cpu_time,
            row_id="mcp",
        )

    def serialize_result(self, result: SandboxOutput) -> str:
        """Serialize the configured result fields."""
        return serialize_sandbox_output(result, self.result_fields)

    async def cleanup(self) -> None:
        """Close the shared HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


async def list_tools_for_state(state: SandboxMCPState) -> list[Tool]:
    """List tools exposed by the sandbox MCP server."""
    return [
        Tool(
            name="run_code",
            description=state.tool_description,
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The source code to execute.",
                    },
                },
                "required": ["code"],
            },
        )
    ]


async def call_tool_for_state(state: SandboxMCPState, name: str, arguments: dict) -> list[TextContent]:
    """Handle an MCP tool call."""
    if name != "run_code":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    code = arguments.get("code", "")
    if not isinstance(code, str) or not code.strip():
        empty = SandboxOutput(
            stdout="",
            stderr="No code provided",
            exit_code=-2,
            message="No code provided",
        )
        return [TextContent(type="text", text=state.serialize_result(empty))]

    result = await state.execute(code)
    return [TextContent(type="text", text=state.serialize_result(result))]


def create_server(env: Mapping[str, str] | None = None) -> tuple[Server, SandboxMCPState]:
    """Create and configure the MCP server.

    Args:
        env: Optional environment mapping for tests or custom launchers.

    Returns:
        The configured server and runtime state.
    """
    state = SandboxMCPState.from_env(env)
    server = Server("data-designer-sandbox-piston")
    server.list_tools()(partial(list_tools_for_state, state))
    server.call_tool()(partial(call_tool_for_state, state))
    return server, state


async def main() -> None:
    """Run the stdio MCP server."""
    server, state = create_server()
    try:
        async with stdio_server() as (read_stream, write_stream):
            init_options = server.create_initialization_options()
            await server.run(read_stream, write_stream, init_options)
    finally:
        await state.cleanup()


def entrypoint() -> None:
    """Console script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()
