# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

import aiohttp
import pandas as pd
import pytest
from data_designer.engine.testing.utils import assert_valid_plugin
from pydantic import ValidationError

from data_designer_sandbox_piston.client import (
    SANDBOX_MAX_RETRIES,
    AdaptiveSlotController,
    PistonStatus,
    SandboxOutput,
    SandboxStats,
    execute_code_in_sandbox,
    parse_execute_response,
    serialize_sandbox_output,
)
from data_designer_sandbox_piston.config import (
    CodeSandboxColumnConfig,
    SandboxMCPConfig,
    create_sandbox_mcp_provider,
)
from data_designer_sandbox_piston.impl import CodeSandboxColumnGenerator
from data_designer_sandbox_piston.mcp_server import (
    SandboxMCPState,
    call_tool_for_state,
    create_server,
    list_tools_for_state,
)
from data_designer_sandbox_piston.plugin import plugin


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)


def make_generator(config: CodeSandboxColumnConfig) -> CodeSandboxColumnGenerator:
    """Create a generator instance without requiring Data Designer resources."""
    generator = CodeSandboxColumnGenerator.__new__(CodeSandboxColumnGenerator)
    generator._config = config
    return generator


class RateLimitedResponse:
    """Minimal async response object for rate-limit retry tests."""

    status = 429
    request_info = None
    history: tuple[object, ...] = ()

    async def __aenter__(self) -> RateLimitedResponse:
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        return None

    async def text(self) -> str:
        return "too many requests"


class RateLimitedSession:
    """Minimal session that always returns HTTP 429."""

    def __init__(self) -> None:
        self.calls = 0

    def post(self, *_args: object, **_kwargs: object) -> RateLimitedResponse:
        self.calls += 1
        return RateLimitedResponse()


async def no_async_sleep(_delay: float) -> None:
    """No-op replacement for retry sleeps in tests."""
    return None


class TestCodeSandboxColumnConfig:
    def test_required_columns(self) -> None:
        config = CodeSandboxColumnConfig(
            name="result",
            target_column="code",
            language="python",
        )
        assert config.column_type == "code-sandbox"
        assert config.required_columns == ["code"]
        assert config.side_effect_columns == []

    def test_validates_sandbox_url(self) -> None:
        config = CodeSandboxColumnConfig(
            name="result",
            target_column="code",
            language="python",
            sandbox_url="http://localhost:2000/",
        )
        assert config.sandbox_url == "http://localhost:2000"

        with pytest.raises(ValidationError, match="sandbox_url"):
            CodeSandboxColumnConfig(
                name="result",
                target_column="code",
                language="python",
                sandbox_url="localhost:2000",
            )

    def test_python_packages_require_python_language(self) -> None:
        with pytest.raises(ValidationError, match="python_packages"):
            CodeSandboxColumnConfig(
                name="result",
                target_column="code",
                language="gcc",
                python_packages=["numpy"],
            )

    def test_python_packages_require_explicit_runtime_version(self) -> None:
        with pytest.raises(ValidationError, match="explicit Piston runtime version"):
            CodeSandboxColumnConfig(
                name="result",
                target_column="code",
                language="python",
                python_packages=["numpy"],
            )

        config = CodeSandboxColumnConfig(
            name="result",
            target_column="code",
            language="python",
            version="3.12.0-ddp-numpy",
            python_packages=["numpy"],
        )
        assert config.version == "3.12.0-ddp-numpy"


class TestSandboxOutput:
    def test_parses_piston_response(self) -> None:
        output = parse_execute_response(
            {
                "run": {
                    "stdout": "42\n",
                    "stderr": "",
                    "output": "42\n",
                    "code": 0,
                    "status": None,
                    "cpu_time": 1.5,
                    "wall_time": 2.0,
                    "memory": 2048,
                }
            }
        )
        assert output.stdout == "42\n"
        assert output.exit_code == 0
        assert output.cpu_time == 1.5

    def test_status_parser(self) -> None:
        assert PistonStatus.from_string("TO") == PistonStatus.TIMEOUT
        assert PistonStatus.from_string(None) is None
        assert PistonStatus.from_string("unknown") is None

    def test_serializes_selected_fields(self) -> None:
        output = SandboxOutput(stdout="ok", stderr="", exit_code=0, memory=128)
        assert json.loads(serialize_sandbox_output(output, ["stdout", "exit_code"])) == {
            "stdout": "ok",
            "exit_code": 0,
        }

    def test_rate_limits_stop_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("data_designer_sandbox_piston.client.asyncio.sleep", no_async_sleep)
        session = RateLimitedSession()
        stats = SandboxStats()

        with pytest.raises(aiohttp.ClientResponseError):
            asyncio.run(
                execute_code_in_sandbox(
                    session=session,
                    slots=AdaptiveSlotController(initial_slots=1, min_slots=1, max_slots=1),
                    sandbox_url="http://localhost:2000",
                    code="print(1)",
                    language="python",
                    stats=stats,
                )
            )

        assert session.calls == SANDBOX_MAX_RETRIES
        assert stats.rate_limit_count == SANDBOX_MAX_RETRIES


class TestCodeSandboxColumnGenerator:
    def test_requires_sandbox_url(self) -> None:
        generator = make_generator(CodeSandboxColumnConfig(name="result", target_column="code", language="python"))
        with pytest.raises(ValueError, match="sandbox_url"):
            generator.generate(pd.DataFrame({"code": ["print(1)"]}))

    def test_requires_target_column(self) -> None:
        generator = make_generator(
            CodeSandboxColumnConfig(
                name="result",
                target_column="code",
                language="python",
                sandbox_url="http://localhost:2000",
            )
        )
        with pytest.raises(ValueError, match="Target column"):
            generator.generate(pd.DataFrame({"other": ["print(1)"]}))

    def test_generates_structured_outputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_execute_code_in_sandbox(**kwargs: object) -> SandboxOutput:
            code = kwargs["code"]
            return SandboxOutput(stdout=f"ran {code}", stderr="", output=f"ran {code}", exit_code=0)

        monkeypatch.setattr(
            "data_designer_sandbox_piston.impl.execute_code_in_sandbox",
            fake_execute_code_in_sandbox,
        )

        generator = make_generator(
            CodeSandboxColumnConfig(
                name="result",
                target_column="code",
                language="python",
                sandbox_url="http://localhost:2000",
            )
        )
        result = generator.generate(pd.DataFrame({"code": ["print(1)", None, "print(2)"]}))

        assert result["result"][0]["stdout"] == "ran print(1)"
        assert result["result"][1]["exit_code"] == -2
        assert result["result"][2]["stdout"] == "ran print(2)"


class TestSandboxMCPConfig:
    def test_builds_provider_and_tool_config(self) -> None:
        config = SandboxMCPConfig(
            name="sandbox",
            sandbox_url="http://localhost:2000",
            language="python",
            version="3.12.0-ddp-numpy",
            python_packages=["numpy"],
        )
        provider = config.to_provider()
        tool_config = config.to_tool_config()

        assert provider.name == "sandbox"
        assert provider.command == "python"
        assert provider.args == ["-m", "data_designer_sandbox_piston.mcp_server"]
        assert provider.env["SANDBOX_URL"] == "http://localhost:2000"
        assert provider.env["SANDBOX_PYTHON_PACKAGES"] == "numpy"
        assert "installed Python packages: numpy" in provider.env["SANDBOX_TOOL_DESCRIPTION"]
        assert tool_config.providers == ["sandbox"]

    def test_provider_can_defer_sandbox_url_to_runtime_env(self) -> None:
        provider = create_sandbox_mcp_provider(SandboxMCPConfig())
        assert "SANDBOX_URL" not in provider.env
        assert provider.env["SANDBOX_LANGUAGE"] == "python"

    def test_validates_result_fields(self) -> None:
        with pytest.raises(ValidationError, match="result_fields"):
            SandboxMCPConfig(result_fields=[])

    def test_python_packages_require_explicit_runtime_version(self) -> None:
        with pytest.raises(ValidationError, match="explicit Piston runtime version"):
            SandboxMCPConfig(language="python", python_packages=["numpy"])


class TestMCPServer:
    def test_creates_server_from_env(self) -> None:
        server, state = create_server(
            {
                "SANDBOX_URL": "http://localhost:2000",
                "SANDBOX_LANGUAGE": "python",
                "SANDBOX_VERSION": "*",
                "SANDBOX_RUN_TIMEOUT": "3000",
                "SANDBOX_RUN_CPU_TIME": "3000",
                "SANDBOX_TOOL_DESCRIPTION": "Execute Python.",
                "SANDBOX_RESULT_FIELDS": "stdout,stderr,exit_code",
            }
        )
        assert server.name == "data-designer-sandbox-piston"
        assert state.sandbox_url == "http://localhost:2000"

    def test_lists_run_code_tool(self) -> None:
        state = SandboxMCPState.from_env(
            {
                "SANDBOX_URL": "http://localhost:2000",
                "SANDBOX_LANGUAGE": "python",
                "SANDBOX_VERSION": "*",
                "SANDBOX_RUN_TIMEOUT": "3000",
                "SANDBOX_RUN_CPU_TIME": "3000",
                "SANDBOX_TOOL_DESCRIPTION": "Execute Python.",
                "SANDBOX_RESULT_FIELDS": "stdout,exit_code",
            }
        )
        tools = asyncio.run(list_tools_for_state(state))
        assert tools[0].name == "run_code"
        assert tools[0].description == "Execute Python."

    def test_empty_code_returns_error_without_http(self) -> None:
        state = SandboxMCPState.from_env(
            {
                "SANDBOX_URL": "http://localhost:2000",
                "SANDBOX_LANGUAGE": "python",
                "SANDBOX_VERSION": "*",
                "SANDBOX_RUN_TIMEOUT": "3000",
                "SANDBOX_RUN_CPU_TIME": "3000",
                "SANDBOX_TOOL_DESCRIPTION": "Execute Python.",
                "SANDBOX_RESULT_FIELDS": "stderr,exit_code",
            }
        )
        response = asyncio.run(call_tool_for_state(state, "run_code", {"code": "  "}))
        assert json.loads(response[0].text) == {
            "stderr": "No code provided",
            "exit_code": -2,
        }
