# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from data_designer.config.base import SingleColumnConfig
from data_designer.config.mcp import LocalStdioMCPProvider, ToolConfig
from pydantic import BaseModel, Field, field_validator, model_validator

SandboxMCPResultField = Literal[
    "stdout",
    "stderr",
    "output",
    "exit_code",
    "signal",
    "message",
    "status",
    "cpu_time",
    "wall_time",
    "memory",
]

DEFAULT_SANDBOX_LANGUAGE = "python"
DEFAULT_SANDBOX_MCP_RESULT_FIELDS: tuple[SandboxMCPResultField, ...] = (
    "stdout",
    "stderr",
    "exit_code",
)
DEFAULT_SANDBOX_RUN_TIMEOUT_MS = 3000
SANDBOX_MCP_MODULE = "data_designer_sandbox_piston.mcp_server"
SANDBOX_MCP_LANGUAGE_DISPLAY_NAMES: dict[str, str] = {"gcc": "GCC"}


def has_python_packages(value: Sequence[str] | None) -> bool:
    """Return whether a package list declares at least one Python package."""
    return bool(value)


def validate_python_runtime_requirements(
    language: str,
    version: str,
    python_packages: Sequence[str] | None,
) -> None:
    """Validate Python package metadata for a prebuilt Piston runtime.

    Args:
        language: Piston runtime language.
        version: Piston runtime version selector.
        python_packages: Optional Python package requirements.

    Raises:
        ValueError: If packages are declared for a non-Python runtime or without
            an explicit custom runtime version.
    """
    if python_packages is not None and language != "python":
        raise ValueError(f"python_packages can only be specified when language='python', but language='{language}'")
    if has_python_packages(python_packages) and version == "*":
        raise ValueError(
            "python_packages requires an explicit Piston runtime version. "
            "Build or provide a custom Python runtime that includes those packages, "
            "then set version to that runtime's version."
        )


def validate_sandbox_url(value: str | None) -> str | None:
    """Validate and normalize a Piston sandbox endpoint URL.

    Args:
        value: Optional HTTP or HTTPS sandbox base URL.

    Returns:
        The URL without a trailing slash, or ``None``.

    Raises:
        ValueError: If a non-empty URL does not use HTTP or HTTPS.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if not re.match(r"^https?://", stripped):
        raise ValueError("sandbox_url must be an HTTP or HTTPS URL")
    return stripped.rstrip("/")


def format_natural_list(values: Sequence[str]) -> str:
    """Format a short list for LLM-facing descriptions.

    Args:
        values: Values to join.

    Returns:
        A comma-separated phrase with a final ``and`` when useful.
    """
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:  # noqa: PLR2004
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def default_sandbox_mcp_tool_description(
    language: str,
    result_fields: Sequence[str] = DEFAULT_SANDBOX_MCP_RESULT_FIELDS,
    python_packages: Sequence[str] | None = None,
) -> str:
    """Build the default LLM-facing description for the sandbox MCP tool.

    Args:
        language: Piston language runtime identifier.
        result_fields: Sandbox output fields included in tool responses.
        python_packages: Optional Python package names available in the runtime.

    Returns:
        A concise tool description.
    """
    language_label = SANDBOX_MCP_LANGUAGE_DISPLAY_NAMES.get(language, language.title())
    package_clause = ""
    if python_packages:
        package_clause = f" with installed Python packages: {format_natural_list(python_packages)}"
    result_field_list = format_natural_list(result_fields)
    return f"Execute {language_label} code in a sandboxed environment{package_clause}. Returns {result_field_list}."


class CodeSandboxColumnConfig(SingleColumnConfig):
    """Configuration for the Piston-backed ``code-sandbox`` column generator.

    Attributes:
        target_column: Existing column containing source code to execute.
        language: Piston runtime language, such as ``python`` or ``gcc``.
        version: Piston runtime version selector.
        python_packages: Optional Python packages expected to be available in a
            prebuilt custom Python runtime. The plugin does not build runtimes
            itself, so non-empty packages require an explicit ``version``.
        stdin: Text passed to the program standard input.
        args: Command-line arguments passed to the program.
        compile_timeout: Compile wall-time limit in milliseconds.
        run_timeout: Run wall-time limit in milliseconds.
        compile_cpu_time: Compile CPU-time limit in milliseconds.
        run_cpu_time: Run CPU-time limit in milliseconds.
        sandbox_url: HTTP(S) base URL for a local or remote Piston API.
    """

    column_type: Literal["code-sandbox"] = "code-sandbox"

    target_column: str = Field(description="Column containing source code to execute.")
    language: str = Field(description="Piston runtime language, such as 'python' or 'gcc'.")
    version: str = Field(default="*", description="Piston runtime version selector.")
    python_packages: list[str] | None = Field(
        default=None,
        description=(
            "Optional Python package requirements. Only valid when language='python'. "
            "Non-empty package lists require an explicit custom Piston runtime version; "
            "deployments must provide or build that runtime before execution."
        ),
    )
    stdin: str = Field(default="", description="Text passed to standard input.")
    args: list[str] = Field(default_factory=list, description="Command-line arguments passed to the program.")
    compile_timeout: int = Field(default=10000, gt=0, description="Compile wall-time limit in milliseconds.")
    run_timeout: int = Field(
        default=DEFAULT_SANDBOX_RUN_TIMEOUT_MS,
        gt=0,
        description="Run wall-time limit in milliseconds.",
    )
    compile_cpu_time: int = Field(default=3000, gt=0, description="Compile CPU-time limit in milliseconds.")
    run_cpu_time: int = Field(default=3000, gt=0, description="Run CPU-time limit in milliseconds.")
    sandbox_url: str | None = Field(
        default=None,
        description=(
            "Piston API base URL, such as 'http://localhost:2000'. Set this directly for local or remote deployments."
        ),
    )

    @staticmethod
    def get_column_emoji() -> str:
        """Return a display marker for Data Designer UIs."""
        return "[]"

    @property
    def required_columns(self) -> list[str]:
        """Return the source-code column dependency."""
        return [self.target_column] if self.target_column else []

    @property
    def side_effect_columns(self) -> list[str]:
        """Return side-effect columns created by this generator."""
        return []

    @field_validator("sandbox_url")
    @classmethod
    def _validate_sandbox_url(cls, value: str | None) -> str | None:
        return validate_sandbox_url(value)

    @field_validator("target_column", "language")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def _validate_python_packages(self) -> CodeSandboxColumnConfig:
        validate_python_runtime_requirements(self.language, self.version, self.python_packages)
        return self


class SandboxMCPConfig(BaseModel):
    """Configuration for exposing a Piston sandbox as a stdio MCP provider.

    Attributes:
        name: MCP provider name referenced by Data Designer ``ToolConfig``.
        sandbox_url: Optional HTTP(S) Piston endpoint. If omitted, the launched
            MCP process must inherit ``SANDBOX_URL`` from its environment.
        language: Piston runtime language used by the ``run_code`` tool.
        version: Piston runtime version selector.
        python_packages: Optional Python packages expected to be available in a
            prebuilt custom Python runtime. Non-empty packages require an
            explicit ``version``.
        result_fields: ``SandboxOutput`` fields returned by the MCP tool.
        run_timeout: Run wall-time limit in milliseconds.
        run_cpu_time: Run CPU-time limit in milliseconds.
        tool_description: Optional LLM-facing description for the ``run_code`` tool.
    """

    name: str = Field(default="sandbox", description="MCP provider name.")
    sandbox_url: str | None = Field(default=None, description="Optional Piston API base URL.")
    language: str = Field(default=DEFAULT_SANDBOX_LANGUAGE, description="Piston runtime language.")
    version: str = Field(default="*", description="Piston runtime version selector.")
    python_packages: list[str] | None = Field(
        default=None,
        description="Optional Python package requirements for a prebuilt custom Python runtime.",
    )
    result_fields: list[SandboxMCPResultField] = Field(
        default_factory=lambda: list(DEFAULT_SANDBOX_MCP_RESULT_FIELDS),
        description="Sandbox output fields included in MCP responses.",
    )
    run_timeout: int = Field(
        default=DEFAULT_SANDBOX_RUN_TIMEOUT_MS,
        gt=0,
        description="Run wall-time limit in milliseconds.",
    )
    run_cpu_time: int = Field(default=3000, gt=0, description="Run CPU-time limit in milliseconds.")
    tool_description: str | None = Field(default=None, description="LLM-facing MCP tool description.")

    @field_validator("sandbox_url")
    @classmethod
    def _validate_sandbox_url(cls, value: str | None) -> str | None:
        return validate_sandbox_url(value)

    @field_validator("name", "language")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must be a non-empty string")
        return stripped

    @field_validator("result_fields")
    @classmethod
    def _validate_result_fields(
        cls,
        value: list[SandboxMCPResultField],
    ) -> list[SandboxMCPResultField]:
        if not value:
            raise ValueError("result_fields must include at least one field")
        if len(set(value)) != len(value):
            raise ValueError("result_fields must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_and_set_defaults(self) -> SandboxMCPConfig:
        validate_python_runtime_requirements(self.language, self.version, self.python_packages)
        if self.tool_description is None:
            self.tool_description = default_sandbox_mcp_tool_description(
                self.language,
                result_fields=self.result_fields,
                python_packages=self.python_packages,
            )
        return self

    def to_provider(self) -> LocalStdioMCPProvider:
        """Create a Data Designer stdio MCP provider for this sandbox."""
        return create_sandbox_mcp_provider(self)

    def to_tool_config(self) -> ToolConfig:
        """Create a Data Designer tool configuration using this provider."""
        return ToolConfig(tool_alias=self.name, providers=[self.name])


def create_sandbox_mcp_provider(config: SandboxMCPConfig | None = None) -> LocalStdioMCPProvider:
    """Create a Data Designer ``LocalStdioMCPProvider`` for the sandbox MCP server.

    Args:
        config: Optional MCP configuration. Defaults are used when omitted.

    Returns:
        A provider that launches ``data_designer_sandbox_piston.mcp_server``.
    """
    resolved = config or SandboxMCPConfig()
    env: dict[str, str] = {
        "SANDBOX_LANGUAGE": resolved.language,
        "SANDBOX_VERSION": resolved.version,
        "SANDBOX_RUN_TIMEOUT": str(resolved.run_timeout),
        "SANDBOX_RUN_CPU_TIME": str(resolved.run_cpu_time),
        "SANDBOX_TOOL_DESCRIPTION": resolved.tool_description or "",
        "SANDBOX_RESULT_FIELDS": ",".join(resolved.result_fields),
    }
    if resolved.sandbox_url:
        env["SANDBOX_URL"] = resolved.sandbox_url
    if resolved.python_packages:
        env["SANDBOX_PYTHON_PACKAGES"] = ",".join(resolved.python_packages)

    return LocalStdioMCPProvider(
        name=resolved.name,
        command="python",
        args=["-m", SANDBOX_MCP_MODULE],
        env=env,
    )
