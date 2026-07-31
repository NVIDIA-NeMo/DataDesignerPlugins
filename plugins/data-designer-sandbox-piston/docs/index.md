# data-designer-sandbox-piston

`data-designer-sandbox-piston` adds Piston-backed code execution to Data
Designer. It provides a `code-sandbox` column type for batch workflows and a
stdio MCP server that exposes the same Piston endpoint as a `run_code` tool for
tool-calling LLM columns.

The plugin is deployment-neutral: point it at any reachable Piston API URL. That
URL can be a local Docker container on macOS or Linux, a service running beside a
Data Designer worker, or a remote endpoint behind your own proxy.

## Installation

```bash
uv add data-designer data-designer-sandbox-piston
```

## Column type

Use the `code-sandbox` column type when a dataset already contains source code
that should be executed by Piston.

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Output column name. Each value is a dictionary with execution results. |
| `target_column` | Yes | Existing column containing source code. |
| `language` | Yes | Piston runtime language, such as `python` or `gcc`. |
| `version` | No | Piston runtime version selector. Defaults to `*`. Required when `python_packages` is non-empty. |
| `python_packages` | No | Optional Python package requirements for a prebuilt custom Python runtime. Only valid with `language="python"`; the deployment must provide the matching runtime before execution. |
| `stdin` | No | Text passed to standard input. Defaults to an empty string. |
| `args` | No | Command-line arguments passed to the program. Defaults to an empty list. |
| `compile_timeout` | No | Compile wall-time limit in milliseconds. Defaults to `10000`. |
| `run_timeout` | No | Run wall-time limit in milliseconds. Defaults to `3000`, matching stock Piston's default run limit. |
| `compile_cpu_time` | No | Compile CPU-time limit in milliseconds. Defaults to `3000`. |
| `run_cpu_time` | No | Run CPU-time limit in milliseconds. Defaults to `3000`. |
| `sandbox_url` | Yes | HTTP or HTTPS Piston API base URL, such as `http://localhost:2000`. |

## Output shape

The output column contains a dictionary per row:

```python
{
    "stdout": "42\n",
    "stderr": "",
    "output": "42\n",
    "exit_code": 0,
    "signal": None,
    "message": None,
    "status": None,
    "cpu_time": 12.5,
    "wall_time": 15.2,
    "memory": 8192,
}
```

Empty or missing source code returns `exit_code=-2`. Sandbox API failures return
`exit_code=-1` with the final error in `stderr` and `message`.

## Python Packages

`python_packages` is declarative metadata. The plugin sends `language` and
`version` to Piston; it does not install packages or build runtimes during
generation. If you set a non-empty `python_packages` list, also set `version` to
the exact custom Python runtime version that your deployment has already built
and installed in Piston.

## Example

```python
import pandas as pd
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource

builder = DataDesignerConfigBuilder()
builder.with_seed_dataset(
    DataFrameSeedSource(df=pd.DataFrame({"code": ["print(6 * 7)"]}))
)
builder.add_column(
    name="sandbox_result",
    column_type="code-sandbox",
    target_column="code",
    language="python",
    version="*",
    sandbox_url="http://localhost:2000",
)
```

## MCP tool

Use `SandboxMCPConfig` to create a `LocalStdioMCPProvider` for Data Designer
tool-calling workflows:

```python
from data_designer_sandbox_piston import SandboxMCPConfig

sandbox_mcp = SandboxMCPConfig(
    name="sandbox",
    sandbox_url="http://localhost:2000",
    language="python",
    result_fields=["stdout", "stderr", "exit_code"],
)
mcp_provider = sandbox_mcp.to_provider()
tool_config = sandbox_mcp.to_tool_config()
```

The MCP process can also be launched directly:

```bash
SANDBOX_URL=http://localhost:2000 \
SANDBOX_LANGUAGE=python \
SANDBOX_VERSION='*' \
SANDBOX_RUN_TIMEOUT=3000 \
SANDBOX_RUN_CPU_TIME=3000 \
SANDBOX_TOOL_DESCRIPTION='Execute Python code in a sandbox.' \
SANDBOX_RESULT_FIELDS=stdout,stderr,exit_code \
python -m data_designer_sandbox_piston.mcp_server
```

## Local and remote Piston

For local development on macOS or Linux, run Piston in Docker and point
`sandbox_url` at `http://localhost:2000`. The package includes a convenience
script and Docker Compose example under `scripts/` and `docker/`.

The local container stores Piston runtime packages under `/piston` in a Docker
volume. A fresh volume has no runtimes installed; install the runtimes you need
through Piston's package API, for example:

```bash
curl -X POST http://localhost:2000/api/v2/packages \
  -H 'Content-Type: application/json' \
  -d '{"language":"python","version":"3.12.0"}'
```

For remote deployment, build or run a Piston API image and expose port `2000`
inside your deployment boundary. Piston must be run with the privileges and
kernel support required by its own sandboxing model. See the
[Piston project](https://github.com/engineer-man/piston) for current runtime
installation and security guidance.
