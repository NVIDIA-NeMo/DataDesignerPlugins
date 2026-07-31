# Usage

This plugin is a client for a running Piston API. It does not install Piston
runtimes for you, which keeps the Data Designer package portable across local,
cluster, and remote service deployments.

## Start a local sandbox

Use Docker on macOS or Linux:

```bash
bash plugins/data-designer-sandbox-piston/scripts/run-local-piston.sh
```

The script starts a container on `http://localhost:2000`. You can also use the
Docker Compose example:

```bash
docker compose -f plugins/data-designer-sandbox-piston/docker/docker-compose.yml up --build
```

Piston runtime packages are stored under `/piston` inside a Docker volume. A new
Piston package volume starts empty, so install the runtimes your workflow needs:

```bash
curl -X POST http://localhost:2000/api/v2/packages \
  -H 'Content-Type: application/json' \
  -d '{"language":"python","version":"3.12.0"}'
```

## Execute a code column

```python
from data_designer.config.config_builder import DataDesignerConfigBuilder

builder = DataDesignerConfigBuilder()
builder.add_column(
    name="result",
    column_type="code-sandbox",
    target_column="python_code",
    language="python",
    sandbox_url="http://localhost:2000",
)
```

Each output value is a JSON-serializable dictionary containing stdout, stderr,
exit code, status, timing, and memory fields.

If a workflow needs Python packages, build or provide a custom Piston Python
runtime first, then set `version` to that exact runtime version and list the
expected packages in `python_packages`. The package list is metadata for humans
and tool descriptions; execution still depends on the configured Piston runtime.

## Add an MCP `run_code` tool

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

Pass `mcp_provider` and `tool_config` into the Data Designer configuration path
that configures MCP providers and tool aliases for your LLM columns.

If the sandbox URL is assigned by your launcher at runtime, omit `sandbox_url`
from `SandboxMCPConfig` and set `SANDBOX_URL` in the environment inherited by the
MCP subprocess.

## Remote endpoint

For a remote deployment, use the same configuration with a different URL:

```python
builder.add_column(
    name="result",
    column_type="code-sandbox",
    target_column="python_code",
    language="python",
    sandbox_url="https://piston.example.internal",
)
```

Keep the Piston endpoint private to trusted Data Designer workers or put it
behind your own authentication proxy. The plugin forwards code to the configured
endpoint and relies on that service for isolation and runtime availability.
