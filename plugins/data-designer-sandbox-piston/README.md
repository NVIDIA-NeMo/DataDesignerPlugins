# data-designer-sandbox-piston

Piston-backed code execution for Data Designer.

This package provides:

- A `code-sandbox` column type that executes code from an existing column via a
  local or remote Piston API.
- A stdio MCP server that exposes the same sandbox as a `run_code` tool.
- Local Docker helper examples for macOS and Linux development.

## Installation

```bash
uv add data-designer data-designer-sandbox-piston
```

## Column usage

```python
builder.add_column(
    name="sandbox_result",
    column_type="code-sandbox",
    target_column="python_code",
    language="python",
    sandbox_url="http://localhost:2000",
)
```

## MCP usage

```python
from data_designer_sandbox_piston import SandboxMCPConfig

sandbox_mcp = SandboxMCPConfig(
    name="sandbox",
    sandbox_url="http://localhost:2000",
    language="python",
)
mcp_provider = sandbox_mcp.to_provider()
tool_config = sandbox_mcp.to_tool_config()
```

Plugin documentation for the repository site lives in this package's `docs/`
directory.
