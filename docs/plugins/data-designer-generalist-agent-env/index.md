# data-designer-generalist-agent-env

The `data-designer-generalist-agent-env` plugin adds a `generalist-agent-env`
column type for creating Generalist-style agent environment tuples inspired by
the DeepSeek-V3.2 automatic environment synthesis workflow.

For each seed row, the plugin builds a row-local sandbox database, exposes
task-specific tool functions, synthesizes a constrained task, emits a solution
function that only calls tools and performs local logic, and emits a verifier
function that checks candidate answers against the database.

## Installation

```bash
uv add "data-designer>=0.5.9" data-designer-generalist-agent-env
```

## Column type

Use the `generalist-agent-env` column type when a dataset needs structured
`<environment, tools, task, verifier>` records for agent training or evaluation.

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Output column name. |
| `task_category_column` | Yes | Existing column containing a task category such as `planning a travel itinerary`. |
| `context_columns` | No | Existing columns copied into the synthesized sandbox database context. |
| `difficulty` | No | Final task difficulty: `simple`, `medium`, or `hard`; defaults to `hard`. |
| `database_size` | No | Number of sandbox records to synthesize per row; defaults to `8`. |
| `required_tag` | No | Optional tag that the valid answer must contain. |
| `max_cost` | No | Optional maximum cost constraint. Unsatisfiable values are repaired upward. |
| `min_score` | No | Optional minimum score constraint. Unsatisfiable values are repaired downward. |

## Usage

```python
import pandas as pd
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource

seed_df = pd.DataFrame(
    {
        "category": ["planning a travel itinerary"],
        "constraints": ["compare candidate plans by score, cost, and family suitability"],
    }
)

builder = DataDesignerConfigBuilder()
builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
builder.add_column(
    name="agent_env",
    column_type="generalist-agent-env",
    task_category_column="category",
    context_columns=["constraints"],
    required_tag="family",
)
```

The generated `agent_env` value is a dictionary with these top-level keys:

| Key | Description |
| --- | --- |
| `environment` | Sandbox metadata, row-local database, schema, and source context. |
| `tools` | Synthesized tool descriptors and Python function sources. |
| `tool_module_source` | Executable Python source defining the hidden database and selected tools. |
| `task` | Prompt, difficulty, constraints, and answer schema. |
| `solution` | Python `solve(tools)` source restricted to tool calls and local logic. |
| `verifier` | Python `verify(answer, database)` source and reference validation status. |
| `reference_answer` | The generated solution output that the verifier accepts. |
| `task_iterations` | Simple-to-final task, solution, verifier, and augmentation artifacts. |
| `synthesis_trace` | Environment construction, task synthesis, tool augmentation, solution, and verification events. |

## Row validation helper

The package includes a helper module for executable row validation. It executes
the generated tool module, smoke-tests the declared tools, runs the generated
solution, checks the generated verifier, and replays every task iteration:

```python
from data_designer_generalist_agent_env.validation import verify_row_record

validation = verify_row_record(result.dataset.loc[0], output_column="agent_env")
assert validation.passed, validation.errors
```

## Behavior Notes

The plugin is deterministic and does not call the Internet. It records `bash`
and `search` as base sandbox tools and uses the seed row to synthesize the
sandbox database locally. Downstream workflows can replace or augment that
database with retrieved records before using the generated task and verifier.
