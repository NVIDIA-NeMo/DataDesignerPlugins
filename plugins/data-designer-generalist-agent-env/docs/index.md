# data-designer-generalist-agent-env

The `data-designer-generalist-agent-env` plugin adds a two-stage Generalist
environment workflow for Data Designer. It is designed for workflows where Data
Designer generates the topic, constraints, database schema, and database rows,
then the plugin assembles those generated artifacts into executable RL rollout
tuples.

The workflow is:

1. Use ordinary Data Designer columns, such as `llm-text` and `llm-structured`,
   to generate a task topic and constraints.
2. Use additional Data Designer generation columns to generate a row-local
   database schema and records that follow that schema.
3. Use `generalist-agent-environment` to validate and assemble the generated
   schema and records into a sandbox with executable tools.
4. Use `generalist-agent-task` to synthesize the task prompt, tool-only solution,
   verifier, reference answer, and simple-to-hard augmentation trace.

No search provider or external retrieval step is required, and the plugin does
not fabricate fallback records.

## Installation

```bash
uv add "data-designer>=0.5.9" data-designer-generalist-agent-env
```

## Column types

Use `generalist-agent-environment` to assemble the generated sandbox and toolset.

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Output environment column name. |
| `task_topic_column` | Yes | Existing column containing a generated task topic such as `trip planning`. |
| `task_constraints_column` | No | Existing column containing generated constraints as text, JSON, or a structured object. |
| `database_schema_column` | Yes | Existing column containing the generated database schema. |
| `database_records_column` | Yes | Existing column containing generated database records. |
| `context_columns` | No | Existing columns copied into environment context. |

Generated records must include `record_id`, `name`, `summary`, `cost`,
`duration`, `score`, and `tags`. Additional fields are preserved, and an
`attributes` object is recommended for topic-specific fields.

Use `generalist-agent-task` to generate tasks from an environment.

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Output task tuple column name. |
| `environment_column` | Yes | Column containing a `generalist-agent-environment` artifact. |
| `difficulty` | No | Final task difficulty: `simple`, `medium`, or `hard`; defaults to `hard`. |
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
        "seed": ["travel planning"],
        "brief": ["family-friendly museums, moderate budget, reliable transport"],
    }
)

constraint_schema = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "success_criteria": {"type": "array", "items": {"type": "string"}},
        "data_dimensions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["goal", "constraints", "success_criteria", "data_dimensions"],
}

database_schema_format = {
    "type": "object",
    "properties": {
        "record_type": {"type": "string"},
        "primary_key": {"type": "string", "const": "record_id"},
        "fields": {"type": "array", "items": {"type": "object"}},
        "attribute_fields": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["record_type", "primary_key", "fields", "attribute_fields"],
}

records_format = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "name": {"type": "string"},
                    "summary": {"type": "string"},
                    "cost": {"type": "integer"},
                    "duration": {"type": "integer"},
                    "score": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "attributes": {"type": "object"},
                },
                "required": ["record_id", "name", "summary", "cost", "duration", "score", "tags"],
            },
        }
    },
    "required": ["records"],
}

builder = DataDesignerConfigBuilder()
builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
builder.add_column(
    name="task_topic",
    column_type="llm-text",
    model_alias="deepseek-v4-pro-live",
    prompt="From {{ seed }} and {{ brief }}, write a concise task topic.",
)
builder.add_column(
    name="task_constraints",
    column_type="llm-structured",
    model_alias="deepseek-v4-pro-live",
    prompt="Generate constraints for topic {{ task_topic }} with brief {{ brief }}.",
    output_format=constraint_schema,
)
builder.add_column(
    name="database_schema",
    column_type="llm-structured",
    model_alias="deepseek-v4-pro-live",
    prompt="Generate a database schema for topic {{ task_topic }} and constraints {{ task_constraints }}.",
    output_format=database_schema_format,
)
builder.add_column(
    name="database_records",
    column_type="llm-structured",
    model_alias="deepseek-v4-pro-live",
    prompt=(
        "Generate 8 records that follow schema {{ database_schema }} for topic "
        "{{ task_topic }} and constraints {{ task_constraints }}. Include varied "
        "cost, duration, score, tags, and attributes."
    ),
    output_format=records_format,
)
builder.add_column(
    name="agent_environment",
    column_type="generalist-agent-environment",
    task_topic_column="task_topic",
    task_constraints_column="task_constraints",
    database_schema_column="database_schema",
    database_records_column="database_records",
    context_columns=["brief"],
)
builder.add_column(
    name="agent_task",
    column_type="generalist-agent-task",
    environment_column="agent_environment",
    difficulty="hard",
    required_tag="reliable",
)
```

The generated `agent_task` value is a dictionary with these top-level keys:

| Key | Description |
| --- | --- |
| `environment` | Sandbox metadata, generated database schema, generated records, and source context. |
| `tools` | Synthesized tool descriptors and Python function sources. |
| `tool_module_source` | Executable Python source defining the generated schema, generated database, and selected tools. |
| `task` | Prompt, difficulty, constraints, and answer schema. |
| `solution` | Python `solve(tools)` source restricted to tool calls and local logic. |
| `verifier` | Python `verify(answer, database)` source and reference validation status. |
| `reference_answer` | The generated solution output that the verifier accepts. |
| `task_iterations` | Simple-to-final task, solution, verifier, and augmentation artifacts. |
| `synthesis_trace` | Topic/constraint intake, schema intake, generated-data intake, task synthesis, solution, and verification events. |

## Row validation helper

The package includes a helper module for executable row validation. It executes
the generated tool module, smoke-tests the declared tools, runs the generated
solution, checks the generated verifier, and replays every task iteration:

```python
from data_designer_generalist_agent_env.validation import verify_row_record

validation = verify_row_record(result.dataset.loc[0], output_column="agent_task")
assert validation.passed, validation.errors
```

## Behavior notes

The plugin does not generate the grounding records. It requires generated
schema and generated records from upstream Data Designer columns, validates the
minimum executable contract, and then builds tools and verifiers around that
generated data.
