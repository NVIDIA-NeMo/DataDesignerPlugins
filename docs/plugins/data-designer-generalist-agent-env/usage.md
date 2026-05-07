# Usage

This example creates one Generalist RL rollout tuple from generated data. The
model generates the topic, constraints, database schema, and database records.
The plugin assembles those generated artifacts, adds executable tools, and then
synthesizes a task, tool-only solution, and verifier.

```python
import pandas as pd
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.interface.data_designer import DataDesigner

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
        "primary_key": {"type": "string"},
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

seed_df = pd.DataFrame(
    {
        "seed": ["planning a travel itinerary"],
        "brief": ["family-friendly museums, moderate budget, reliable transport"],
    }
)

builder = DataDesignerConfigBuilder()
builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
builder.add_column(
    name="task_topic",
    column_type="llm-text",
    model_alias="deepseek-v4-pro-live",
    prompt="From this seed {{ seed }}, write a concise task topic.",
)
builder.add_column(
    name="task_constraints",
    column_type="llm-structured",
    model_alias="deepseek-v4-pro-live",
    prompt=(
        "For topic {{ task_topic }} and brief {{ brief }}, generate constraints "
        "that make the task hard to solve but easy to verify."
    ),
    output_format=constraint_schema,
)
builder.add_column(
    name="database_schema",
    column_type="llm-structured",
    model_alias="deepseek-v4-pro-live",
    prompt=(
        "Generate a database schema for topic {{ task_topic }} and constraints "
        "{{ task_constraints }}. Include record_id, name, summary, cost, "
        "duration, score, tags, and topic-specific attributes."
    ),
    output_format=database_schema_format,
)
builder.add_column(
    name="database_records",
    column_type="llm-structured",
    model_alias="deepseek-v4-pro-live",
    prompt=(
        "Generate 8 diverse records that follow schema {{ database_schema }} "
        "for topic {{ task_topic }} and constraints {{ task_constraints }}. "
        "At least two records must include the tag reliable."
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

result = DataDesigner(artifact_path="artifacts").preview(builder, num_records=1)
environment_tuple = result.dataset.loc[0, "agent_task"]
```

The generated row can be validated with the package helper:

```python
from data_designer_generalist_agent_env.validation import verify_environment_tuple

validation = verify_environment_tuple(environment_tuple)
assert validation.passed, validation.errors
assert validation.answer == environment_tuple["reference_answer"]
```

The output task is intentionally search-like: the solving agent must inspect the
generated schema, filter records, and rank candidates through the tool interface.
The verifier remains straightforward because it checks fixed constraints and a
deterministic tie-break order directly against the generated database.

## Expected output shape

`generalist-agent-environment` emits:

```text
schema_version
environment.database_schema
environment.database
environment.data_generation
tools
tool_module_source
synthesis_trace
```

`generalist-agent-task` emits:

```text
schema_version
environment
tools
tool_module_source
task
solution
verifier
reference_answer
task_iterations
synthesis_trace
rl_filter_note
```
