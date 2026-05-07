# data-designer-generalist-agent-env

Generate Generalist-style agent environments and tasks from Data Designer
generated topics, constraints, database schemas, and records. The plugin
assembles generated grounding data into executable tool environments, then
generates task prompts, tool-only solution functions, and verifier functions
from those environments.

## Installation

```bash
uv add "data-designer>=0.5.9" data-designer-generalist-agent-env
```

## Usage

Once installed, the `generalist-agent-environment` and
`generalist-agent-task` column types are automatically discovered by
[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).

Configure the workflow after generating a task topic, constraints, schema, and
records:

```python
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

For the full plugin authoring guide, see the
[main repository docs](https://nvidia-nemo.github.io/DataDesignerPlugins/authoring/).

Plugin documentation for the repository site lives in this package's `docs/`
directory.
