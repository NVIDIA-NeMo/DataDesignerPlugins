# data-designer-generalist-agent-env

Generate Generalist-style agent environment tuples from seed task categories.
Each output value contains a sandbox database, synthesized task-specific tool
functions, a task prompt, a tool-only solution function, and a verifier function.

## Installation

```bash
uv add "data-designer>=0.5.9" data-designer-generalist-agent-env
```

## Usage

Once installed, the `generalist-agent-env` column type is automatically discovered by
[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).

Configure the column with a task category column and optional context columns:

```python
builder.add_column(
    name="agent_env",
    column_type="generalist-agent-env",
    task_category_column="category",
    context_columns=["constraints"],
    difficulty="hard",
)
```

For the full plugin authoring guide, see the
[main repository docs](https://nvidia-nemo.github.io/DataDesignerPlugins/authoring/).

Plugin documentation for the repository site lives in this package's `docs/`
directory.
