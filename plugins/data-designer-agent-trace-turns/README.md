# data-designer-agent-trace-turns

Normalize agent traces into one row per conversation turn.

## Installation

```bash
pip install data-designer-agent-trace-turns
```

## Usage

Once installed, the `agent-trace-turns` column type is automatically discovered by
[NeMo Data Designer](https://github.com/NVIDIA/NeMo-Data-Designer). The plugin
accepts Python dict/list payloads, JSON strings, and Data Designer `__trace`
columns, then expands each input row into one row per extracted conversational
step.

```python
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.interface.data_designer import DataDesigner

builder = DataDesignerConfigBuilder()
builder.add_seed_dataframe(
    [
        {
            "trace": [
                {"role": "user", "content": [{"type": "text", "text": "Summarize this trace."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Here is the summary."}]},
            ]
        }
    ]
)
builder.add_column(
    name="turn_text",
    column_type="agent-trace-turns",
    source_column="trace",
)

preview = DataDesigner().preview(builder, num_records=1).dataset
```

The output column contains normalized turn text. Side-effect columns provide the
conversation index, turn index, role, speaker, turn kind, extraction path, and
optionally the raw serialized turn payload.

For the full plugin authoring guide, see the
[main repository docs](https://gitlab-master.nvidia.com/etramel/data-designer-plugins/-/blob/main/docs/adding-a-plugin.md).
