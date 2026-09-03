# Data Designer Gym

Export any Data Designer-generated task to NVIDIA NeMo Gym and join its rollout back to a stable scenario ID.

## Python workflow

Map generated columns into Gym's task structure in the processor config:

```python
from data_designer_gym.config import GymTaskProcessorConfig

builder.add_processor(
    GymTaskProcessorConfig(
        name="gym_tasks",
        messages=[
            {"role": "system", "content_column": "system_prompt"},
            {"role": "user", "content_column": "user_query"},
        ],
        tools_column="tools_json",
        response_params={"parallel_tool_calls": False},
        task_columns={
            "ground_truth": "reference_plan.tool_calls",
            "category": "category",
        },
        task_values={"environment_name": "my_environment"},
        include="reference_plan.is_valid",
        provenance_columns=["seed_id", "source"],
    )
)
```

Nested source paths such as `reference_plan.tool_calls` are resolved by the plugin. Other task fields remain opaque and pass through unchanged.

For tasks that require arbitrary Python transformations, assemble the complete object in a Data Designer column and use `task_column` instead:

```python
builder.add_processor(
    GymTaskProcessorConfig(
        name="gym_tasks",
        task_column="gym_task",
        provenance_columns=["seed_id", "source"],
    )
)
```

The preassembled column must contain Gym's required `responses_create_params` object:

```python
gym_task = {
    "responses_create_params": {
        "input": [{"role": "user", "content": "Write about research."}],
    },
    "instructions": [
        {"instruction_id": "keywords:existence", "keywords": ["research"]},
    ],
}
```

Fields such as `ground_truth`, category, tools, multimodal input, or environment state remain opaque to the plugin.

After generation, write the processor artifacts as Gym JSONL without another model call:

```python
import json
from pathlib import Path

results = designer.create(builder, num_records=100, dataset_name="agent-tasks")
artifacts = results.load_processor_dataset("gym_tasks")

output_path = Path("artifacts/gym-tasks.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(artifacts["task_json"]) + "\n")
```

The processor also supports:

- `scenario_id_column` to preserve an existing stable ID
- `include` to export only rows whose source path is truthy
- `tool_fields` to keep only runtime-compatible fields from each tool schema
- `scenario_namespace` for generated IDs
- `provenance_columns` to carry source information through ingestion

It writes the canonical scenario and Gym-native task as lossless JSON inside normal Data Designer processor artifacts.

## From-scratch notebook

The [Workplace Assistant notebook](examples/workplace-assistant.ipynb) rebuilds Gym's [existing synthetic-data example](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/workplace_assistant/notebooks/synthetic-data-generation) as a normal Data Designer workflow. It defines the environment-specific seeds, prompts, schemas, and qualification in the notebook.

The only plugin object used to define the workflow is `GymTaskProcessorConfig`, which assembles and exports the Gym task without a custom column.

## Scenario bundle

Each processor record contains a canonical scenario and its Gym-native task:

```json
{
  "schema_version": "1",
  "scenario_id": "my-environment-1f4b...",
  "task": {
    "responses_create_params": {"input": []},
    "ground_truth": {}
  },
  "provenance": {"seed_id": 12}
}
```

When exporting, the plugin adds `_ng_task_index`, `_dd_scenario_id`, and `_dd_provenance`. It supplies `id` from the stable task index only when the task does not already have one.

## CLI utilities

Translate saved scenario bundles to Gym JSONL:

```bash
data-designer-gym export scenarios.jsonl --output gym-tasks.jsonl
```

After Gym runs the tasks, normalize completed rollouts and the optional failure sidecar:

```bash
data-designer-gym ingest \
  --tasks gym-tasks.jsonl \
  --rollouts rollouts.jsonl \
  --failures rollouts_failures.jsonl \
  --output normalized-rollouts.jsonl
```

Normalized records retain the complete Gym task and rollout with stable identity restored. They can seed another Data Designer or downstream workflow.

## Current scope

The plugin covers generic Gym task export, stable scenario bundles, and rollout ingestion. Runtime orchestration, Harbor adapters, and interactive user simulation remain separate concerns.
