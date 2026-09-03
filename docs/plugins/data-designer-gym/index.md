# Data Designer Gym

`data-designer-gym` exports any Data Designer-generated Gym task and joins its rollout back to a stable scenario ID.

## Generic processor

The processor can assemble Gym tasks directly from generated columns:

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
        task_columns={"ground_truth": "reference_plan.tool_calls"},
        task_values={"environment_name": "my_environment"},
        include="reference_plan.is_valid",
        provenance_columns=["seed_id"],
    )
)
```

Nested paths map generated values into arbitrary task fields. A preassembled `task_column` remains available when an environment needs custom Python transformations. The processor owns only common-boundary validation, stable identity, and artifact serialization.

## From-scratch notebook

The [Workplace Assistant notebook](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/main/plugins/data-designer-gym/examples/workplace-assistant.ipynb) rebuilds Gym's existing example from Data Designer primitives. Its seeds, prompts, schemas, and qualification stay in the notebook. `GymTaskProcessorConfig` declaratively assembles the task without a custom column.

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

Export and ingestion are available independently of the notebook:

```bash
data-designer-gym export scenarios.jsonl --output gym-tasks.jsonl

data-designer-gym ingest \
  --tasks gym-tasks.jsonl \
  --rollouts rollouts.jsonl \
  --failures rollouts_failures.jsonl \
  --output normalized-rollouts.jsonl
```

The normalized result retains the complete task and rollout and can seed Data Designer or another downstream workflow.
