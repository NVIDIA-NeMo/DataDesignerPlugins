# Data Designer Gym

Call NVIDIA NeMo Gym resources and harnesses inside the Data Designer DAG, or export tasks and ingest rollouts.

## Resource and harness columns

For a two-turn Pi conversation with durable history and an agent-server restart, see the optional [persistent Pi service](examples/pi_sessions/README.md). It extends upstream Gym's Pi adapter in a separate runtime and works with the existing harness columns.

The package also installs `gym-resource` and `gym-harness` column generators. These execute inside the DD DAG and declare dependencies on every request, session, identity, and provenance source path.

```python
from data_designer_gym.config import GymResourceColumnConfig, GymHarnessColumnConfig

builder.add_column(GymResourceColumnConfig(
    name="execution",
    endpoint="http://gym-resource:8000",
    route="/execute",  # An execution extension supplied by your service.
    row_id_column="row_id",
    request_columns={"database_path": "database_path", "sql": "candidate.sql"},
    provenance_columns=["candidate.id", "database_variant"],
    max_parallel_requests=4,
    timeout=60,
))

builder.add_column(GymHarnessColumnConfig(
    name="answer",
    evidence_column="harness_evidence",
    endpoint="http://gym-agent:8001",
    row_id_column="row_id",
    request_columns={"input": "question", "tools": "tools"},
    provenance_columns=["candidate.id", "database_variant"],
    session={
        "initialize": [{"endpoint": "http://gym-resource:8000", "route": "/seed_session", "request_column": "task"}],
        "cleanup": [{"endpoint": "http://gym-resource:8000", "route": "/cleanup"}],
    },
))
```

`request_columns` maps request field names to upstream paths. `request_values` adds literal JSON values. Use `request_column` instead for a preassembled JSON object, including a JSON string. Resource requests do not need `responses_create_params` or a reference answer.

The resource cell is a structured envelope with `status`, `row_id`, `attempt_id`, `artifact`, failure fields, `provenance_json`, `cleanup_errors_json`, and `response_json`. Native request/response objects are stored under the DD dataset's `gym/` directory, including during `preview()`. Preview calls therefore write evidence to disk. JSON strings keep arbitrary payloads lossless and safe for Parquet, including heterogeneous SQL results. Downstream DD judges can reference `{{ execution.response_json }}`. A subsequent Gym request can resolve paths such as `execution.response_json.rows` using the normal nested source mapping.

The harness cell contains only the final assistant text. Its declared evidence column links the full native response and available tool trace to the same row ID and provenance. The last output item must be a completed assistant message with text. An unset top-level status from Gym chat adapters is accepted only with that terminal message and no incomplete details or pending tools; pending tools, commentary, refusals without text, and incomplete responses produce a null answer with failure evidence. Adapters that require external tool continuation need a service-side completed-loop adapter. The plugin does not perform a caller-side tool loop.

Supply a unique, stable `row_id_column`. Keep candidate ID, database variant, and source database provenance in workflow columns and list them in `provenance_columns`. Calls with repeated row IDs still get separate attempt artifacts. Prompts, model selection, SQL generation/reasoning, judges, and filtering belong in the workflow.

### Lifecycle and failures

Session routes differ between environments. Configure the initialization and cleanup routes supported by your deployment, in their required order. Initialization may include both resource seeding and agent setup. The `/cleanup` route above is an example, not a universal Gym API. Stateful initialization requires a cleanup configuration; stateless calls may omit `session`.

Each row owns its HTTP client, cookies, and session headers. Connections are reused across that row's setup, execution, and cleanup. Cookies are explicitly forwarded between the trusted configured services. An initialization step can map a JSON token into a header with `response_headers={"X-Session-Token": "token"}`. Tokens and cookies are excluded from lifecycle artifacts. `bearer_token_env` reads a credential at generator startup and sends it only to that endpoint; another service may declare its own `bearer_token_env`.

`max_parallel_requests` bounds active row lifecycles per generator, including cleanup. DD also receives service scheduling metadata. Limits are per worker/generator; aggregate concurrency across columns and Slurm workers belongs in the runner. `timeout` bounds setup plus execution, and `session.cleanup_timeout` bounds each cleanup step. Admission waits are cancellable without an execution deadline; set `queue_timeout` for a separate queue deadline. Python callers with a running event loop should use `await generator.agenerate(row)` for long calls.

Failures are retained as data, with distinctions for request validation, admission timeout, execution timeout, HTTP rejection, server failure, transport failure, malformed response, unfinished harness output, cleanup, and artifact storage. HTTP success means the resource call completed. SQL errors, verification reward/mask/failure fields, and semantic rejection remain in the native response for downstream policies. The plugin does not retry state-changing requests. Cleanup runs after partial initialization, timeout, and cancellation, but a remote operation may continue after a client timeout; the service must enforce its own execution deadline. Configure a TTL or administrative cleanup for operations whose initialization fails before returning a session identifier.

Servers must implement their own execution restrictions, database allowlists, deadlines, and cleanup semantics. Run stateful Gym services with one worker unless the deployed revision explicitly supports more. The plugin uses direct HTTP connections (`trust_env=False`), so it does not inherit proxy environment variables.

### Live SQL example

[execution_grounded_sql.py](examples/execution_grounded_sql.py) replays saved public SQL candidates against original/variant toy databases, runs a tool-assisted harness, applies ordinary DD judges, and exports accepted rows. This isolates integration parity from independently sampled generations. Replace the saved question/SQL/reasoning seed fields with normal DD generators for SDG.

[toy_gym.py](examples/toy_gym.py) supplies the demo-only SQLite `/execute` extension and uses upstream Gym `SimpleAgent`, `SimpleModelServer`, and `ExampleMultiStepResourcesServer`. Standard BIRD/text-to-SQL verification requires reference SQL and does not replace execution-only SDG. The prototype's custom execution server is not shipped here.

Install Gym separately in its own environment. The plugin itself only needs DD and HTTPX. Run the services with the Gym environment and the workflow with the installed plugin environment:

```bash
# Terminal 1, with Gym installed and its source checkout on PYTHONPATH:
export MODEL_API_KEY="your endpoint credential"
/path/to/gym/.venv/bin/python examples/toy_gym.py \
    --database-dir /tmp/dd-gym-databases \
    --model-endpoint "$MODEL_ENDPOINT" --model "$MODEL_NAME"

# Terminal 2, with data-designer-gym installed:
.venv/bin/python examples/execution_grounded_sql.py \
    --model-endpoint "$MODEL_ENDPOINT" --model "$MODEL_NAME" \
    --output /tmp/dd-gym-smoke
```

The model endpoint must support Responses (harness) and Chat Completions with structured output (DD judges). For a Chat Completions endpoint, pass `--model-adapter chat` to the demo service to use Gym's `VLLMModel` conversion adapter without deploying inference locally. Run smoke tests on allocated compute nodes on Slurm. Service launch, placement, readiness, endpoint publication, and teardown stay in the runner.

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
