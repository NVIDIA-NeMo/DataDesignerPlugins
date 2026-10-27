# Data Designer Gym

Experimental initial integration: Data Designer columns call operator-managed NVIDIA NeMo Gym services, preserve outputs and attempt evidence, and feed downstream processing. Task export and rollout ingestion are also available.

The operator starts, checks, and stops Gym services and any model runtime. The plugin installs neither Gym nor a model server and does not manage service placement or Slurm allocations.

## Installation

Once 0.1.0 is published, install from the NVIDIA-NeMo plugin index into an active Python environment:

```bash
uv pip install \
    --default-index https://pypi.org/simple/ \
    --index https://nvidia-nemo.github.io/DataDesignerPlugins/simple/ \
    data-designer-gym
```

From a source checkout containing this plugin, start in `plugins/data-designer-gym`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'data-designer==0.9.7' .
```

The package declares Python >=3.10 and DD >=0.9.7. The checked combinations below are narrower than those dependency ranges; other combinations require validation.

| Check | DD runtime | Gym runtime |
| --- | --- | --- |
| Plugin suite | DD 0.9.7, Python 3.10.21 and 3.12.12 | Simulated HTTP responses |
| Installed wheel and local resource example | DD 0.9.7, Python 3.10.21 | Gym `58b3240e04f1fa2061328a5d9a8ec7581262f8eb`, Python 3.14.4 |
| Optional Pi restart smoke | DD 0.9.7, Python 3.10.21 | Same Gym revision, Python 3.14.4, Pi 1.2.0, Node 25.9.0 |

### First local run

This resource-only path uses a synthetic SQLite database and makes no model calls. Install the example's Gym runtime separately:

```bash
uv venv --python 3.14 .venv-gym
uv pip install --python .venv-gym/bin/python \
    'nemo-gym @ https://github.com/NVIDIA-NeMo/Gym/archive/58b3240e04f1fa2061328a5d9a8ec7581262f8eb.tar.gz'
```

In terminal 1, start the demo services. The unused model URL satisfies the launcher configuration; this path calls only the resource service.

```bash
MODEL_API_KEY=unused .venv-gym/bin/python examples/toy_gym.py \
    --database-dir "$PWD/.gym-demo/databases" \
    --model-endpoint http://127.0.0.1:1/v1 --model unused
```

In terminal 2, in the same plugin directory, wait until `curl --fail http://127.0.0.1:8765/stats` succeeds, then generate one row and consume its evidence in a downstream column:

```bash
.venv/bin/python - <<'PY'
import json

import data_designer.config as dd
import httpx
import pandas as pd
from data_designer.interface import DataDesigner

from data_designer_gym.config import GymResourceColumnConfig

builder = dd.DataDesignerConfigBuilder().with_seed_dataset(
    dd.DataFrameSeedSource(df=pd.DataFrame([{"row_id": "demo-1", "database": "original"}])),
    sampling_strategy="ordered",
)
for model in tuple(builder.model_configs):
    builder.delete_model_config(model.alias)
builder.add_column(GymResourceColumnConfig(
    name="execution",
    endpoint="http://127.0.0.1:8765",
    route="/execute",
    row_id_column="row_id",
    request_columns={"database": "database"},
    request_values={"sql": "SELECT count(*) FROM items"},
    session={
        "initialize": [{"endpoint": "http://127.0.0.1:8765", "route": "/seed_session",
                        "request_columns": {"database": "database"}}],
        "cleanup": [{"endpoint": "http://127.0.0.1:8765", "route": "/cleanup"}],
    },
))
builder.add_column(dd.ExpressionColumnConfig(
    name="call_completed", expr="{{ execution.status == 'completed' }}", dtype="bool",
))
result = DataDesigner(artifact_path=".gym-demo/artifacts").create(builder, num_records=1)
rows = result.load_dataset()
assert rows["call_completed"].all()
assert json.loads(rows.iloc[0]["execution"]["response_json"])["rows"] == [[3]]
rows[rows["call_completed"]].to_json(".gym-demo/completed.jsonl", orient="records", lines=True)
assert httpx.get("http://127.0.0.1:8765/stats").json()["active_sessions"] == 0
print(rows[["row_id", "call_completed"]])
PY
```

The run retains native evidence under `.gym-demo/artifacts/` and exports the completed row. Call completion does not establish semantic correctness. Stop terminal 1 with Ctrl+C after generation; the operator owns service shutdown. Keep `.gym-demo/` for inspection, or remove that run directory after both terminals have finished. The plugin's configured per-row cleanup removes the demo session, not its databases or dataset artifacts.

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

`max_parallel_requests` bounds active row lifecycles per generator, including cleanup. DD also receives service scheduling metadata. Limits are per worker/generator; aggregate concurrency across columns and workers belongs in the runner. Sharing a model endpoint does not share DD/Gym clients, model admission, retries, or usage accounting. `timeout` bounds setup plus execution, and `session.cleanup_timeout` bounds each cleanup step. Admission waits are cancellable without an execution deadline; set `queue_timeout` for a separate queue deadline. Python callers with a running event loop should use `await generator.agenerate(row)` for long calls.

Failures are retained as data, with distinctions for request validation, admission timeout, execution timeout, HTTP rejection, server failure, transport failure, malformed response, unfinished harness output, cleanup, and artifact storage. HTTP success means the resource call completed. SQL errors, verification reward/mask/failure fields, and semantic rejection remain in the native response for downstream policies. The plugin does not retry state-changing requests. Cleanup runs after partial initialization, timeout, and cancellation, but a remote operation may continue after a client timeout; the service must enforce its own execution deadline. Configure a TTL or administrative cleanup for operations whose initialization fails before returning a session identifier.

`failure_stage` identifies `request`, `admission`, `initialize`, `execute`, `cleanup`, or `storage`; `failure_route` identifies the configured HTTP route when applicable. Both are null on success. Initialization response bodies remain excluded because they may contain session credentials. Cleanup errors remain available separately when initialization or execution also fails.

DD resume can re-execute a column whose dataset batch was not persisted, even if its remote operation already committed. Stable row IDs and attempt artifacts do not provide remote deduplication. Use the service's idempotency contract where available, and persist generated request inputs before a separate execution stage when re-generating them could change an already committed request.

Cleanup failure marks the call failed and clears a harness answer; artifact-write failure also clears it. Cancellation writes attempt evidence and propagates to DD instead of returning an ordinary failed row. The workflow must decide which returned rows to keep or reject.

Servers must implement their own execution restrictions, database allowlists, deadlines, and cleanup semantics. Run stateful Gym services with one worker unless the deployed revision explicitly supports more. The plugin uses direct HTTP connections (`trust_env=False`), so it does not inherit proxy environment variables.

### Gym episodes and response projections

Use `response_protocol="gym-episode"` on a resource column to interpret Gym's `result` / `failure` envelope. The existing `status` and `failure_kind` continue to describe the call and cleanup. The additional `episode_status`, `episode_failure_kind`, `episode_failure_reason`, `episode_failure_stage`, `episode_terminal`, `reward` and `mask_sample` describe the episode. A returned result is `completed` even when its reward is zero or its sample is masked; filtering and acceptance stay in the workflow. Malformed outcome fields produce a `response` failure, with the original response retained in the artifact.

```python
builder.add_column(GymResourceColumnConfig(
    name="episode",
    endpoint="http://gym-environment:8000",
    route="/run",
    row_id_column="row_id",
    request_column="episode_request",  # A prepared Gym episode request, including identity.
    response_protocol="gym-episode",
    response_fields={"trajectory_id": "result.usersim_result.trajectory_id"},
))
```

`response_fields` maps output names to dot-separated JSON object paths. On a resource column it replaces the inline `response_json` with the projection. Missing optional paths become null; `{}` keeps an empty inline object. Omitting the option preserves the existing full resource response. The full response remains in the attempt artifact in every case. Harness columns can also set `response_fields` to add projected `response_json` to their existing evidence column without changing the answer text.

This calls a whole Gym-hosted episode. It does not move UserSim's conversation loop into DD or add a sandbox client. Prepare the native episode request separately; the task exporter still uses its existing Responses-based task format. Align the column timeout with the deployed server's episode deadline and cleanup budget.

### Live SQL example

[execution_grounded_sql.py](examples/execution_grounded_sql.py) replays saved public SQL candidates against original/variant toy databases, runs a tool-assisted harness, applies ordinary DD judges, and exports accepted rows. This isolates integration parity from independently sampled generations. Replace the saved question/SQL/reasoning seed fields with normal DD generators for SDG.

[toy_gym.py](examples/toy_gym.py) supplies the demo-only SQLite `/execute` extension and uses upstream Gym `SimpleAgent`, `SimpleModelServer`, and `ExampleMultiStepResourcesServer`. Standard BIRD/text-to-SQL verification requires reference SQL and does not replace execution-only SDG.

Use the separate environments from [Installation](#installation). Set `MODEL_ENDPOINT`, `MODEL_NAME`, and `MODEL_API_KEY` for an existing compatible model service in both terminals. Run from the plugin directory:

```bash
# Terminal 1:
.venv-gym/bin/python examples/toy_gym.py \
    --database-dir /tmp/dd-gym-databases \
    --model-endpoint "$MODEL_ENDPOINT" --model "$MODEL_NAME"

# Terminal 2, after /stats responds on port 8765:
.venv/bin/python examples/execution_grounded_sql.py \
    --model-endpoint "$MODEL_ENDPOINT" --model "$MODEL_NAME" \
    --output /tmp/dd-gym-smoke
```

The model endpoint must support Responses (harness) and Chat Completions with structured output (DD judges). For a Chat Completions endpoint, pass `--model-adapter chat` to the demo service to use Gym's `VLLMModel` conversion adapter without deploying inference locally. Stop terminal 1 with Ctrl+C after inspecting the results and zero active sessions at `/stats`. On Slurm, service launch, placement, readiness, endpoint publication, and teardown stay in the runner; this example does not implement that orchestration.

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

Missing `include` values skip the row. A configured `scenario_id_column` must contain a non-missing ID. Task payloads and provenance reject non-finite or non-JSON values; use `None` for intentional JSON nulls.

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

Default scenario IDs hash the task content. Set `scenario_id_column` to a unique stable row ID when identical tasks have different provenance. Ingestion rejects conflicting tasks with the same index; identical duplicate task records are allowed.

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

Omit `--failures` when no sidecar was produced. An explicitly supplied sidecar path must exist.

Ingest also exposes `rollout_id`, `attempt`, `task_id` and `environment_server`, and maps native `_ng_failure_message` / `_ng_failure_stage` into `failure_reason` / `failure_stage`. Explicit legacy failure fields take precedence. Missing attempt IDs and failure kinds remain null rather than being inferred. Joining still requires the exported task index; native materialized-task export is separate work.

## Current scope

Implemented plugin features are resource/harness HTTP columns, retained native evidence, optional episode-outcome interpretation and projections, task export, and rollout ingestion. Row IDs must be supplied by the workflow; generated attempt IDs are retained in artifacts but are not automatically injected into service requests.

The optional Pi service has its own runtime and lifecycle. A UserSim session client/probe, DD wrapper, recorder, and environment-specific cleanup shim are not shipped plugin APIs. Whole-episode HTTP calls do not establish support for a UserSim loop inside DD, sandbox sessions, or arbitrary Gym environments. Runtime orchestration and Harbor adapters remain outside this initial integration.

## Proposed follow-up stages

These are technical milestones for discussion, not approved delivery commitments:

1. **UserSim session client/probe:** document the service and cleanup contract; pass a public multi-call conversation with stable identity, downstream evidence, typed failures, cancellation, and zero leaked sessions.
2. **Shared model admission/accounting:** prove DD and Gym calls to the same model share an admission budget and usage ledger; test rate limits from either caller, cancellation, and separate caller totals.
3. **Managed worker lifecycle and Slurm planning:** plan service and peak sandbox resources before submission; start each compatible graph once per worker attempt, gate generation on readiness, and verify cleanup on success, startup failure, service crash, cancellation, and retry.
4. **Sandbox persistence workflow:** pin one sandbox provider and coding harness; create a file in one DD stage, restart the agent service, resume the same conversation/workspace in a follow-up stage, verify the file change and cross-row isolation, and remove owned resources. Define interrupted-turn recovery separately from completed-turn persistence.
