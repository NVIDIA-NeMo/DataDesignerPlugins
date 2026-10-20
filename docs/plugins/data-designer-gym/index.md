# Data Designer Gym

Experimental initial integration: `data-designer-gym` columns call operator-managed Gym services, preserve outputs and attempt evidence, and feed downstream processing. Task export and rollout ingestion remain available. The operator owns service/model installation, startup, readiness, placement, and shutdown.

For installation, tested runtime combinations, and a local resource-to-downstream-column run with cleanup and no model calls, follow the [package README](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/main/plugins/data-designer-gym/README.md#installation).

## Resource and harness columns

For a two-turn Pi conversation with durable history and an agent-server restart, see the optional [persistent Pi service](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/main/plugins/data-designer-gym/examples/pi_sessions/README.md). It extends upstream Gym's Pi adapter in a separate runtime and works with the existing harness columns.

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

[execution_grounded_sql.py](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/main/plugins/data-designer-gym/examples/execution_grounded_sql.py) replays saved public SQL candidates against original/variant toy databases, runs a tool-assisted harness, applies ordinary DD judges, and exports accepted rows. This isolates integration parity from independently sampled generations. Replace the saved question/SQL/reasoning seed fields with normal DD generators for SDG.

[toy_gym.py](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/main/plugins/data-designer-gym/examples/toy_gym.py) supplies the demo-only SQLite `/execute` extension and uses upstream Gym `SimpleAgent`, `SimpleModelServer`, and `ExampleMultiStepResourcesServer`. Standard BIRD/text-to-SQL verification requires reference SQL and does not replace execution-only SDG.

Use the separate environments from the package README. Set `MODEL_ENDPOINT`, `MODEL_NAME`, and `MODEL_API_KEY` for an existing compatible model service in both terminals. Run from `plugins/data-designer-gym` in the source checkout:

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

Missing `include` values skip the row. A configured `scenario_id_column` must contain a non-missing ID. Task payloads and provenance reject non-finite or non-JSON values; use `None` for intentional JSON nulls.

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

Default scenario IDs hash the task content. Set `scenario_id_column` to a unique stable row ID when identical tasks have different provenance. Ingestion rejects conflicting tasks with the same index; identical duplicate task records are allowed. Omit `--failures` when no sidecar was produced; an explicitly supplied path must exist.

Ingest also exposes `rollout_id`, `attempt`, `task_id` and `environment_server`, and maps native `_ng_failure_message` / `_ng_failure_stage` into `failure_reason` / `failure_stage`. Explicit legacy failure fields take precedence. Missing attempt IDs and failure kinds remain null rather than being inferred. Joining still requires the exported task index; native materialized-task export is separate work.

## Limits and proposed follow-up

The optional Pi service has its own runtime and lifecycle. A UserSim session client/probe, DD wrapper, recorder, and environment-specific cleanup shim are not shipped plugin APIs. Whole-episode HTTP calls do not establish support for a UserSim loop inside DD, sandbox sessions, or arbitrary Gym environments. Generated attempt IDs are retained in artifacts but are not automatically injected into requests.

Proposed stages, with completion criteria, are documented in the [package roadmap](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/main/plugins/data-designer-gym/README.md#proposed-follow-up-stages): a supported UserSim session/probe and cleanup contract; shared model admission/accounting; managed worker startup/readiness/cleanup and Slurm resource planning; then one pinned sandbox/persistence workflow. These are discussion milestones, not approved commitments.
