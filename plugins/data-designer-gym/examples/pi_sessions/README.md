# Persistent Pi sessions

This optional experimental service extends Gym's upstream `PiAgent` with completed-turn persistence. It reuses upstream model configuration, Responses conversion, Gym-side usage accounting, and agent observations. The extension owns native Pi history, per-conversation workspaces, process cleanup, and committed-response replay. It runs separately from the DD plugin; no upstream Gym changes or DD column changes are required.

The pinned runtime is Gym `58b3240e04f1fa2061328a5d9a8ec7581262f8eb` and Pi `1.2.0`. Keep it in its own Python environment (Python >=3.13.14; Python 3.14 works). The main DD plugin retains its Python 3.10 baseline. Install from this directory:

```bash
make install PYTHON=python3.14
make install-pi
```

Pi requires Node >=22.19.0. `install-pi` installs into `.runtime/`, not the global Node environment. Supply an existing exact-version installation through `command` instead if desired. The adapter checks its version at startup and never installs software while serving requests.

Create `agent.json` with absolute paths appropriate to the deployment:

```json
{
  "name": "pi",
  "host": "127.0.0.1",
  "port": 39336,
  "entrypoint": "app.py",
  "num_workers": 1,
  "command": "/absolute/path/pi_sessions/.runtime/node_modules/.bin/pi",
  "session_root": "/absolute/path/run/episodes",
  "model": "managed/my-model",
  "concurrency": 3,
  "timeout": 300,
  "models_config": {
    "providers": {
      "managed": {
        "baseUrl": "http://127.0.0.1:10000/v1",
        "api": "openai-completions",
        "apiKey": "MODEL_API_KEY",
        "models": [{
          "id": "my-model",
          "name": "Configured model endpoint",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 16384,
          "maxTokens": 2048
        }]
      }
    }
  }
}
```

Pi resolves `MODEL_API_KEY` from the service environment. For an unauthenticated local endpoint, use a literal dummy key instead. Use an operator-provided endpoint and served model. This service does not provision or stop model serving. Pointing DD and Pi at the same endpoint does not share model clients, throttling, retries, or accounting.

```bash
.venv/bin/dd-gym-pi-sessions --config agent.json
```

Before generation, check `curl --fail http://127.0.0.1:39336/stats`. With one server worker, `concurrency` bounds active Pi turns in this service. The plugin's `max_parallel_requests` separately bounds each generator in each DD worker; size their aggregate load against this service and the model endpoint.

Alternatively, configure `model_server` using Gym's normal model reference and pass the deployment's merged configuration as `--gym-config gym.yaml`. Upstream Pi then builds the rollout-prefixed Gym model URL. These Gym-to-model connections require the same network reachability as an ordinary Gym deployment.

## DD columns

Keep the existing `gym-harness` request contract. Supply a run-qualified, unique row ID. The adapter accepts only a text user message, optional instructions/system message, `user`, and `metadata.turn`; upstream Pi validation rejects unsupported Responses controls.

```python
from data_designer_gym.config import GymHarnessColumnConfig

builder.add_column(GymHarnessColumnConfig(
    name="reply_1",
    evidence_column="reply_1_evidence",
    endpoint="http://127.0.0.1:39336",
    row_id_column="row_id",
    request_columns={"input": "prompt", "user": "row_id"},
    request_values={"metadata": {"turn": "1"}},
    timeout=330,
    queue_timeout=1800,
    max_parallel_requests=3,
))

# Generate user_reaction from the actual reply_1 with an ordinary DD column.

builder.add_column(GymHarnessColumnConfig(
    name="reply_2",
    evidence_column="reply_2_evidence",
    endpoint="http://127.0.0.1:39336",
    row_id_column="row_id",
    request_columns={"input": "user_reaction", "user": "row_id"},
    request_values={"metadata": {"turn": "2"}},
    timeout=330,
    queue_timeout=1800,
    max_parallel_requests=3,
))
```

Omit the plugin's per-call `session` initialization/cleanup: this service owns the conversation across columns. To restart the server between turns, finish the first DD stage, save its rows, restart this service with the same configuration and session root, then seed the follow-up DD stage from those rows. `tests/dd_flow.py` demonstrates that staged boundary.

## Persistence and ownership

- Each conversation owns a workspace and Pi native session file. Every turn opens a new Pi process, restores the exact committed history, and closes the process before publishing its receipt.
- A nonblocking filesystem lock excludes concurrent writers, including writers in another server process. `num_workers` must remain one; the lock does not establish distributed deployment support.
- The manifest publishes a turn only after its native session and response receipt have been flushed. Identical committed requests return the saved Responses object. Changed input, out-of-order turns, changed agent/model configuration, and edited native history are rejected.
- A crash between writing the receipt and publishing the manifest remains an ambiguous interrupted turn. It is not replayed. Cancellation, timeout, and process failure also leave an interrupted turn for inspection.
- The default conversation length is two turns; increase `max_turns` before starting a new run if needed. Final-turn receipts remain available for retries. Keep the session root until the workflow's retention policy removes the run.
- The configuration fingerprint includes the model endpoint. Moving a checkpoint to a different endpoint requires an explicit migration outside this example; an accidental endpoint/model change is rejected.
- The response and turn receipt preserve upstream agent observations, including evidence gaps. Local execution reports the upstream `no_sandbox_runtime` gap. DD's own model-usage summary excludes Pi's calls.

This extension currently uses local Pi with read/write/edit tools. Separate workspaces are not a sandbox isolation boundary. It does not enable upstream native sandbox sessions, `/run` verification, mid-tool checkpoint restoration, or distributed routing. Native agent-session initialization is rejected so a stale session cookie cannot silently select the upstream single-activation path. Sandbox adoption requires separate validation and is not implied by inheritance from `PiAgent`.

After the workflow drains, verify `/stats` reports zero active Pi processes and stop the service with Ctrl+C. Keep the configured session root for retries or later turns; the operator removes it when the run's retention period ends. Shutdown cancels active requests and closes their Pi processes. Client timeouts alone do not guarantee that the remote turn has stopped.

Slurm service placement, readiness, resource budgeting, and teardown require a separate runner integration. This example does not supply managed startup, sandbox provisioning, or Slurm acceptance evidence.

## Validation

```bash
make test
make smoke
```

`test` exercises public HTTP endpoints and the session store with a deterministic Pi protocol fixture. `smoke` runs real Pi against a deterministic local streaming model fixture, writes two independent workspaces, restarts the complete adapter process, restores both native histories, and verifies follow-up file changes. It uses no hosted inference or GPUs.

To include the installed DD plugin in that smoke test, point `DD_PYTHON` at its separate environment:

```bash
DD_PYTHON=/absolute/path/dd/.venv/bin/python make smoke
```

The DD smoke uses a deterministic expression for the user reaction, allowing exact assertions. Replace it with an LLM column in a generation recipe. Passing this check establishes integration and persistence, not coding-model quality or Slurm sandbox acceptance.
