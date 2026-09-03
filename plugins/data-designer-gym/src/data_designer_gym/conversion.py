# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from data_designer_gym.config import GymTaskProcessorConfig
from data_designer_gym.models import GymScenario, GymTask

TASK_INDEX_KEY = "_ng_task_index"
ROLLOUT_INDEX_KEY = "_ng_rollout_index"
FAILURE_CLASS_KEY = "_ng_failure_class"
FAILURE_TERMINAL_KEY = "_ng_failure_terminal"
SCENARIO_ID_KEY = "_dd_scenario_id"
PROVENANCE_KEY = "_dd_provenance"


def _json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must contain valid JSON") from exc
    return _json_compatible(value, field_name)


def _json_compatible(value: Any, field_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, default=_json_default))
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain JSON-compatible data") from exc


def _json_default(value: Any) -> Any:
    for method_name in ("model_dump", "tolist", "item"):
        method = getattr(value, method_name, None)
        if callable(method):
            return method()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _source_value(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    parts = path.split(".")
    for index, part in enumerate(parts):
        if index and isinstance(value, str):
            value = _json_value(value, ".".join(parts[:index]))
        if isinstance(value, Mapping):
            if part not in value:
                raise ValueError(f"missing source path: {path}")
            value = value[part]
            continue
        value = getattr(value, part, None)
        if value is None:
            raise ValueError(f"missing source path: {path}")
    return value


def gym_task_from_row(row: Mapping[str, Any], config: GymTaskProcessorConfig) -> GymTask:
    """Build or load one Gym task from a Data Designer row."""
    if config.task_column is not None:
        return GymTask.model_validate(_json_value(_source_value(row, config.task_column), config.task_column))

    responses_create_params = _json_compatible(config.response_params, "response_params")
    responses_create_params["input"] = [
        {
            "role": message.role,
            "content": _json_compatible(
                _source_value(row, message.content_column),
                message.content_column,
            ),
        }
        for message in config.messages
    ]
    if config.tools_column is not None:
        tools = _json_value(_source_value(row, config.tools_column), config.tools_column)
        if not isinstance(tools, list) or not all(isinstance(tool, Mapping) for tool in tools):
            raise ValueError(f"{config.tools_column} must contain a JSON list of objects")
        if config.tool_fields is not None:
            tools = [{field: tool[field] for field in config.tool_fields if field in tool} for tool in tools]
        responses_create_params["tools"] = tools

    task = _json_compatible(config.task_values, "task_values")
    task.update(
        {field: _json_compatible(_source_value(row, path), path) for field, path in config.task_columns.items()}
    )
    task["responses_create_params"] = responses_create_params
    return GymTask.model_validate(task)


def _stable_scenario_id(namespace: str, task: GymTask) -> str:
    payload = json.dumps(task.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{namespace}-{digest}"


def scenario_from_task(
    task: GymTask | Mapping[str, Any],
    *,
    namespace: str = "gym",
    scenario_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> GymScenario:
    """Create a canonical scenario from any Gym-native task."""
    validated_task = task if isinstance(task, GymTask) else GymTask.model_validate(_json_value(task, "task"))
    resolved_id = scenario_id.strip() if scenario_id is not None else _stable_scenario_id(namespace, validated_task)
    if not resolved_id:
        raise ValueError("scenario_id must not be empty")
    return GymScenario(
        scenario_id=resolved_id,
        task=validated_task,
        provenance=dict(provenance or {}),
    )


def scenario_from_row(row: Mapping[str, Any], config: GymTaskProcessorConfig) -> GymScenario | None:
    """Validate one Data Designer row and return its canonical scenario."""
    if config.include is not None and not bool(_source_value(row, config.include)):
        return None

    task = gym_task_from_row(row, config)
    scenario_id = str(_source_value(row, config.scenario_id_column)) if config.scenario_id_column else None
    provenance = {
        column: _json_compatible(value, column)
        for column in config.provenance_columns
        if (value := _source_value(row, column)) is not None
    }
    return scenario_from_task(
        task,
        namespace=config.scenario_namespace,
        scenario_id=scenario_id,
        provenance=provenance,
    )


def scenarios_from_dataframe(data: pd.DataFrame, config: GymTaskProcessorConfig) -> list[GymScenario]:
    """Convert accepted rows in a DataFrame to canonical scenarios."""
    scenarios = []
    for row in data.to_dict(orient="records"):
        scenario = scenario_from_row(row, config)
        if scenario is not None:
            scenarios.append(scenario)
    return scenarios


def scenario_to_gym_task(scenario: GymScenario) -> dict[str, Any]:
    """Add stable plugin identity to a Gym-native task."""
    task = scenario.task.model_dump(mode="json", exclude_none=True)
    task_index = int(hashlib.sha256(scenario.scenario_id.encode()).hexdigest()[:15], 16)
    task.setdefault("id", task_index)
    task[TASK_INDEX_KEY] = task_index
    task[SCENARIO_ID_KEY] = scenario.scenario_id
    if scenario.provenance:
        task[PROVENANCE_KEY] = scenario.provenance
    return task


def gym_tasks_from_dataframe(data: pd.DataFrame, config: GymTaskProcessorConfig) -> pd.DataFrame:
    """Convert accepted Data Designer rows to Gym task records."""
    tasks = [scenario_to_gym_task(scenario) for scenario in scenarios_from_dataframe(data, config)]
    return pd.DataFrame(tasks)


def gym_task_artifacts_from_dataframe(data: pd.DataFrame, config: GymTaskProcessorConfig) -> pd.DataFrame:
    """Serialize Gym tasks losslessly for Data Designer's Parquet processor storage."""
    records = []
    for scenario in scenarios_from_dataframe(data, config):
        task = scenario_to_gym_task(scenario)
        records.append(
            {
                "scenario_id": scenario.scenario_id,
                TASK_INDEX_KEY: task[TASK_INDEX_KEY],
                "scenario_json": scenario.model_dump_json(),
                "task_json": json.dumps(task),
            }
        )
    return pd.DataFrame(
        records,
        columns=["scenario_id", TASK_INDEX_KEY, "scenario_json", "task_json"],
    )


def scenarios_from_artifacts(data: pd.DataFrame) -> list[GymScenario]:
    """Decode canonical scenarios from Data Designer processor artifacts."""
    if "scenario_json" not in data.columns:
        raise ValueError("Gym task processor artifacts must contain scenario_json")
    return [GymScenario.model_validate(_json_value(value, "scenario_json")) for value in data["scenario_json"]]


def gym_tasks_from_artifacts(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Decode Gym tasks from Data Designer processor artifacts."""
    if "task_json" not in data.columns:
        raise ValueError("Gym task processor artifacts must contain task_json")
    return [_json_value(value, "task_json") for value in data["task_json"]]


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read and validate a JSONL object stream."""
    input_path = Path(path)
    if not input_path.is_file():
        raise ValueError(f"JSONL file does not exist: {input_path}")
    records = []
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {input_path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} of {input_path} must contain a JSON object")
            records.append(record)
    return records


def write_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write JSON objects as one record per line."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), default=str) + "\n")
    return output_path


def export_scenarios(input_path: Path | str, output_path: Path | str) -> Path:
    """Export canonical scenario JSONL as native Gym task JSONL."""
    scenarios = [GymScenario.model_validate(record) for record in read_jsonl(input_path)]
    return write_jsonl(output_path, (scenario_to_gym_task(scenario) for scenario in scenarios))


def normalize_rollouts(
    tasks_path: Path | str,
    rollouts_path: Path | str,
    output_path: Path | str,
    *,
    failures_path: Path | str | None = None,
) -> Path:
    """Join Gym rollouts and failures back to stable scenario IDs."""
    tasks = read_jsonl(tasks_path)
    tasks_by_index = {task.get(TASK_INDEX_KEY, task.get("id")): task for task in tasks}
    if None in tasks_by_index:
        raise ValueError("every task must contain _ng_task_index or id")

    runtime_records = [("completed", record) for record in read_jsonl(rollouts_path)]
    if failures_path is not None and Path(failures_path).exists():
        runtime_records.extend(("failed", record) for record in read_jsonl(failures_path))

    normalized = []
    for status, rollout in runtime_records:
        task_index = rollout.get(TASK_INDEX_KEY)
        task = tasks_by_index.get(task_index)
        if task is None:
            raise ValueError(f"rollout references unknown task index: {task_index!r}")
        scenario_id = task.get(SCENARIO_ID_KEY)
        if not scenario_id:
            raise ValueError(f"task {task_index!r} does not contain {SCENARIO_ID_KEY}")
        normalized.append(
            {
                "scenario_id": scenario_id,
                "task_index": task_index,
                "rollout_index": rollout.get(ROLLOUT_INDEX_KEY),
                "status": status,
                "reward": rollout.get("reward"),
                "provenance": task.get(PROVENANCE_KEY, {}),
                "agent_ref": rollout.get("agent_ref", task.get("agent_ref")),
                "failure_class": rollout.get(FAILURE_CLASS_KEY),
                "failure_terminal": rollout.get(FAILURE_TERMINAL_KEY, False),
                "task": task,
                "rollout": rollout,
            }
        )
    return write_jsonl(output_path, normalized)
