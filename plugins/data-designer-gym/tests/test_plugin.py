# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd
import pytest
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner
from data_designer.plugins.plugin import PluginType
from pydantic import ValidationError

from data_designer_gym.cli import run_cli
from data_designer_gym.config import GymTaskProcessorConfig
from data_designer_gym.conversion import (
    SCENARIO_ID_KEY,
    gym_tasks_from_dataframe,
    normalize_rollouts,
    read_jsonl,
    scenario_from_row,
    scenario_from_task,
    scenario_to_gym_task,
    write_jsonl,
)
from data_designer_gym.impl import GymTaskProcessor
from data_designer_gym.models import GymTask
from data_designer_gym.plugin import plugin


@pytest.fixture()
def tool() -> dict:
    return {
        "type": "function",
        "name": "email_send_email",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {"recipient": {"type": "string"}},
            "required": ["recipient"],
            "additionalProperties": False,
        },
        "strict": False,
        "database": "email",
    }


@pytest.fixture()
def seed_row(tool: dict) -> dict:
    return {
        "seed_id": 7,
        "user_query": "Send an email to Pat.",
        "system_prompt": "You are a workplace assistant.",
        "tools_json": json.dumps([tool]),
        "reference_plan": {
            "tool_calls": [{"name": "email_send_email", "arguments": '{"recipient":"pat@example.com"}'}],
            "is_valid": True,
        },
    }


@pytest.fixture()
def generic_row(seed_row: dict) -> dict:
    row = dict(seed_row)
    row["include"] = True
    row["gym_task"] = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_query"]},
            ],
            "tools": [tool for tool in json.loads(row["tools_json"])],
        },
        "environment_name": "example_environment",
        "ground_truth": row["reference_plan"]["tool_calls"],
    }
    return row


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)
    assert plugin.plugin_type == PluginType.PROCESSOR
    assert plugin.impl_cls is GymTaskProcessor
    assert issubclass(plugin.impl_cls, Processor)


@pytest.mark.parametrize("column", ["task_column", "scenario_id_column", "include"])
def test_config_rejects_empty_column_names(column: str) -> None:
    kwargs = {"task_column": "gym_task", column: ""}
    with pytest.raises(ValidationError, match="column paths must contain non-empty dot-separated names"):
        GymTaskProcessorConfig(name="gym_tasks", **kwargs)


def test_config_requires_one_task_mode() -> None:
    with pytest.raises(ValidationError, match="exactly one of task_column or messages"):
        GymTaskProcessorConfig(name="gym_tasks")
    with pytest.raises(ValidationError, match="exactly one of task_column or messages"):
        GymTaskProcessorConfig(
            name="gym_tasks",
            task_column="gym_task",
            messages=[{"role": "user", "content_column": "user_query"}],
        )


def test_processor_converts_only_included_rows(generic_row: dict) -> None:
    rejected = generic_row | {"include": False}
    config = GymTaskProcessorConfig(
        name="gym_tasks",
        task_column="gym_task",
        include="include",
        provenance_columns=["seed_id"],
        scenario_namespace="email",
    )
    tasks = gym_tasks_from_dataframe(pd.DataFrame([generic_row, rejected]), config)

    assert len(tasks) == 1
    task = tasks.iloc[0].to_dict()
    assert task[SCENARIO_ID_KEY].startswith("email-")
    assert task["_ng_task_index"] == task["id"]
    assert task["responses_create_params"]["input"][1]["content"] == "Send an email to Pat."
    assert task["environment_name"] == "example_environment"
    assert task["_dd_provenance"] == {"seed_id": 7}


def test_processor_runs_in_workflow_built_from_scratch(tmp_path: Path, seed_row: dict) -> None:
    builder = DataDesignerConfigBuilder().with_seed_dataset(DataFrameSeedSource(df=pd.DataFrame([seed_row])))
    builder.add_column(ExpressionColumnConfig(name="request", expr="{{ user_query }}"))
    builder.add_processor(
        GymTaskProcessorConfig(
            name="gym_tasks",
            messages=[
                {"role": "system", "content_column": "system_prompt"},
                {"role": "user", "content_column": "request"},
            ],
            tools_column="tools_json",
            tool_fields=["type", "name", "description", "parameters", "strict"],
            task_columns={"ground_truth": "reference_plan.tool_calls"},
            task_values={"environment_name": "example_environment"},
            include="reference_plan.is_valid",
            provenance_columns=["seed_id"],
        )
    )

    result = DataDesigner(artifact_path=tmp_path / "artifacts").preview(builder, num_records=1)

    assert result.processor_artifacts is not None
    artifact = result.processor_artifacts["gym_tasks"][0]
    task = json.loads(artifact["task_json"])
    assert task[SCENARIO_ID_KEY] == artifact["scenario_id"]
    assert "database" not in task["responses_create_params"]["tools"][0]
    assert task["ground_truth"] == seed_row["reference_plan"]["tool_calls"]
    assert json.loads(artifact["scenario_json"])["task"]["environment_name"] == "example_environment"


def test_cli_exports_and_ingests_scenario(tmp_path: Path, generic_row: dict) -> None:
    scenario = scenario_from_row(generic_row, GymTaskProcessorConfig(name="gym_tasks", task_column="gym_task"))
    assert scenario is not None
    scenarios_path = write_jsonl(tmp_path / "scenarios.jsonl", [scenario.model_dump(mode="json")])
    tasks_path = tmp_path / "tasks.jsonl"
    assert run_cli(["export", str(scenarios_path), "--output", str(tasks_path)]) == 0
    task = read_jsonl(tasks_path)[0]

    rollouts_path = write_jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {
                "_ng_task_index": task["_ng_task_index"],
                "_ng_rollout_index": 0,
                "reward": 1.0,
                "response": {"output": []},
            }
        ],
    )
    normalized_path = tmp_path / "normalized.jsonl"
    assert (
        run_cli(
            [
                "ingest",
                "--tasks",
                str(tasks_path),
                "--rollouts",
                str(rollouts_path),
                "--output",
                str(normalized_path),
            ]
        )
        == 0
    )

    normalized = read_jsonl(normalized_path)[0]
    assert normalized["scenario_id"] == task[SCENARIO_ID_KEY]
    assert normalized["status"] == "completed"
    assert normalized["reward"] == 1.0


def test_normalize_rollouts_includes_failure_sidecar(tmp_path: Path, generic_row: dict) -> None:
    scenario = scenario_from_row(generic_row, GymTaskProcessorConfig(name="gym_tasks", task_column="gym_task"))
    assert scenario is not None
    task = scenario_to_gym_task(scenario)
    tasks_path = write_jsonl(tmp_path / "tasks.jsonl", [task])
    rollouts_path = write_jsonl(tmp_path / "rollouts.jsonl", [])
    failures_path = write_jsonl(
        tmp_path / "rollouts_failures.jsonl",
        [
            {
                "_ng_task_index": task["_ng_task_index"],
                "_ng_rollout_index": 0,
                "_ng_failure_class": "timeout",
                "_ng_failure_terminal": True,
            }
        ],
    )

    output = normalize_rollouts(tasks_path, rollouts_path, tmp_path / "normalized.jsonl", failures_path=failures_path)

    normalized = read_jsonl(output)[0]
    assert normalized["status"] == "failed"
    assert normalized["failure_class"] == "timeout"
    assert normalized["failure_terminal"] is True


@pytest.mark.parametrize(
    "managed_field",
    ["_dd_provenance", "_dd_scenario_id", "_ng_rollout_index", "_ng_task_index"],
)
def test_gym_task_rejects_plugin_managed_fields(managed_field: str) -> None:
    with pytest.raises(ValidationError, match="plugin-managed fields"):
        GymTask(responses_create_params={}, **{managed_field: "reserved"})


def test_generic_task_preserves_environment_metadata() -> None:
    task = GymTask(
        responses_create_params={"input": [{"role": "user", "content": "Write about research."}]},
        instructions=[{"instruction_id": "keywords:existence", "keywords": ["research"]}],
        llm_judge=[{"uid": 1, "content": "Is the response professional?"}],
    )

    exported = scenario_to_gym_task(scenario_from_task(task, namespace="verifif"))

    assert exported["instructions"][0]["instruction_id"] == "keywords:existence"
    assert exported["llm_judge"][0]["uid"] == 1
