# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner
from pydantic import ValidationError

from data_designer_generalist_agent_env.config import (
    GeneralistAgentEnvironmentColumnConfig,
    GeneralistAgentTaskColumnConfig,
)
from data_designer_generalist_agent_env.impl import (
    GeneralistAgentEnvironmentColumnGenerator,
    GeneralistAgentTaskColumnGenerator,
    build_environment_artifact,
    build_reference_answer,
    build_task_tuple,
    default_constraints,
    selected_tool_names,
)
from data_designer_generalist_agent_env.plugin import environment_plugin, task_plugin
from data_designer_generalist_agent_env.validation import verify_environment_tuple, verify_row_record


def generated_schema() -> dict:
    """Return a representative upstream-generated database schema."""
    return {
        "record_type": "trip_candidate",
        "primary_key": "record_id",
        "fields": [
            {"name": "record_id", "type": "string"},
            {"name": "name", "type": "string"},
            {"name": "summary", "type": "string"},
            {"name": "cost", "type": "integer"},
            {"name": "duration", "type": "integer"},
            {"name": "score", "type": "integer"},
            {"name": "tags", "type": "list[string]"},
            {"name": "attributes", "type": "object"},
        ],
        "attribute_fields": [
            {"name": "hotel_fit", "type": "integer"},
            {"name": "transport_risk", "type": "integer"},
            {"name": "restaurant_quality", "type": "integer"},
        ],
    }


def generated_records() -> list[dict]:
    """Return representative upstream-generated database records."""
    return [
        {
            "record_id": "trip-001",
            "name": "Museum Rail Plan",
            "summary": "Generated itinerary candidate with reliable transit and moderate cost.",
            "cost": 240,
            "duration": 3,
            "score": 92,
            "tags": ["reliable", "museum", "budget"],
            "attributes": {"hotel_fit": 88, "transport_risk": 12, "restaurant_quality": 82},
        },
        {
            "record_id": "trip-002",
            "name": "Luxury Dining Plan",
            "summary": "Generated itinerary candidate with high restaurant quality and higher cost.",
            "cost": 520,
            "duration": 3,
            "score": 97,
            "tags": ["restaurant", "premium", "ranked"],
            "attributes": {"hotel_fit": 90, "transport_risk": 18, "restaurant_quality": 96},
        },
        {
            "record_id": "trip-003",
            "name": "Compact Family Plan",
            "summary": "Generated itinerary candidate that balances family activities and reliable transport.",
            "cost": 180,
            "duration": 2,
            "score": 95,
            "tags": ["reliable", "family", "verified"],
            "attributes": {"hotel_fit": 91, "transport_risk": 10, "restaurant_quality": 80},
        },
    ]


def test_valid_plugin() -> None:
    assert_valid_plugin(environment_plugin)
    assert_valid_plugin(task_plugin)


def make_environment_generator(
    config: GeneralistAgentEnvironmentColumnConfig,
) -> GeneralistAgentEnvironmentColumnGenerator:
    """Create an environment generator instance without requiring a ResourceProvider."""
    generator = GeneralistAgentEnvironmentColumnGenerator.__new__(GeneralistAgentEnvironmentColumnGenerator)
    generator._config = config
    return generator


def make_task_generator(config: GeneralistAgentTaskColumnConfig) -> GeneralistAgentTaskColumnGenerator:
    """Create a task generator instance without requiring a ResourceProvider."""
    generator = GeneralistAgentTaskColumnGenerator.__new__(GeneralistAgentTaskColumnGenerator)
    generator._config = config
    return generator


def build_valid_task_tuple() -> dict:
    """Build a representative valid task tuple for validation tests."""
    task_config = GeneralistAgentTaskColumnConfig(
        name="agent_task",
        environment_column="agent_environment",
        difficulty="hard",
        required_tag="reliable",
    )
    environment = build_environment_artifact(
        "trip planning",
        {
            "goal": "plan a constrained itinerary",
            "constraints": ["moderate budget", "reliable transport", "strong local evidence"],
        },
        "goal: plan a constrained itinerary; constraints: moderate budget; reliable transport",
        {"notes": "family-friendly museums and restaurants"},
        generated_schema(),
        generated_records(),
        row_number=0,
    )
    return build_task_tuple(environment, task_config)


class TestGeneralistAgentEnvColumnConfig:
    def test_environment_config_required_columns_include_generated_schema_and_records(self) -> None:
        config = GeneralistAgentEnvironmentColumnConfig(
            name="agent_environment",
            task_topic_column="topic",
            task_constraints_column="constraints",
            database_schema_column="schema",
            database_records_column="records",
            context_columns=["notes", "persona"],
        )

        assert config.required_columns == ["topic", "constraints", "schema", "records", "notes", "persona"]
        assert config.side_effect_columns == []

    def test_task_config_requires_environment_column(self) -> None:
        config = GeneralistAgentTaskColumnConfig(
            name="agent_task",
            environment_column="agent_environment",
        )

        assert config.required_columns == ["agent_environment"]
        assert config.side_effect_columns == []

    def test_rejects_repeated_input_columns(self) -> None:
        with pytest.raises(ValidationError, match="must be distinct"):
            GeneralistAgentEnvironmentColumnConfig(
                name="agent_environment",
                task_topic_column="topic",
                task_constraints_column="constraints",
                database_schema_column="schema",
                database_records_column="records",
                context_columns=["constraints"],
            )

    def test_rejects_empty_topic_column(self) -> None:
        with pytest.raises(ValidationError, match="task_topic_column must not be empty"):
            GeneralistAgentEnvironmentColumnConfig(
                name="agent_environment",
                task_topic_column=" ",
                database_schema_column="schema",
                database_records_column="records",
            )

    def test_normalizes_task_required_tag(self) -> None:
        config = GeneralistAgentTaskColumnConfig(
            name="agent_task",
            environment_column="agent_environment",
            required_tag="  Reliable  ",
        )

        assert config.required_tag == "reliable"


class TestGeneralistAgentEnvHelpers:
    def test_tool_names_follow_difficulty(self) -> None:
        assert selected_tool_names("simple") == ["describe_schema", "list_records", "search_records", "get_record"]
        assert selected_tool_names("medium") == [
            "describe_schema",
            "list_records",
            "search_records",
            "get_record",
            "filter_records",
        ]
        assert selected_tool_names("hard") == [
            "describe_schema",
            "list_records",
            "search_records",
            "get_record",
            "filter_records",
            "rank_records",
        ]

    def test_reference_answer_is_verifier_optimal(self) -> None:
        environment_tuple = build_valid_task_tuple()

        validation = verify_environment_tuple(environment_tuple)

        assert validation.passed is True
        assert validation.verifier_passed is True
        assert validation.tools_passed is True
        assert validation.answer == environment_tuple["reference_answer"]
        assert environment_tuple["verifier"]["reference_solution_passed"] is True
        assert environment_tuple["task"]["constraints"]["required_tag"] == "reliable"

    def test_constraints_are_repaired_when_user_values_are_unsat(self) -> None:
        task_config = GeneralistAgentTaskColumnConfig(
            name="agent_task",
            environment_column="agent_environment",
            required_tag="rare",
            max_cost=1,
            min_score=100,
        )
        environment = build_environment_artifact(
            "debugging a build failure",
            {},
            "",
            {},
            generated_schema(),
            generated_records(),
            row_number=0,
        )
        task_tuple = build_task_tuple(environment, task_config)
        database = task_tuple["environment"]["database"]
        constraints = default_constraints(database, task_config)
        answer = build_reference_answer(database, constraints)

        assert constraints["repair_notes"]
        assert answer["record_id"] is not None


class TestGeneralistAgentEnvColumnGenerator:
    def test_two_step_environment_then_task_generation(self) -> None:
        source_df = pd.DataFrame(
            {
                "topic": ["trip planning"],
                "constraints": [
                    {
                        "goal": "build a three-day itinerary",
                        "constraints": ["hotels, restaurants, and attractions", "moderate budget"],
                        "success_criteria": ["reliable transport", "strong local evidence"],
                    }
                ],
                "schema": [generated_schema()],
                "records": [{"records": generated_records()}],
                "notes": ["family-friendly museums, moderate budget, reliable transport"],
            }
        )
        environment_config = GeneralistAgentEnvironmentColumnConfig(
            name="agent_environment",
            task_topic_column="topic",
            task_constraints_column="constraints",
            database_schema_column="schema",
            database_records_column="records",
            context_columns=["notes"],
        )
        task_config = GeneralistAgentTaskColumnConfig(
            name="agent_task",
            environment_column="agent_environment",
            difficulty="hard",
            required_tag="reliable",
        )
        environment_generator = make_environment_generator(environment_config)
        task_generator = make_task_generator(task_config)

        with_environment = environment_generator.generate(source_df)
        result = task_generator.generate(with_environment)
        environment_artifact = result.loc[0, "agent_environment"]
        task_tuple = result.loc[0, "agent_task"]
        validation = verify_environment_tuple(task_tuple)

        assert environment_artifact["schema_version"] == "generalist-agent-environment/v1"
        assert environment_artifact["environment"]["data_generation"]["mode"] == "generated_by_data_designer_columns"
        assert environment_artifact["environment"]["database_schema"]["record_type"] == "trip_candidate"
        assert environment_artifact["environment"]["database"][0]["record_id"] == "trip-001"
        assert task_tuple["schema_version"] == "generalist-agent-task/v1"
        assert task_tuple["task"]["constraints"]["required_tag"] == "reliable"
        assert "describe_schema" in task_tuple["solution"]["source"]
        assert validation.passed is True

    def test_generated_python_sources_pass_verifier(self) -> None:
        task_tuple = build_valid_task_tuple()

        validation = verify_environment_tuple(task_tuple)

        assert validation.passed is True
        assert validation.answer["record_id"]
        assert {check.name for check in validation.tool_checks} == set(selected_tool_names("hard"))
        assert [check.difficulty for check in validation.iteration_checks] == ["simple", "medium", "hard"]

    def test_row_record_validation_reads_named_output_column(self) -> None:
        task_tuple = build_valid_task_tuple()
        row = pd.Series({"agent_task": task_tuple})

        validation = verify_row_record(row, output_column="agent_task")

        assert validation.passed is True
        assert validation.verifier_passed is True

    def test_row_record_validation_reports_missing_tool_implementation(self) -> None:
        task_tuple = build_valid_task_tuple()
        broken_tuple = deepcopy(task_tuple)
        broken_tuple["tool_module_source"] = broken_tuple["tool_module_source"].replace(
            "def rank_records(",
            "def missing_rank_records(",
        )

        validation = verify_environment_tuple(broken_tuple)

        assert validation.passed is False
        assert validation.tools_passed is False
        assert any("rank_records" in error for error in validation.errors)

    def test_rejects_generated_records_missing_required_fields(self) -> None:
        source_df = pd.DataFrame(
            {
                "topic": ["trip planning"],
                "schema": [generated_schema()],
                "records": [{"records": [{"record_id": "bad"}]}],
            }
        )
        config = GeneralistAgentEnvironmentColumnConfig(
            name="agent_environment",
            task_topic_column="topic",
            database_schema_column="schema",
            database_records_column="records",
        )
        generator = make_environment_generator(config)

        with pytest.raises(ValueError, match="missing required fields"):
            generator.generate(source_df)

    def test_row_record_validation_accepts_parquet_restored_arrays(self, tmp_path: Path) -> None:
        task_tuple = build_valid_task_tuple()
        dataset_path = tmp_path / "dataset.parquet"
        pd.DataFrame({"agent_task": [task_tuple]}).to_parquet(dataset_path)
        restored = pd.read_parquet(dataset_path)

        validation = verify_row_record(restored.loc[0], output_column="agent_task")

        assert validation.passed is True
        assert validation.answer == task_tuple["reference_answer"]


class TestGeneralistAgentEnvPreviewIntegration:
    def test_preview_generates_environment_tuple(self, tmp_path: Path) -> None:
        seed_df = pd.DataFrame(
            {
                "topic": ["planning a travel itinerary"],
                "constraints": ["compare candidate plans by score, cost, and family suitability"],
                "schema": [generated_schema()],
                "records": [{"records": generated_records()}],
                "notes": ["family-friendly museums and restaurants"],
            }
        )

        builder = DataDesignerConfigBuilder()
        builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
        builder.add_column(
            name="agent_environment",
            column_type="generalist-agent-environment",
            task_topic_column="topic",
            task_constraints_column="constraints",
            database_schema_column="schema",
            database_records_column="records",
            context_columns=["notes"],
        )
        builder.add_column(
            name="agent_task",
            column_type="generalist-agent-task",
            environment_column="agent_environment",
            required_tag="family",
        )

        result = DataDesigner(artifact_path=tmp_path / "artifacts").preview(builder, num_records=1)

        assert result.dataset is not None
        environment_tuple = result.dataset.loc[0, "agent_task"]
        assert environment_tuple["task"]["constraints"]["required_tag"] == "family"
        assert environment_tuple["verifier"]["reference_solution_passed"] is True
