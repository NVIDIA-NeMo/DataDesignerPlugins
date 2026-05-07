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

from data_designer_generalist_agent_env.config import GeneralistAgentEnvColumnConfig
from data_designer_generalist_agent_env.impl import (
    GeneralistAgentEnvColumnGenerator,
    build_environment_tuple,
    build_reference_answer,
    default_constraints,
    selected_tool_names,
)
from data_designer_generalist_agent_env.plugin import plugin
from data_designer_generalist_agent_env.validation import verify_environment_tuple, verify_row_record


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)


def make_generator(config: GeneralistAgentEnvColumnConfig) -> GeneralistAgentEnvColumnGenerator:
    """Create a generator instance without requiring a ResourceProvider."""
    generator = GeneralistAgentEnvColumnGenerator.__new__(GeneralistAgentEnvColumnGenerator)
    generator._config = config
    return generator


class TestGeneralistAgentEnvColumnConfig:
    def test_required_columns_include_category_and_context(self) -> None:
        config = GeneralistAgentEnvColumnConfig(
            name="agent_env",
            task_category_column="category",
            context_columns=["constraints", "persona", "constraints"],
        )

        assert config.required_columns == ["category", "constraints", "persona"]
        assert config.side_effect_columns == []

    def test_column_emoji(self) -> None:
        config = GeneralistAgentEnvColumnConfig(name="agent_env", task_category_column="category")

        assert config.get_column_emoji() == "🧰"

    def test_rejects_empty_category_column(self) -> None:
        with pytest.raises(ValidationError, match="task_category_column must not be empty"):
            GeneralistAgentEnvColumnConfig(name="agent_env", task_category_column=" ")

    def test_rejects_category_repeated_as_context(self) -> None:
        with pytest.raises(ValidationError, match="context_columns must not repeat task_category_column"):
            GeneralistAgentEnvColumnConfig(
                name="agent_env",
                task_category_column="category",
                context_columns=["category"],
            )

    def test_normalizes_required_tag(self) -> None:
        config = GeneralistAgentEnvColumnConfig(
            name="agent_env",
            task_category_column="category",
            required_tag="  Family  ",
        )

        assert config.required_tag == "family"


class TestGeneralistAgentEnvHelpers:
    def test_tool_names_follow_difficulty(self) -> None:
        assert selected_tool_names("simple") == ["list_records", "search_records", "get_record"]
        assert selected_tool_names("medium") == ["list_records", "search_records", "get_record", "filter_records"]
        assert selected_tool_names("hard") == [
            "list_records",
            "search_records",
            "get_record",
            "filter_records",
            "rank_records",
        ]

    def test_reference_answer_is_verifier_optimal(self) -> None:
        config = GeneralistAgentEnvColumnConfig(
            name="agent_env",
            task_category_column="category",
            difficulty="hard",
            required_tag="family",
        )
        environment_tuple = build_environment_tuple(
            "planning a travel itinerary",
            {"city": "Seoul", "budget": "1200"},
            config,
            row_number=0,
        )

        validation = verify_environment_tuple(environment_tuple)

        assert validation.passed is True
        assert validation.verifier_passed is True
        assert validation.tools_passed is True
        assert validation.answer == environment_tuple["reference_answer"]
        assert environment_tuple["verifier"]["reference_solution_passed"] is True
        assert environment_tuple["task"]["constraints"]["required_tag"] == "family"

    def test_constraints_are_repaired_when_user_values_are_unsat(self) -> None:
        config = GeneralistAgentEnvColumnConfig(
            name="agent_env",
            task_category_column="category",
            required_tag="rare",
            max_cost=1,
            min_score=100,
        )
        environment_tuple = build_environment_tuple("debugging a build failure", {}, config, row_number=0)
        database = environment_tuple["environment"]["database"]
        constraints = default_constraints(database, config)
        answer = build_reference_answer(database, constraints)

        assert constraints["repair_notes"]
        assert answer["record_id"] is not None


class TestGeneralistAgentEnvColumnGenerator:
    def test_generate_creates_environment_tuple(self) -> None:
        source_df = pd.DataFrame(
            {
                "category": ["planning a travel itinerary"],
                "constraints": ["visit museums and stay under a moderate budget"],
            }
        )
        config = GeneralistAgentEnvColumnConfig(
            name="agent_env",
            task_category_column="category",
            context_columns=["constraints"],
        )
        generator = make_generator(config)

        result = generator.generate(source_df)
        environment_tuple = result.loc[0, "agent_env"]

        assert environment_tuple["schema_version"] == "generalist-agent-env/v1"
        assert environment_tuple["environment"]["sandbox"]["base_tools"] == ["bash", "search"]
        assert environment_tuple["environment"]["database_record_count"] == config.database_size
        assert {tool["name"] for tool in environment_tuple["tools"]} == set(selected_tool_names("hard"))
        assert environment_tuple["task"]["difficulty"] == "hard"
        assert [iteration["difficulty"] for iteration in environment_tuple["task_iterations"]] == [
            "simple",
            "medium",
            "hard",
        ]
        assert all(iteration["reference_solution_passed"] for iteration in environment_tuple["task_iterations"])
        assert environment_tuple["solution"]["restrictions"] == [
            "may call synthesized tool functions",
            "may perform local logical computation",
            "must not directly access the sandbox database",
        ]

    def test_generated_python_sources_pass_verifier(self) -> None:
        source_df = pd.DataFrame({"category": ["planning a travel itinerary"]})
        config = GeneralistAgentEnvColumnConfig(name="agent_env", task_category_column="category")
        generator = make_generator(config)
        result = generator.generate(source_df)

        validation = verify_environment_tuple(result.loc[0, "agent_env"])

        assert validation.passed is True
        assert validation.answer["record_id"]
        assert {check.name for check in validation.tool_checks} == set(selected_tool_names("hard"))
        assert [check.difficulty for check in validation.iteration_checks] == ["simple", "medium", "hard"]

    def test_row_record_validation_reads_named_output_column(self) -> None:
        source_df = pd.DataFrame({"category": ["planning a travel itinerary"]})
        config = GeneralistAgentEnvColumnConfig(name="agent_env", task_category_column="category")
        generator = make_generator(config)
        result = generator.generate(source_df)

        validation = verify_row_record(result.loc[0], output_column="agent_env")

        assert validation.passed is True
        assert validation.verifier_passed is True

    def test_row_record_validation_reports_missing_tool_implementation(self) -> None:
        config = GeneralistAgentEnvColumnConfig(name="agent_env", task_category_column="category")
        environment_tuple = build_environment_tuple("planning a travel itinerary", {}, config, row_number=0)
        broken_tuple = deepcopy(environment_tuple)
        broken_tuple["tool_module_source"] = broken_tuple["tool_module_source"].replace(
            "def rank_records(",
            "def missing_rank_records(",
        )

        validation = verify_environment_tuple(broken_tuple)

        assert validation.passed is False
        assert validation.tools_passed is False
        assert any("rank_records" in error for error in validation.errors)


class TestGeneralistAgentEnvPreviewIntegration:
    def test_preview_generates_environment_tuple(self, tmp_path: Path) -> None:
        seed_df = pd.DataFrame(
            {
                "category": ["planning a travel itinerary"],
                "constraints": ["compare candidate plans by score, cost, and family suitability"],
            }
        )

        builder = DataDesignerConfigBuilder()
        builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
        builder.add_column(
            name="agent_env",
            column_type="generalist-agent-env",
            task_category_column="category",
            context_columns=["constraints"],
            required_tag="family",
        )

        result = DataDesigner(artifact_path=tmp_path / "artifacts").preview(builder, num_records=1)

        assert result.dataset is not None
        environment_tuple = result.dataset.loc[0, "agent_env"]
        assert environment_tuple["task"]["constraints"]["required_tag"] == "family"
        assert environment_tuple["verifier"]["reference_solution_passed"] is True
