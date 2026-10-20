# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL candidates -> execution evidence -> DD judge; harness -> answer -> DD judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pandas as pd
from data_designer.config.column_configs import ExpressionColumnConfig, LLMJudgeColumnConfig, Score
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.interface.data_designer import DataDesigner

from data_designer_gym.config import GymHarnessColumnConfig, GymResourceColumnConfig


def build_workflow(args) -> DataDesignerConfigBuilder:
    seeds = pd.DataFrame(
        [
            {
                "row_id": f"toy-{database}",
                "database": database,
                "schema": "items(id INTEGER, name TEXT)",
                "question": "How many items are there?",
                "sql": "SELECT COUNT(*) FROM items",
                "reasoning": "Count rows in the items table.",
                "candidate_id": "candidate-1",
                "database_variant": database,
            }
            for database in ("original", "variant")
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    seed_path = args.output / "seeds.parquet"
    seeds.to_parquet(seed_path, index=False)
    builder = DataDesignerConfigBuilder().with_seed_dataset(LocalFileSeedSource(path=str(seed_path)))
    builder.add_model_config(
        ModelConfig(
            alias="judge",
            model=args.model,
            provider="smoke-provider",
            inference_parameters=ChatCompletionInferenceParams(temperature=0, max_tokens=512),
        )
    )
    builder.add_column(
        GymResourceColumnConfig(
            name="execution",
            endpoint=args.resource_endpoint,
            route="/execute",
            row_id_column="row_id",
            request_columns={"database": "database", "sql": "sql"},
            provenance_columns=["candidate_id", "database_variant", "schema"],
            max_parallel_requests=2,
        )
    )
    builder.add_column(
        LLMJudgeColumnConfig(
            name="sql_judgment",
            scores=[
                Score(
                    name="correct",
                    description="Does the candidate SQL answer the question?",
                    options={0: "No", 1: "Yes"},
                )
            ],
            model_alias="judge",
            prompt="Question: {{ question }}\nSchema: {{ schema }}\nSQL: {{ sql }}\nReasoning: {{ reasoning }}\nExecution: {{ execution }}\nAssess whether this SQL answers the question. Execution success alone does not prove correctness.",
        )
    )
    tools = [
        {
            "type": "function",
            "name": "query_database",
            "description": "Execute a read-only SQLite query in the initialized database.",
            "parameters": {
                "type": "object",
                "properties": {"database": {"type": "string"}, "sql": {"type": "string"}},
                "required": ["database", "sql"],
                "additionalProperties": False,
            },
            "strict": False,
        }
    ]
    builder.add_column(
        ExpressionColumnConfig(
            name="harness_input",
            expr="Use query_database with database={{ database }} to answer: {{ question }} Schema: {{ schema }}. Call the tool before answering. After the tool result, return only the integer count.",
        )
    )
    builder.add_column(
        GymHarnessColumnConfig(
            name="answer",
            evidence_column="harness_evidence",
            endpoint=args.harness_endpoint,
            row_id_column="row_id",
            request_columns={"input": "harness_input"},
            request_values={"tools": tools, "max_output_tokens": 256, "temperature": 0},
            provenance_columns=["candidate_id", "database_variant"],
            max_parallel_requests=2,
            session={
                "initialize": [
                    {
                        "endpoint": args.resource_endpoint,
                        "route": "/seed_session",
                        "request_columns": {"database": "database"},
                    }
                ],
                "cleanup": [{"endpoint": args.resource_endpoint, "route": "/cleanup"}],
            },
        )
    )
    builder.add_column(
        LLMJudgeColumnConfig(
            name="answer_judgment",
            scores=[
                Score(
                    name="correct",
                    description="Does the answer agree with the execution evidence?",
                    options={0: "No", 1: "Yes"},
                )
            ],
            model_alias="judge",
            prompt="Question: {{ question }}\nExecution evidence: {{ execution }}\nHarness answer: {{ answer }}\nAssess whether the final answer agrees with the execution evidence and answers the question.",
        )
    )
    builder.add_column(
        ExpressionColumnConfig(
            name="accepted",
            expr="{{ execution.status == 'completed' and harness_evidence.status == 'completed' and (sql_judgment.correct.score | int) == 1 and (answer_judgment.correct.score | int) == 1 }}",
            dtype="bool",
        )
    )
    return builder


def inspect_result(dataset: pd.DataFrame, output: Path, endpoint: str, baseline: dict) -> dict:
    records = dataset.to_dict(orient="records")
    assert len(records) == 2
    for row in records:
        expected = 3 if row["database_variant"] == "original" else 4
        assert json.loads(row["execution"]["response_json"])["rows"] == [[expected]]
        assert str(row["answer"]).strip() == str(expected)
        assert row["harness_evidence"]["status"] == "completed"
        artifact = list(output.glob(f"artifacts/*/{row['harness_evidence']['artifact']}"))
        assert len(artifact) == 1
        response = json.loads(artifact[0].read_text())["response"]
        assert any(item["type"] == "function_call" for item in response["output"])
        assert any(item["type"] == "function_call_output" for item in response["output"])
        assert row["sql_judgment"] is not None and row["answer_judgment"] is not None
        assert row["accepted"]
    stats = httpx.get(endpoint + "/stats", timeout=10).json()
    assert stats == {
        "active_sessions": 0,
        "initialized": baseline["initialized"] + 2,
        "cleaned": baseline["cleaned"] + 2,
    }
    summary = {
        "rows": len(records),
        "answers": [row["answer"] for row in records],
        "tool_assisted": True,
        "downstream_judges": True,
        "sessions": stats,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-endpoint", default="http://127.0.0.1:8765")
    parser.add_argument("--harness-endpoint", default="http://127.0.0.1:8767")
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = httpx.get(args.resource_endpoint + "/stats", timeout=10).json()
    builder = build_workflow(args)
    builder.write_config(args.output / "workflow.yaml")
    builder = DataDesignerConfigBuilder.from_config(args.output / "workflow.yaml")
    provider = ModelProvider(
        name="smoke-provider", endpoint=args.model_endpoint, provider_type="openai", api_key=args.api_key_env
    )
    result = DataDesigner(artifact_path=args.output / "artifacts", model_providers=[provider]).create(
        builder, num_records=2, dataset_name="toy-sql"
    )
    dataset = result.load_dataset()
    dataset.to_json(args.output / "results.jsonl", orient="records", lines=True)
    dataset[dataset["accepted"]].to_json(args.output / "accepted.jsonl", orient="records", lines=True)
    print(json.dumps(inspect_result(dataset, args.output, args.resource_endpoint, baseline)))


if __name__ == "__main__":
    main()
