# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from pprint import pformat
from typing import TYPE_CHECKING, Any

from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from data_designer_generalist_agent_env.config import Difficulty, GeneralistAgentEnvColumnConfig

if TYPE_CHECKING:
    import pandas as pd

BASE_SANDBOX_TOOLS = ["bash", "search"]
BASE_TAGS = [
    "budget",
    "reliable",
    "fast",
    "verified",
    "flexible",
    "local",
    "safe",
    "ranked",
]
DIFFICULTY_ORDER: list[Difficulty] = ["simple", "medium", "hard"]

DATABASE_SCHEMA = {
    "record_id": "Stable row-local identifier.",
    "name": "Human-readable option name.",
    "category": "Task category supplied by the seed row.",
    "summary": "Short synthesized description for search.",
    "cost": "Integer cost proxy; lower is better.",
    "duration": "Integer duration proxy.",
    "score": "Integer quality score from 55 to 100; higher is better.",
    "tags": "Searchable task-specific labels.",
    "source_values": "Context columns copied from the seed row.",
}

TOOL_FUNCTION_SOURCES = {
    "list_records": '''
def list_records():
    """Return every record in the sandbox database."""
    return [dict(record) for record in DATABASE]
''',
    "search_records": '''
def search_records(query="", max_results=10):
    """Search database records by name, summary, category, or tag."""
    needle = str(query or "").casefold()
    limit = max(0, int(max_results))
    matches = []
    for record in DATABASE:
        haystack = " ".join(
            [
                str(record.get("name", "")),
                str(record.get("summary", "")),
                str(record.get("category", "")),
                " ".join(str(tag) for tag in record.get("tags", [])),
            ],
        ).casefold()
        if not needle or needle in haystack:
            matches.append(dict(record))
    return matches[:limit]
''',
    "get_record": '''
def get_record(record_id):
    """Return one record by id, or None when the id is unknown."""
    for record in DATABASE:
        if str(record.get("record_id")) == str(record_id):
            return dict(record)
    return None
''',
    "filter_records": '''
def filter_records(max_cost=None, min_score=None, required_tag=None):
    """Filter records by cost, score, and tag constraints."""
    matches = []
    for record in DATABASE:
        if max_cost is not None and int(record["cost"]) > int(max_cost):
            continue
        if min_score is not None and int(record["score"]) < int(min_score):
            continue
        if required_tag is not None and str(required_tag) not in record.get("tags", []):
            continue
        matches.append(dict(record))
    return matches
''',
    "rank_records": '''
def rank_records(records=None, metric="score", descending=True):
    """Rank supplied records, or all database records, by a numeric metric."""
    source = DATABASE if records is None else records
    return sorted(
        [dict(record) for record in source],
        key=lambda record: int(record.get(metric, 0)),
        reverse=bool(descending),
    )
''',
}

TOOL_DESCRIPTIONS = {
    "list_records": "Inspect all rows in the hidden sandbox database.",
    "search_records": "Retrieve category-relevant records through a search-style interface.",
    "get_record": "Fetch one database record by identifier.",
    "filter_records": "Apply verifier-aligned constraints without exposing the database directly.",
    "rank_records": "Rank candidate records for the final combinatorial selection step.",
}


def normalize_cell(value: object) -> str:
    """Normalize one pandas cell into a stable text value.

    Args:
        value: Cell value from a seed row.

    Returns:
        A stripped string, or an empty string for null-like values.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def slugify(value: str, fallback: str) -> str:
    """Convert text into a stable lowercase identifier fragment.

    Args:
        value: Input text.
        fallback: Value to use when no identifier characters remain.

    Returns:
        A slug containing lowercase letters, digits, and hyphens.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback


def stable_int(seed: str, modulo: int) -> int:
    """Hash text into a deterministic integer range.

    Args:
        seed: Hash seed.
        modulo: Exclusive upper bound.

    Returns:
        A deterministic integer in ``[0, modulo)``.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def unique_values(values: list[str]) -> list[str]:
    """Return values with duplicates removed while preserving order.

    Args:
        values: Candidate values.

    Returns:
        De-duplicated values.
    """
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def context_tags(category: str, context_values: dict[str, str], required_tag: str | None) -> list[str]:
    """Build a task-specific tag vocabulary.

    Args:
        category: Seed task category.
        context_values: Row context values.
        required_tag: Optional tag that must be present in the database.

    Returns:
        A non-empty tag list used to populate synthesized records.
    """
    seed_text = " ".join([category, *context_values.values()])
    words = [word for word in re.findall(r"[a-z0-9]+", seed_text.lower()) if len(word) > 3]
    tags = unique_values([required_tag or "", *words[:6], *BASE_TAGS])
    return tags[:12]


def build_context_summary(context_values: dict[str, str]) -> str:
    """Summarize row context for record descriptions.

    Args:
        context_values: Context columns extracted from the seed row.

    Returns:
        Compact text suitable for generated summaries.
    """
    if not context_values:
        return "seed category only"
    return "; ".join(f"{name}: {value}" for name, value in context_values.items() if value) or "empty context"


def build_database(
    category: str,
    context_values: dict[str, str],
    database_size: int,
    required_tag: str | None,
) -> list[dict[str, Any]]:
    """Synthesize a row-local sandbox database.

    Args:
        category: Seed task category.
        context_values: Optional context copied from the input row.
        database_size: Number of database records to create.
        required_tag: Optional tag that must be inserted into at least one record.

    Returns:
        JSON-compatible database records.
    """
    category_slug = slugify(category, "task")
    seed_context = json.dumps(context_values, sort_keys=True)
    tags = context_tags(category, context_values, required_tag)
    context_summary = build_context_summary(context_values)
    records: list[dict[str, Any]] = []

    for position in range(database_size):
        record_seed = f"{category}|{seed_context}|{position}"
        cost = 80 + stable_int(f"{record_seed}|cost", 920)
        duration = 1 + stable_int(f"{record_seed}|duration", 14)
        score = 55 + stable_int(f"{record_seed}|score", 46)
        tag_start = stable_int(f"{record_seed}|tags", len(tags))
        record_tags = [tags[(tag_start + offset) % len(tags)] for offset in range(min(3, len(tags)))]
        if required_tag and position == 0 and required_tag not in record_tags:
            record_tags[0] = required_tag
        name = f"{category.title()} Option {position + 1}"
        records.append(
            {
                "record_id": f"{category_slug}-{position + 1:03d}",
                "name": name,
                "category": category,
                "summary": f"{name} synthesized from {context_summary}.",
                "cost": cost,
                "duration": duration,
                "score": score,
                "tags": unique_values(record_tags),
                "source_values": dict(context_values),
            }
        )

    return records


def record_matches_constraints(record: dict[str, Any], constraints: dict[str, Any]) -> bool:
    """Return whether a record satisfies task constraints.

    Args:
        record: Database record.
        constraints: Task constraints.

    Returns:
        ``True`` when the record is eligible.
    """
    required_tag = constraints.get("required_tag")
    return (
        int(record["cost"]) <= int(constraints["max_cost"])
        and int(record["score"]) >= int(constraints["min_score"])
        and (required_tag is None or str(required_tag) in record.get("tags", []))
    )


def eligible_records(database: list[dict[str, Any]], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter records that satisfy task constraints.

    Args:
        database: Sandbox database records.
        constraints: Task constraints.

    Returns:
        Eligible records.
    """
    return [record for record in database if record_matches_constraints(record, constraints)]


def select_best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the optimal answer under the verifier ordering.

    Args:
        records: Candidate records.

    Returns:
        The best record, or ``None`` when no candidates exist.
    """
    if not records:
        return None
    return sorted(records, key=lambda record: (-int(record["score"]), int(record["cost"]), str(record["record_id"])))[0]


def default_constraints(
    database: list[dict[str, Any]],
    config: GeneralistAgentEnvColumnConfig,
    difficulty: Difficulty | None = None,
) -> dict[str, Any]:
    """Create feasible default constraints for the requested difficulty.

    Args:
        database: Sandbox database records.
        config: Column configuration.
        difficulty: Difficulty to synthesize; defaults to the configured final difficulty.

    Returns:
        Constraint values and repair notes.
    """
    difficulty = difficulty or config.difficulty
    required_tag = config.required_tag
    target_pool = [record for record in database if required_tag is None or required_tag in record["tags"]]
    target = select_best_record(target_pool) or select_best_record(database)
    if target is None:
        msg = "database must contain at least one record"
        raise ValueError(msg)

    if required_tag is None and difficulty in ("medium", "hard"):
        required_tag = str(target["tags"][0])

    if difficulty == "simple":
        default_max_cost = max(int(record["cost"]) for record in database)
        default_min_score = min(int(record["score"]) for record in database)
    elif difficulty == "medium":
        default_max_cost = int(target["cost"]) + 120
        default_min_score = max(0, int(target["score"]) - 12)
    else:
        default_max_cost = int(target["cost"]) + 40
        default_min_score = max(0, int(target["score"]) - 4)

    constraints = {
        "max_cost": config.max_cost if config.max_cost is not None else default_max_cost,
        "min_score": config.min_score if config.min_score is not None else default_min_score,
        "required_tag": required_tag,
        "repair_notes": [],
    }
    return repair_constraints(database, constraints)


def repair_constraints(database: list[dict[str, Any]], constraints: dict[str, Any]) -> dict[str, Any]:
    """Repair constraints that would otherwise make the task unsatisfiable.

    Args:
        database: Sandbox database records.
        constraints: Initial task constraints.

    Returns:
        Feasible constraints plus repair notes.
    """
    if eligible_records(database, constraints):
        return constraints

    required_tag = constraints.get("required_tag")
    target_pool = [record for record in database if required_tag is None or required_tag in record["tags"]]
    target = select_best_record(target_pool) or select_best_record(database)
    if target is None:
        return constraints

    if required_tag is not None and required_tag not in target["tags"]:
        constraints["required_tag"] = target["tags"][0]
        constraints["repair_notes"].append("required_tag changed to a tag present in the database")

    if int(target["cost"]) > int(constraints["max_cost"]):
        constraints["max_cost"] = int(target["cost"])
        constraints["repair_notes"].append("max_cost increased to keep at least one valid candidate")

    if int(target["score"]) < int(constraints["min_score"]):
        constraints["min_score"] = int(target["score"])
        constraints["repair_notes"].append("min_score decreased to keep at least one valid candidate")

    return constraints


def selected_tool_names(difficulty: Difficulty) -> list[str]:
    """Select the synthesized toolset for a difficulty level.

    Args:
        difficulty: Final task difficulty.

    Returns:
        Tool names to expose to the solution function.
    """
    tool_names = ["list_records", "search_records", "get_record"]
    if difficulty in ("medium", "hard"):
        tool_names.append("filter_records")
    if difficulty == "hard":
        tool_names.append("rank_records")
    return tool_names


def build_tool_specs(tool_names: list[str]) -> list[dict[str, str]]:
    """Build tool metadata and function source snippets.

    Args:
        tool_names: Selected tool names.

    Returns:
        Tool descriptors for the output tuple.
    """
    return [
        {
            "name": tool_name,
            "description": TOOL_DESCRIPTIONS[tool_name],
            "source": textwrap.dedent(TOOL_FUNCTION_SOURCES[tool_name]).strip(),
        }
        for tool_name in tool_names
    ]


def build_tool_module_source(database: list[dict[str, Any]], tool_names: list[str]) -> str:
    """Build executable Python source for the synthesized tool module.

    Args:
        database: Hidden sandbox database.
        tool_names: Selected tool names.

    Returns:
        Python module source defining ``DATABASE`` and tool functions.
    """
    parts = [f"DATABASE = {pformat(database, sort_dicts=False, width=120)}"]
    parts.extend(textwrap.dedent(TOOL_FUNCTION_SOURCES[tool_name]).strip() for tool_name in tool_names)
    return "\n\n".join(parts) + "\n"


def build_task_prompt(category: str, difficulty: Difficulty, constraints: dict[str, Any]) -> str:
    """Create the task prompt presented to a solving agent.

    Args:
        category: Seed task category.
        difficulty: Final task difficulty.
        constraints: Task constraints.

    Returns:
        Natural language task prompt.
    """
    clauses = [
        f"Use the synthesized tools to solve this {difficulty} {category!r} task.",
        "Return the record_id for the eligible database record with the highest score.",
        f"Only consider records with cost <= {constraints['max_cost']} and score >= {constraints['min_score']}.",
    ]
    if constraints.get("required_tag") is not None:
        clauses.append(f"The record must include the tag {constraints['required_tag']!r}.")
    clauses.append("Break ties by lower cost, then lexicographic record_id.")
    return " ".join(clauses)


def build_reference_answer(database: list[dict[str, Any]], constraints: dict[str, Any]) -> dict[str, Any]:
    """Compute the verifier's expected answer.

    Args:
        database: Sandbox database records.
        constraints: Task constraints.

    Returns:
        JSON-compatible answer object.
    """
    best = select_best_record(eligible_records(database, constraints))
    if best is None:
        return {"record_id": None, "reason": "no eligible records"}
    return {
        "record_id": best["record_id"],
        "score": best["score"],
        "cost": best["cost"],
        "tags": list(best["tags"]),
    }


def verify_answer(answer: dict[str, Any], database: list[dict[str, Any]], constraints: dict[str, Any]) -> bool:
    """Verify an answer against the database and constraints.

    Args:
        answer: Candidate answer.
        database: Sandbox database records.
        constraints: Task constraints.

    Returns:
        ``True`` when the answer is exactly the verifier-optimal record.
    """
    if not isinstance(answer, dict):
        return False
    best = select_best_record(eligible_records(database, constraints))
    if best is None:
        return answer.get("record_id") is None
    return (
        answer.get("record_id") == best["record_id"]
        and int(answer.get("score", -1)) == int(best["score"])
        and int(answer.get("cost", -1)) == int(best["cost"])
    )


def build_solution_source(constraints: dict[str, Any], difficulty: Difficulty) -> str:
    """Build a tool-only Python solution function.

    Args:
        constraints: Task constraints.
        difficulty: Final task difficulty.

    Returns:
        Python source defining ``solve(tools)``.
    """
    required_tag = repr(constraints.get("required_tag"))
    lines = [
        "def solve(tools):",
        '    """Solve the task using only synthesized tool functions and local logic."""',
        f"    required_tag = {required_tag}",
    ]

    if difficulty == "simple":
        lines.extend(
            [
                "    candidates = []",
                '    for record in tools["list_records"]():',
                f'        if int(record["cost"]) > {constraints["max_cost"]}:',
                "            continue",
                f'        if int(record["score"]) < {constraints["min_score"]}:',
                "            continue",
                '        if required_tag is not None and required_tag not in record.get("tags", []):',
                "            continue",
                "        candidates.append(record)",
            ]
        )
    else:
        lines.extend(
            [
                '    candidates = tools["filter_records"](',
                f"        max_cost={constraints['max_cost']},",
                f"        min_score={constraints['min_score']},",
                "        required_tag=required_tag,",
                "    )",
            ]
        )

    lines.extend(
        [
            "    if not candidates:",
            '        return {"record_id": None, "reason": "no eligible records"}',
        ]
    )
    if difficulty == "hard":
        lines.extend(
            [
                '    ranked = tools["rank_records"](candidates, metric="score", descending=True)',
                '    ranked = sorted(ranked, key=lambda record: (-int(record["score"]), int(record["cost"]), str(record["record_id"])))',
            ]
        )
    else:
        lines.append(
            '    ranked = sorted(candidates, key=lambda record: (-int(record["score"]), int(record["cost"]), str(record["record_id"])))'
        )

    lines.extend(
        [
            "    best = ranked[0]",
            "    return {",
            '        "record_id": best["record_id"],',
            '        "score": best["score"],',
            '        "cost": best["cost"],',
            '        "tags": list(best.get("tags", [])),',
            "    }",
        ]
    )
    return "\n".join(lines)


def build_verifier_source(constraints: dict[str, Any]) -> str:
    """Build a Python verifier function for the synthesized task.

    Args:
        constraints: Task constraints.

    Returns:
        Python source defining ``verify(answer, database)``.
    """
    verifier_constraints = {
        "max_cost": constraints["max_cost"],
        "min_score": constraints["min_score"],
        "required_tag": constraints.get("required_tag"),
    }
    return textwrap.dedent(
        f'''
        CONSTRAINTS = {pformat(verifier_constraints, sort_dicts=False, width=120)}


        def verify(answer, database):
            """Return True when answer satisfies the task and is verifier-optimal."""
            if not isinstance(answer, dict):
                return False

            eligible = []
            for record in database:
                if int(record["cost"]) > int(CONSTRAINTS["max_cost"]):
                    continue
                if int(record["score"]) < int(CONSTRAINTS["min_score"]):
                    continue
                required_tag = CONSTRAINTS.get("required_tag")
                if required_tag is not None and str(required_tag) not in record.get("tags", []):
                    continue
                eligible.append(record)

            if not eligible:
                return answer.get("record_id") is None

            best = sorted(
                eligible,
                key=lambda record: (-int(record["score"]), int(record["cost"]), str(record["record_id"])),
            )[0]
            return (
                answer.get("record_id") == best["record_id"]
                and int(answer.get("score", -1)) == int(best["score"])
                and int(answer.get("cost", -1)) == int(best["cost"])
            )
        '''
    ).strip()


def build_task_iteration(
    category: str,
    database: list[dict[str, Any]],
    config: GeneralistAgentEnvColumnConfig,
    difficulty: Difficulty,
) -> dict[str, Any]:
    """Build one synthesized task, solution, and verifier iteration.

    Args:
        category: Seed task category.
        database: Sandbox database records.
        config: Column configuration.
        difficulty: Difficulty level for this iteration.

    Returns:
        JSON-compatible iteration artifact.
    """
    constraints = default_constraints(database, config, difficulty)
    answer = build_reference_answer(database, constraints)
    verified = verify_answer(answer, database, constraints)
    return {
        "difficulty": difficulty,
        "tool_names": selected_tool_names(difficulty),
        "task_prompt": build_task_prompt(category, difficulty, constraints),
        "constraints": constraints,
        "solution_source": build_solution_source(constraints, difficulty),
        "verifier_source": build_verifier_source(constraints),
        "reference_answer": answer,
        "reference_solution_passed": verified,
        "augmentation_required": difficulty in ("medium", "hard"),
    }


def build_task_iterations(
    category: str,
    database: list[dict[str, Any]],
    config: GeneralistAgentEnvColumnConfig,
) -> list[dict[str, Any]]:
    """Build the simple-to-final task synthesis iterations.

    Args:
        category: Seed task category.
        database: Sandbox database records.
        config: Column configuration.

    Returns:
        Ordered task iteration artifacts.
    """
    return [
        build_task_iteration(category, database, config, difficulty)
        for difficulty in difficulty_trace(config.difficulty)
    ]


def difficulty_trace(final_difficulty: Difficulty) -> list[Difficulty]:
    """List difficulty levels synthesized before the final task.

    Args:
        final_difficulty: Requested final difficulty.

    Returns:
        Ordered difficulty names through the final level.
    """
    return DIFFICULTY_ORDER[: DIFFICULTY_ORDER.index(final_difficulty) + 1]


def build_synthesis_trace(
    category: str,
    difficulty: Difficulty,
    tool_names: list[str],
    constraints: dict[str, Any],
    verified: bool,
) -> list[dict[str, Any]]:
    """Describe the Generalist-style synthesis workflow for one row.

    Args:
        category: Seed task category.
        difficulty: Final task difficulty.
        tool_names: Synthesized tool names.
        constraints: Final task constraints.
        verified: Whether the generated reference answer passes verification.

    Returns:
        Ordered workflow events.
    """
    trace: list[dict[str, Any]] = [
        {
            "stage": "environment_and_toolset_construction",
            "category": category,
            "sandbox_tools": list(BASE_SANDBOX_TOOLS),
            "database_created": True,
        }
    ]
    for level in difficulty_trace(difficulty):
        trace.append(
            {
                "stage": "task_synthesis",
                "difficulty": level,
                "goal": "hard to solve through search, easy to verify by deterministic constraints",
            }
        )
        if level in ("medium", "hard"):
            trace.append(
                {
                    "stage": "toolset_augmentation",
                    "difficulty": level,
                    "available_tools": selected_tool_names(level),
                }
            )
    trace.append(
        {
            "stage": "solution_generation",
            "solution_restriction": "solution source calls synthesized tools and uses local logical computation only",
            "final_tools": tool_names,
        }
    )
    trace.append(
        {
            "stage": "verification",
            "constraints": {key: value for key, value in constraints.items() if key != "repair_notes"},
            "reference_solution_passed": verified,
        }
    )
    return trace


def build_environment_tuple(
    category: str,
    context_values: dict[str, str],
    config: GeneralistAgentEnvColumnConfig,
    row_number: int,
) -> dict[str, Any]:
    """Build one ``<environment, tools, task, verifier>`` tuple.

    Args:
        category: Seed task category.
        context_values: Context copied from the seed row.
        config: Column configuration.
        row_number: Zero-based row position used for stable ids.

    Returns:
        Structured Generalist environment tuple.
    """
    database = build_database(category, context_values, config.database_size, config.required_tag)
    task_iterations = build_task_iterations(category, database, config)
    final_iteration = task_iterations[-1]
    constraints = final_iteration["constraints"]
    answer = final_iteration["reference_answer"]
    verified = bool(final_iteration["reference_solution_passed"])
    tool_names = final_iteration["tool_names"]
    category_slug = slugify(category, "task")
    context_slug = stable_int(json.dumps(context_values, sort_keys=True), 10_000)
    environment_id = f"{category_slug}-{row_number + 1:04d}-{context_slug:04d}"

    return {
        "schema_version": "generalist-agent-env/v1",
        "source_workflow": "DeepSeek-V3.2 Generalist automatic environment synthesis",
        "environment": {
            "environment_id": environment_id,
            "category": category,
            "sandbox": {
                "base_tools": list(BASE_SANDBOX_TOOLS),
                "database_name": f"{environment_id}_db",
                "database_schema": dict(DATABASE_SCHEMA),
            },
            "database": database,
            "database_record_count": len(database),
            "source_context": dict(context_values),
            "data_acquisition": {
                "mode": "synthetic",
                "base_sandbox_tools": list(BASE_SANDBOX_TOOLS),
                "note": "Records are generated locally from seed data; downstream workflows may replace them with search-retrieved records.",
            },
        },
        "tools": build_tool_specs(tool_names),
        "tool_module_source": build_tool_module_source(database, tool_names),
        "task": {
            "difficulty": config.difficulty,
            "category": category,
            "prompt": build_task_prompt(category, config.difficulty, constraints),
            "constraints": constraints,
            "answer_schema": {
                "record_id": "string or null",
                "score": "integer when record_id is not null",
                "cost": "integer when record_id is not null",
                "tags": "list of strings when record_id is not null",
            },
        },
        "solution": {
            "language": "python",
            "entrypoint": "solve",
            "source": final_iteration["solution_source"],
            "restrictions": [
                "may call synthesized tool functions",
                "may perform local logical computation",
                "must not directly access the sandbox database",
            ],
        },
        "verifier": {
            "language": "python",
            "entrypoint": "verify",
            "source": final_iteration["verifier_source"],
            "reference_solution_passed": verified,
        },
        "reference_answer": answer,
        "task_iterations": task_iterations,
        "synthesis_trace": build_synthesis_trace(category, config.difficulty, tool_names, constraints, verified),
        "rl_filter_note": "Downstream RL retention can keep generated tuples with non-zero pass@100.",
    }


class GeneralistAgentEnvColumnGenerator(ColumnGeneratorFullColumn[GeneralistAgentEnvColumnConfig]):
    """Generate Generalist agent environment tuples for each input row."""

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate structured environment tuples.

        Args:
            data: Input DataFrame containing the configured task category and context columns.

        Returns:
            The input DataFrame with the configured output column populated.
        """
        tuples: list[dict[str, Any]] = []
        for row_number, (_, row) in enumerate(data.iterrows()):
            category = normalize_cell(row[self.config.task_category_column]) or "general task"
            context_values = {
                column: normalize_cell(row[column])
                for column in self.config.context_columns
                if normalize_cell(row[column])
            }
            tuples.append(build_environment_tuple(category, context_values, self.config, row_number))
        data[self.config.name] = tuples
        return data
