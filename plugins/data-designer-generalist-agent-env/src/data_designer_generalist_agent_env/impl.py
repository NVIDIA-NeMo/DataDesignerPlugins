# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from collections.abc import Mapping
from pprint import pformat
from typing import TYPE_CHECKING, Any

from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from data_designer_generalist_agent_env.config import (
    Difficulty,
    GeneralistAgentEnvironmentColumnConfig,
    GeneralistAgentTaskColumnConfig,
)

if TYPE_CHECKING:
    import pandas as pd

BASE_SANDBOX_TOOLS = ["data_designer_generated_schema", "data_designer_generated_records"]
DIFFICULTY_ORDER: list[Difficulty] = ["simple", "medium", "hard"]
REQUIRED_RECORD_FIELDS = ["record_id", "name", "summary", "cost", "duration", "score", "tags"]
DEFAULT_DATABASE_SCHEMA = {
    "record_type": "generated_candidate",
    "primary_key": "record_id",
    "fields": [
        {"name": "record_id", "type": "string", "description": "Stable row-local identifier."},
        {"name": "name", "type": "string", "description": "Human-readable candidate name."},
        {"name": "summary", "type": "string", "description": "Short generated candidate description."},
        {"name": "cost", "type": "integer", "description": "Integer cost proxy; lower is better."},
        {"name": "duration", "type": "integer", "description": "Integer duration or effort proxy."},
        {"name": "score", "type": "integer", "description": "Integer quality score from 0 to 100; higher is better."},
        {"name": "tags", "type": "list[string]", "description": "Searchable task-specific labels."},
        {"name": "attributes", "type": "object", "description": "Topic-specific generated attributes."},
    ],
}

TOOL_FUNCTION_SOURCES = {
    "describe_schema": '''
def describe_schema():
    """Return the generated database schema."""
    return dict(DATABASE_SCHEMA)
''',
    "list_records": '''
def list_records():
    """Return every record in the sandbox database."""
    return [dict(record) for record in DATABASE]
''',
    "search_records": '''
def search_records(query="", max_results=10):
    """Search database records by name, summary, topic, tag, or generated attribute."""
    needle = str(query or "").casefold()
    limit = max(0, int(max_results))
    matches = []
    for record in DATABASE:
        attributes = record.get("attributes", {})
        attribute_text = " ".join(str(value) for value in attributes.values()) if isinstance(attributes, dict) else ""
        haystack = " ".join(
            [
                str(record.get("name", "")),
                str(record.get("summary", "")),
                str(record.get("topic", "")),
                " ".join(str(tag) for tag in record.get("tags", [])),
                attribute_text,
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
    "describe_schema": "Inspect the generated row-local database schema.",
    "list_records": "Inspect all generated rows in the hidden sandbox database.",
    "search_records": "Retrieve topic-relevant records through a search-style interface.",
    "get_record": "Fetch one generated database record by identifier.",
    "filter_records": "Apply verifier-aligned constraints without exposing the database directly.",
    "rank_records": "Rank generated candidate records for the final selection step.",
}


def is_null_like(value: object) -> bool:
    """Return whether a value is empty or pandas-null-like.

    Args:
        value: Candidate cell value.

    Returns:
        ``True`` when the value should be treated as missing.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def normalize_cell(value: object) -> str:
    """Normalize one pandas cell into a stable text value.

    Args:
        value: Cell value from a seed row.

    Returns:
        A stripped string, or an empty string for null-like values.
    """
    if is_null_like(value):
        return ""
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


def coerce_list_like(value: Any) -> list[Any] | None:
    """Coerce common list-like values into a Python list.

    Args:
        value: Candidate list-like value.

    Returns:
        A Python list when coercion is possible, otherwise ``None``.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
    return None


def to_plain_data(value: Any) -> Any:
    """Convert nested array-like values into JSON-style Python containers.

    Args:
        value: Arbitrary generated artifact value.

    Returns:
        The value with mappings and list-like values recursively normalized.
    """
    if isinstance(value, Mapping):
        return {key: to_plain_data(nested_value) for key, nested_value in value.items()}

    values = coerce_list_like(value)
    if values is not None:
        return [to_plain_data(nested_value) for nested_value in values]

    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def parse_generated_payload(value: Any, field_name: str) -> Any:
    """Parse generated JSON-like cell values.

    Args:
        value: Cell value generated by a previous Data Designer column.
        field_name: Name used in validation errors.

    Returns:
        Parsed JSON-like data.

    Raises:
        ValueError: If the value is missing or a JSON string cannot be parsed.
    """
    if is_null_like(value):
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            msg = f"{field_name} must not be empty"
            raise ValueError(msg)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return to_plain_data(value)


def constraint_payload_to_text(value: Any) -> str:
    """Flatten a generated constraints payload into compact text.

    Args:
        value: Constraint value from a row. Supported values include strings,
            mappings, lists, JSON strings, and scalar values.

    Returns:
        Text used for environment provenance.
    """
    if is_null_like(value):
        return ""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return constraint_payload_to_text(json.loads(stripped))
        except json.JSONDecodeError:
            return stripped

    plain = to_plain_data(value)
    if isinstance(plain, Mapping):
        parts = []
        for key, nested_value in plain.items():
            nested_text = constraint_payload_to_text(nested_value)
            if nested_text:
                parts.append(f"{key}: {nested_text}")
        return "; ".join(parts)

    values = coerce_list_like(plain)
    if values is not None:
        return "; ".join(text for text in (constraint_payload_to_text(item) for item in values) if text)

    return normalize_cell(plain)


def normalize_database_schema(value: Any) -> dict[str, Any]:
    """Normalize a generated database schema payload.

    Args:
        value: Generated schema value from a row.

    Returns:
        Schema metadata as a dictionary.

    Raises:
        ValueError: If the schema is not mapping-like.
    """
    parsed = parse_generated_payload(value, "database_schema")
    if not isinstance(parsed, Mapping):
        msg = "database_schema must be a mapping generated by an upstream Data Designer column"
        raise ValueError(msg)

    schema = dict(parsed)
    if "record_type" not in schema:
        schema["record_type"] = DEFAULT_DATABASE_SCHEMA["record_type"]
    if "primary_key" not in schema:
        schema["primary_key"] = DEFAULT_DATABASE_SCHEMA["primary_key"]
    if "fields" not in schema:
        schema["fields"] = DEFAULT_DATABASE_SCHEMA["fields"]
    return schema


def extract_records_payload(value: Any) -> list[Any]:
    """Extract generated record payloads from common structured output shapes.

    Args:
        value: Generated records value from a row.

    Returns:
        List of record payloads.

    Raises:
        ValueError: If no list-like records can be extracted.
    """
    parsed = parse_generated_payload(value, "database_records")
    if isinstance(parsed, Mapping):
        for key in ("records", "items", "data", "rows"):
            nested = parsed.get(key)
            if nested is not None:
                records = coerce_list_like(nested)
                if records is not None:
                    return records
        msg = "database_records mapping must contain a records, items, data, or rows list"
        raise ValueError(msg)

    records = coerce_list_like(parsed)
    if records is None:
        msg = "database_records must be a list or an object containing a records list"
        raise ValueError(msg)
    return records


def normalize_tags(value: Any, record_id: str) -> list[str]:
    """Normalize generated record tags.

    Args:
        value: Generated tags value.
        record_id: Record id used in errors.

    Returns:
        List of tag strings.

    Raises:
        ValueError: If tags are missing or cannot be interpreted as a non-empty list.
    """
    tags = coerce_list_like(value)
    if tags is None and isinstance(value, str):
        tags = [tag.strip() for tag in re.split(r"[,;]", value) if tag.strip()]
    if tags is None or not tags:
        msg = f"generated record {record_id!r} must include at least one tag"
        raise ValueError(msg)
    return [str(tag).strip().lower() for tag in tags if str(tag).strip()]


def normalize_int_field(value: Any, field_name: str, record_id: str) -> int:
    """Normalize an integer field from a generated record.

    Args:
        value: Generated field value.
        field_name: Field name.
        record_id: Record id used in errors.

    Returns:
        Integer field value.

    Raises:
        ValueError: If the value cannot be converted to an integer.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        msg = f"generated record {record_id!r} field {field_name!r} must be an integer"
        raise ValueError(msg) from exc


def normalize_generated_record(value: Any, index: int, topic: str) -> dict[str, Any]:
    """Normalize and validate one generated database record.

    Args:
        value: Generated record value.
        index: Zero-based record index.
        topic: Generated task topic.

    Returns:
        Normalized record.

    Raises:
        ValueError: If required fields are absent or invalid.
    """
    value = to_plain_data(value)
    if not isinstance(value, Mapping):
        msg = f"database_records[{index}] must be a mapping"
        raise ValueError(msg)

    record = dict(value)
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        msg = f"database_records[{index}] is missing required fields: {', '.join(missing)}"
        raise ValueError(msg)

    record_id = str(record["record_id"]).strip()
    if not record_id:
        msg = f"database_records[{index}].record_id must not be empty"
        raise ValueError(msg)

    normalized = dict(record)
    normalized["record_id"] = record_id
    normalized["name"] = str(record["name"]).strip()
    normalized["summary"] = str(record["summary"]).strip()
    normalized["topic"] = str(record.get("topic") or topic)
    normalized["cost"] = normalize_int_field(record["cost"], "cost", record_id)
    normalized["duration"] = normalize_int_field(record["duration"], "duration", record_id)
    normalized["score"] = normalize_int_field(record["score"], "score", record_id)
    normalized["tags"] = normalize_tags(record["tags"], record_id)
    attributes = record.get("attributes", {})
    normalized["attributes"] = dict(attributes) if isinstance(attributes, Mapping) else {"value": attributes}
    return normalized


def normalize_database_records(value: Any, topic: str | None = None) -> list[dict[str, Any]]:
    """Normalize generated records restored from memory or saved artifacts.

    Args:
        value: Database records payload.
        topic: Generated topic used when records omit a topic field.

    Returns:
        Database records as plain dictionaries.

    Raises:
        ValueError: If records are absent or invalid.
    """
    records = extract_records_payload(value)
    if not records:
        msg = "database_records must contain at least one generated record"
        raise ValueError(msg)
    topic = topic or "general task"
    normalized = [normalize_generated_record(record, index, topic) for index, record in enumerate(records)]
    duplicate_ids = sorted(
        {
            record["record_id"]
            for record in normalized
            if [candidate["record_id"] for candidate in normalized].count(record["record_id"]) > 1
        }
    )
    if duplicate_ids:
        msg = f"database_records contain duplicate record_id values: {', '.join(duplicate_ids)}"
        raise ValueError(msg)
    return normalized


def validate_schema_covers_records(schema: Mapping[str, Any], records: list[dict[str, Any]]) -> None:
    """Validate that generated records are compatible with the generated schema.

    Args:
        schema: Generated schema metadata.
        records: Normalized generated records.

    Raises:
        ValueError: If the schema primary key is incompatible with records.
    """
    primary_key = str(schema.get("primary_key", "record_id"))
    if primary_key != "record_id":
        msg = "generated database schema primary_key must be 'record_id'"
        raise ValueError(msg)
    for field in REQUIRED_RECORD_FIELDS:
        if field not in records[0]:
            msg = f"generated records must include required field {field!r}"
            raise ValueError(msg)


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
    config: GeneralistAgentTaskColumnConfig,
    difficulty: Difficulty | None = None,
) -> dict[str, Any]:
    """Create feasible default constraints for the requested difficulty.

    Args:
        database: Sandbox database records.
        config: Task column configuration.
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
    tool_names = ["describe_schema", "list_records", "search_records", "get_record"]
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


def build_tool_module_source(
    database_schema: Mapping[str, Any], database: list[dict[str, Any]], tool_names: list[str]
) -> str:
    """Build executable Python source for the generated tool module.

    Args:
        database_schema: Generated row-local database schema.
        database: Generated sandbox database.
        tool_names: Selected tool names.

    Returns:
        Python module source defining ``DATABASE_SCHEMA``, ``DATABASE``, and tool functions.
    """
    parts = [
        f"DATABASE_SCHEMA = {pformat(dict(database_schema), sort_dicts=False, width=120)}",
        f"DATABASE = {pformat(database, sort_dicts=False, width=120)}",
    ]
    parts.extend(textwrap.dedent(TOOL_FUNCTION_SOURCES[tool_name]).strip() for tool_name in tool_names)
    return "\n\n".join(parts) + "\n"


def build_task_prompt(topic: str, difficulty: Difficulty, constraints: dict[str, Any]) -> str:
    """Create the task prompt presented to a solving agent.

    Args:
        topic: Generated task topic.
        difficulty: Final task difficulty.
        constraints: Task constraints.

    Returns:
        Natural language task prompt.
    """
    clauses = [
        f"Use the synthesized tools to solve this {difficulty} {topic!r} task.",
        "Inspect the generated schema and records through the tool interface; do not access the database directly.",
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
        '    tools["describe_schema"]()',
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
    topic: str,
    database: list[dict[str, Any]],
    config: GeneralistAgentTaskColumnConfig,
    difficulty: Difficulty,
) -> dict[str, Any]:
    """Build one synthesized task, solution, and verifier iteration.

    Args:
        topic: Generated task topic.
        database: Sandbox database records.
        config: Task column configuration.
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
        "task_prompt": build_task_prompt(topic, difficulty, constraints),
        "constraints": constraints,
        "solution_source": build_solution_source(constraints, difficulty),
        "verifier_source": build_verifier_source(constraints),
        "reference_answer": answer,
        "reference_solution_passed": verified,
        "augmentation_required": difficulty in ("medium", "hard"),
    }


def difficulty_trace(final_difficulty: Difficulty) -> list[Difficulty]:
    """List difficulty levels synthesized before the final task.

    Args:
        final_difficulty: Requested final difficulty.

    Returns:
        Ordered difficulty names through the final level.
    """
    return DIFFICULTY_ORDER[: DIFFICULTY_ORDER.index(final_difficulty) + 1]


def build_task_iterations(
    topic: str,
    database: list[dict[str, Any]],
    config: GeneralistAgentTaskColumnConfig,
) -> list[dict[str, Any]]:
    """Build the simple-to-final task synthesis iterations.

    Args:
        topic: Generated task topic.
        database: Sandbox database records.
        config: Task column configuration.

    Returns:
        Ordered task iteration artifacts.
    """
    return [
        build_task_iteration(topic, database, config, difficulty) for difficulty in difficulty_trace(config.difficulty)
    ]


def build_task_synthesis_trace(
    topic: str,
    difficulty: Difficulty,
    tool_names: list[str],
    constraints: dict[str, Any],
    verified: bool,
) -> list[dict[str, Any]]:
    """Describe the task synthesis workflow for one row.

    Args:
        topic: Generated task topic.
        difficulty: Final task difficulty.
        tool_names: Synthesized tool names.
        constraints: Final task constraints.
        verified: Whether the generated reference answer passes verification.

    Returns:
        Ordered workflow events.
    """
    trace: list[dict[str, Any]] = []
    for level in difficulty_trace(difficulty):
        trace.append(
            {
                "stage": "task_synthesis",
                "difficulty": level,
                "goal": "hard to solve through tools, easy to verify by deterministic constraints",
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
            "topic": topic,
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


def build_environment_id(topic: str, context_values: dict[str, str], row_number: int) -> str:
    """Build a stable row-local environment identifier.

    Args:
        topic: Generated task topic.
        context_values: Context copied from the seed row.
        row_number: Zero-based row position.

    Returns:
        Stable environment identifier.
    """
    topic_slug = slugify(topic, "task")
    context_slug = stable_int(json.dumps(context_values, sort_keys=True), 10_000)
    return f"{topic_slug}-{row_number + 1:04d}-{context_slug:04d}"


def build_environment_artifact(
    topic: str,
    constraints_payload: Any,
    constraints_text: str,
    context_values: dict[str, str],
    database_schema: dict[str, Any],
    database: list[dict[str, Any]],
    row_number: int,
) -> dict[str, Any]:
    """Build one standalone generated environment and toolset artifact.

    Args:
        topic: Generated task topic.
        constraints_payload: Raw generated constraints payload normalized to JSON-like data.
        constraints_text: Generated constraints flattened to text.
        context_values: Context copied from the seed row.
        database_schema: Generated database schema.
        database: Generated database records.
        row_number: Zero-based row position used for stable ids.

    Returns:
        Structured Generalist environment artifact.
    """
    validate_schema_covers_records(database_schema, database)
    environment_id = build_environment_id(topic, context_values, row_number)
    tool_names = selected_tool_names("hard")
    return {
        "schema_version": "generalist-agent-environment/v1",
        "source_workflow": "Generated Generalist environment and toolset assembly",
        "environment": {
            "environment_id": environment_id,
            "topic": topic,
            "sandbox": {
                "base_tools": list(BASE_SANDBOX_TOOLS),
                "database_name": f"{environment_id}_db",
            },
            "database_schema": database_schema,
            "database": database,
            "database_record_count": len(database),
            "task_constraints": constraints_payload,
            "task_constraints_text": constraints_text,
            "source_context": dict(context_values),
            "data_generation": {
                "mode": "generated_by_data_designer_columns",
                "note": "Topic, constraints, schema, and records are generated upstream by Data Designer columns.",
            },
        },
        "tools": build_tool_specs(tool_names),
        "tool_module_source": build_tool_module_source(database_schema, database, tool_names),
        "synthesis_trace": [
            {
                "stage": "topic_and_constraint_intake",
                "topic": topic,
                "constraints_available": bool(constraints_text),
            },
            {
                "stage": "schema_intake",
                "record_type": database_schema.get("record_type"),
                "primary_key": database_schema.get("primary_key"),
            },
            {
                "stage": "generated_data_intake",
                "database_record_count": len(database),
                "toolset": tool_names,
            },
        ],
    }


def build_task_tuple(
    environment_artifact: dict[str, Any],
    config: GeneralistAgentTaskColumnConfig,
) -> dict[str, Any]:
    """Build one ``<environment, tools, task, verifier>`` tuple from an environment.

    Args:
        environment_artifact: Output from ``generalist-agent-environment``.
        config: Task column configuration.

    Returns:
        Structured Generalist task tuple.
    """
    environment = dict(environment_artifact["environment"])
    database_schema = normalize_database_schema(environment["database_schema"])
    topic = str(environment.get("topic") or "general task")
    database = normalize_database_records(environment["database"], topic)
    environment["database_schema"] = database_schema
    environment["database"] = database
    task_iterations = build_task_iterations(topic, database, config)
    final_iteration = task_iterations[-1]
    constraints = final_iteration["constraints"]
    answer = final_iteration["reference_answer"]
    verified = bool(final_iteration["reference_solution_passed"])
    tool_names = final_iteration["tool_names"]

    return {
        "schema_version": "generalist-agent-task/v1",
        "source_workflow": "Generalist task synthesis from generated environment",
        "environment": environment,
        "tools": build_tool_specs(tool_names),
        "tool_module_source": build_tool_module_source(database_schema, database, tool_names),
        "task": {
            "difficulty": config.difficulty,
            "topic": topic,
            "prompt": build_task_prompt(topic, config.difficulty, constraints),
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
        "synthesis_trace": [
            *environment_artifact.get("synthesis_trace", []),
            *build_task_synthesis_trace(topic, config.difficulty, tool_names, constraints, verified),
        ],
        "rl_filter_note": "Downstream RL retention can keep generated tuples with non-zero pass@100.",
    }


class GeneralistAgentEnvironmentColumnGenerator(ColumnGeneratorFullColumn[GeneralistAgentEnvironmentColumnConfig]):
    """Assemble generated Generalist environment and toolset artifacts."""

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate environment artifacts from upstream generated schema and records.

        Args:
            data: Input DataFrame containing generated task topic, optional
                generated constraints, generated database schema, generated
                database records, and optional context columns.

        Returns:
            The input DataFrame with the configured output column populated.
        """
        artifacts: list[dict[str, Any]] = []
        for row_number, (_, row) in enumerate(data.iterrows()):
            topic = normalize_cell(row[self.config.task_topic_column]) or "general task"
            constraints_cell = (
                row[self.config.task_constraints_column] if self.config.task_constraints_column is not None else None
            )
            constraints_payload = to_plain_data(constraints_cell) if constraints_cell is not None else {}
            constraints_text = constraint_payload_to_text(constraints_cell)
            database_schema = normalize_database_schema(row[self.config.database_schema_column])
            database = normalize_database_records(row[self.config.database_records_column], topic)
            context_values = {
                column: normalize_cell(row[column])
                for column in self.config.context_columns
                if normalize_cell(row[column])
            }
            artifacts.append(
                build_environment_artifact(
                    topic,
                    constraints_payload,
                    constraints_text,
                    context_values,
                    database_schema,
                    database,
                    row_number,
                )
            )
        data[self.config.name] = artifacts
        return data


class GeneralistAgentTaskColumnGenerator(ColumnGeneratorFullColumn[GeneralistAgentTaskColumnConfig]):
    """Generate Generalist task tuples from constructed generated environments."""

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate task, solution, and verifier tuples from environments.

        Args:
            data: Input DataFrame containing the configured environment column.

        Returns:
            The input DataFrame with the configured output column populated.
        """
        tuples: list[dict[str, Any]] = []
        for _, row in data.iterrows():
            environment_artifact = row[self.config.environment_column]
            if not isinstance(environment_artifact, dict):
                msg = f"{self.config.environment_column!r} must contain environment artifact dictionaries"
                raise ValueError(msg)
            tuples.append(build_task_tuple(environment_artifact, self.config))
        data[self.config.name] = tuples
        return data
