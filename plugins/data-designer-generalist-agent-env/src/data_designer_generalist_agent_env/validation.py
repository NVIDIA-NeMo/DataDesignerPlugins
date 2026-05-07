# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolExecutionCheck:
    """Execution result for one generated tool.

    Attributes:
        name: Tool name from the generated row artifact.
        passed: Whether the tool executed and returned the expected output shape.
        output_type: Python type name returned by the smoke invocation.
        output_size: Length of the output when the output is a sized collection.
        error: Error message when execution failed.
    """

    name: str
    passed: bool
    output_type: str | None = None
    output_size: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class IterationExecutionCheck:
    """Execution result for one generated task iteration.

    Attributes:
        difficulty: Iteration difficulty label.
        passed: Whether the iteration solution was accepted by its verifier.
        answer: Answer returned by the iteration solution.
        verifier_passed: Raw verifier decision for the generated answer.
        error: Error message when execution failed.
    """

    difficulty: str
    passed: bool
    answer: Any | None = None
    verifier_passed: bool = False
    error: str | None = None


@dataclass(frozen=True)
class RowRecordValidationResult:
    """Validation result for a generated Generalist environment row record.

    Attributes:
        passed: Whether all executable artifacts passed validation.
        answer: Answer returned by the final generated solution.
        verifier_passed: Raw verifier decision for the final generated answer.
        tools_passed: Whether all generated tools passed smoke execution.
        tool_checks: Per-tool execution checks.
        iteration_checks: Per-iteration solution and verifier checks.
        errors: Validation errors collected across executable artifacts.
    """

    passed: bool
    answer: Any | None
    verifier_passed: bool
    tools_passed: bool
    tool_checks: list[ToolExecutionCheck] = field(default_factory=list)
    iteration_checks: list[IterationExecutionCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def failed_validation(error: str) -> RowRecordValidationResult:
    """Build a failed validation result.

    Args:
        error: Error message to attach to the result.

    Returns:
        A failed validation result with no executable artifacts.
    """
    return RowRecordValidationResult(
        passed=False,
        answer=None,
        verifier_passed=False,
        tools_passed=False,
        errors=[error],
    )


def execute_source_module(source: str, expected_entrypoints: list[str] | None = None) -> dict[str, Any]:
    """Execute generated Python source and return its namespace.

    Args:
        source: Python module source emitted by the plugin.
        expected_entrypoints: Callable names that must exist after execution.

    Returns:
        The execution namespace.

    Raises:
        ValueError: If an expected entrypoint is missing or is not callable.
        Exception: Any exception raised by the generated source during execution.
    """
    namespace: dict[str, Any] = {}
    exec(source, namespace)
    for entrypoint in expected_entrypoints or []:
        candidate = namespace.get(entrypoint)
        if not callable(candidate):
            msg = f"expected callable {entrypoint!r} in generated source"
            raise ValueError(msg)
    return namespace


def extract_environment_tuple(row_record: Mapping[str, Any], output_column: str | None = None) -> Mapping[str, Any]:
    """Extract the generated environment tuple from a row-like record.

    Args:
        row_record: Either the generated environment tuple itself or a row mapping
            that contains the tuple in ``output_column``.
        output_column: Optional output column containing the generated tuple.

    Returns:
        The generated environment tuple mapping.

    Raises:
        KeyError: If ``output_column`` is supplied but absent.
        TypeError: If the extracted value is not mapping-like.
        ValueError: If ``row_record`` is not already an environment tuple and no
            ``output_column`` is supplied.
    """
    if output_column is None:
        if "schema_version" in row_record and "environment" in row_record and "tools" in row_record:
            return row_record
        msg = "row_record must be an environment tuple unless output_column is provided"
        raise ValueError(msg)

    if output_column not in row_record:
        msg = f"row_record does not contain output column {output_column!r}"
        raise KeyError(msg)

    environment_tuple = row_record[output_column]
    if not isinstance(environment_tuple, Mapping):
        msg = f"row_record[{output_column!r}] must be a mapping"
        raise TypeError(msg)
    return environment_tuple


def output_size(output: Any) -> int | None:
    """Return a compact size for common collection outputs.

    Args:
        output: Tool output.

    Returns:
        The output length for common collections, otherwise ``None``.
    """
    if isinstance(output, (dict, list, set, tuple)):
        return len(output)
    return None


def tool_output_error(tool_name: str, output: Any) -> str | None:
    """Validate the expected output shape for a generated tool.

    Args:
        tool_name: Tool name.
        output: Value returned by the tool smoke invocation.

    Returns:
        An error message when the output shape is unexpected, otherwise ``None``.
    """
    if tool_name in {"list_records", "search_records", "filter_records", "rank_records"}:
        if not isinstance(output, list):
            return f"{tool_name} returned {type(output).__name__}; expected list"
    if tool_name == "get_record" and output is not None and not isinstance(output, dict):
        return f"get_record returned {type(output).__name__}; expected dict or None"
    return None


def invoke_tool_for_smoke_check(
    tool_name: str,
    tool: Callable[..., Any],
    database: list[dict[str, Any]],
    constraints: Mapping[str, Any],
) -> Any:
    """Invoke one generated tool with a row-local smoke-test call.

    Args:
        tool_name: Tool name.
        tool: Callable loaded from the generated tool module.
        database: Row-local sandbox database.
        constraints: Final task constraints.

    Returns:
        The tool output.
    """
    if tool_name == "list_records":
        return tool()
    if tool_name == "search_records":
        return tool("", max_results=2)
    if tool_name == "get_record":
        record_id = database[0].get("record_id") if database else "__missing__"
        return tool(record_id)
    if tool_name == "filter_records":
        return tool(
            max_cost=constraints.get("max_cost"),
            min_score=constraints.get("min_score"),
            required_tag=constraints.get("required_tag"),
        )
    if tool_name == "rank_records":
        return tool(list(database), metric="score", descending=True)
    return tool()


def run_tool_execution_check(
    tool_name: str,
    tool: Callable[..., Any],
    database: list[dict[str, Any]],
    constraints: Mapping[str, Any],
) -> ToolExecutionCheck:
    """Execute one generated tool and validate its output shape.

    Args:
        tool_name: Tool name.
        tool: Callable loaded from the generated tool module.
        database: Row-local sandbox database.
        constraints: Final task constraints.

    Returns:
        A structured per-tool execution result.
    """
    try:
        output = invoke_tool_for_smoke_check(tool_name, tool, database, constraints)
    except Exception as exc:  # noqa: BLE001
        return ToolExecutionCheck(name=tool_name, passed=False, error=str(exc))

    error = tool_output_error(tool_name, output)
    return ToolExecutionCheck(
        name=tool_name,
        passed=error is None,
        output_type=type(output).__name__,
        output_size=output_size(output),
        error=error,
    )


def build_tools_from_namespace(
    tool_names: list[str],
    namespace: Mapping[str, Any],
) -> tuple[dict[str, Callable[..., Any]], list[str]]:
    """Build the generated tool mapping from an executed namespace.

    Args:
        tool_names: Names requested by the row artifact.
        namespace: Namespace returned by ``execute_source_module``.

    Returns:
        A tuple of callable tools and validation errors.
    """
    tools: dict[str, Callable[..., Any]] = {}
    errors: list[str] = []
    for tool_name in tool_names:
        candidate = namespace.get(tool_name)
        if not callable(candidate):
            errors.append(f"generated tool {tool_name!r} is missing or not callable")
            continue
        tools[tool_name] = candidate
    return tools, errors


def tool_names_from_specs(tool_specs: Any) -> tuple[list[str], list[str]]:
    """Extract tool names from generated tool specs.

    Args:
        tool_specs: Value from the environment tuple ``tools`` field.

    Returns:
        A tuple of tool names and validation errors.
    """
    if not isinstance(tool_specs, list):
        return [], ["environment tuple tools field must be a list"]

    tool_names: list[str] = []
    errors: list[str] = []
    for index, tool_spec in enumerate(tool_specs):
        if not isinstance(tool_spec, Mapping):
            errors.append(f"tools[{index}] must be a mapping")
            continue
        tool_name = tool_spec.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            errors.append(f"tools[{index}].name must be a non-empty string")
            continue
        tool_names.append(tool_name)
    return tool_names, errors


def run_solution_and_verifier(
    solution_source: str,
    solution_entrypoint: str,
    verifier_source: str,
    verifier_entrypoint: str,
    tools: Mapping[str, Callable[..., Any]],
    database: list[dict[str, Any]],
) -> tuple[Any | None, bool, list[str]]:
    """Execute generated solution source and validate it with generated verifier source.

    Args:
        solution_source: Python source defining the solution function.
        solution_entrypoint: Name of the solution function.
        verifier_source: Python source defining the verifier function.
        verifier_entrypoint: Name of the verifier function.
        tools: Callable tool mapping exposed to the solution.
        database: Row-local sandbox database exposed to the verifier.

    Returns:
        The solution answer, verifier decision, and collected execution errors.
    """
    errors: list[str] = []
    try:
        solution_namespace = execute_source_module(solution_source, [solution_entrypoint])
        answer = solution_namespace[solution_entrypoint](dict(tools))
    except Exception as exc:  # noqa: BLE001
        return None, False, [f"solution execution failed: {exc}"]

    try:
        verifier_namespace = execute_source_module(verifier_source, [verifier_entrypoint])
        verifier_passed = bool(verifier_namespace[verifier_entrypoint](answer, database))
    except Exception as exc:  # noqa: BLE001
        return answer, False, [f"verifier execution failed: {exc}"]

    if not verifier_passed:
        errors.append("verifier rejected generated solution answer")
    return answer, verifier_passed, errors


def run_iteration_execution_check(
    iteration: Mapping[str, Any],
    tool_namespace: Mapping[str, Any],
    database: list[dict[str, Any]],
) -> IterationExecutionCheck:
    """Execute and verify one generated task iteration.

    Args:
        iteration: Task iteration artifact from ``task_iterations``.
        tool_namespace: Executed namespace from ``tool_module_source``.
        database: Row-local sandbox database.

    Returns:
        A structured per-iteration execution result.
    """
    difficulty = str(iteration.get("difficulty", "unknown"))
    tool_names = iteration.get("tool_names", [])
    if not isinstance(tool_names, list) or not all(isinstance(tool_name, str) for tool_name in tool_names):
        return IterationExecutionCheck(
            difficulty=difficulty,
            passed=False,
            error="iteration tool_names must be a list of strings",
        )

    tools, tool_errors = build_tools_from_namespace(tool_names, tool_namespace)
    if tool_errors:
        return IterationExecutionCheck(difficulty=difficulty, passed=False, error="; ".join(tool_errors))

    answer, verifier_passed, errors = run_solution_and_verifier(
        str(iteration.get("solution_source", "")),
        "solve",
        str(iteration.get("verifier_source", "")),
        "verify",
        tools,
        database,
    )
    reference_answer = iteration.get("reference_answer")
    if reference_answer is not None and answer != reference_answer:
        errors.append("iteration answer does not match reference_answer")

    expected_passed = iteration.get("reference_solution_passed")
    if expected_passed is not None and bool(expected_passed) != verifier_passed:
        errors.append("iteration reference_solution_passed does not match verifier result")

    return IterationExecutionCheck(
        difficulty=difficulty,
        passed=not errors and verifier_passed,
        answer=answer,
        verifier_passed=verifier_passed,
        error="; ".join(errors) if errors else None,
    )


def verify_environment_tuple(environment_tuple: Mapping[str, Any]) -> RowRecordValidationResult:
    """Verify one generated environment tuple by executing all generated artifacts.

    The helper executes the generated tool module, smoke-tests every declared
    tool, runs the final generated ``solve(tools)`` function, and checks the
    result with the generated ``verify(answer, database)`` function. It also
    replays every artifact in ``task_iterations`` when present.

    Args:
        environment_tuple: Generated ``generalist-agent-env`` output value.

    Returns:
        A structured validation result with per-artifact status and errors.
    """
    errors: list[str] = []
    try:
        environment = environment_tuple["environment"]
        database = environment["database"]
        task = environment_tuple["task"]
        constraints = task["constraints"]
    except KeyError as exc:
        return failed_validation(f"environment tuple is missing required key: {exc}")

    if not isinstance(database, list):
        return failed_validation("environment.database must be a list")
    if not all(isinstance(record, dict) for record in database):
        return failed_validation("environment.database must contain dict records")
    if not isinstance(constraints, Mapping):
        return failed_validation("task.constraints must be a mapping")

    tool_names, tool_spec_errors = tool_names_from_specs(environment_tuple.get("tools"))
    errors.extend(tool_spec_errors)

    try:
        tool_namespace = execute_source_module(str(environment_tuple["tool_module_source"]), tool_names)
    except Exception as exc:  # noqa: BLE001
        return RowRecordValidationResult(
            passed=False,
            answer=None,
            verifier_passed=False,
            tools_passed=False,
            errors=[*errors, f"tool_module_source execution failed: {exc}"],
        )

    tools, tool_errors = build_tools_from_namespace(tool_names, tool_namespace)
    errors.extend(tool_errors)
    tool_checks = [
        run_tool_execution_check(tool_name, tool, database, constraints) for tool_name, tool in tools.items()
    ]
    errors.extend(
        f"tool {check.name!r} failed smoke execution: {check.error}" for check in tool_checks if not check.passed
    )
    tools_passed = not tool_spec_errors and not tool_errors and all(check.passed for check in tool_checks)

    solution = environment_tuple.get("solution", {})
    verifier = environment_tuple.get("verifier", {})
    if not isinstance(solution, Mapping) or not isinstance(verifier, Mapping):
        return RowRecordValidationResult(
            passed=False,
            answer=None,
            verifier_passed=False,
            tools_passed=tools_passed,
            tool_checks=tool_checks,
            errors=[*errors, "solution and verifier fields must be mappings"],
        )

    answer, verifier_passed, source_errors = run_solution_and_verifier(
        str(solution.get("source", "")),
        str(solution.get("entrypoint", "solve")),
        str(verifier.get("source", "")),
        str(verifier.get("entrypoint", "verify")),
        tools,
        database,
    )
    errors.extend(source_errors)

    reference_answer = environment_tuple.get("reference_answer")
    if reference_answer is not None and answer != reference_answer:
        errors.append("final solution answer does not match reference_answer")

    expected_passed = verifier.get("reference_solution_passed")
    if expected_passed is not None and bool(expected_passed) != verifier_passed:
        errors.append("verifier.reference_solution_passed does not match verifier result")

    iteration_checks: list[IterationExecutionCheck] = []
    task_iterations = environment_tuple.get("task_iterations", [])
    if isinstance(task_iterations, list):
        iteration_checks = [
            run_iteration_execution_check(iteration, tool_namespace, database)
            for iteration in task_iterations
            if isinstance(iteration, Mapping)
        ]
        errors.extend(
            f"iteration {check.difficulty!r} failed execution: {check.error}"
            for check in iteration_checks
            if not check.passed
        )
    elif task_iterations is not None:
        errors.append("task_iterations must be a list when present")

    passed = not errors and tools_passed and verifier_passed and all(check.passed for check in iteration_checks)
    return RowRecordValidationResult(
        passed=passed,
        answer=answer,
        verifier_passed=verifier_passed,
        tools_passed=tools_passed,
        tool_checks=tool_checks,
        iteration_checks=iteration_checks,
        errors=errors,
    )


def verify_row_record(row_record: Mapping[str, Any], output_column: str | None = None) -> RowRecordValidationResult:
    """Verify a Data Designer row record containing a generated environment tuple.

    Args:
        row_record: Row mapping or generated environment tuple.
        output_column: Optional column name containing the generated tuple.

    Returns:
        A structured validation result.
    """
    try:
        environment_tuple = extract_environment_tuple(row_record, output_column)
    except (KeyError, TypeError, ValueError) as exc:
        return failed_validation(str(exc))
    return verify_environment_tuple(environment_tuple)
