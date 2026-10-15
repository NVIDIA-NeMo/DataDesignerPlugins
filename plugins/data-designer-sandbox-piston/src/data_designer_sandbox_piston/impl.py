# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Coroutine, Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import aiohttp
import pandas as pd
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from data_designer_sandbox_piston.client import (
    EXIT_CODE_EMPTY_INPUT,
    SANDBOX_MAX_RETRIES,
    SLOTS_MAX,
    STATS_LOG_INTERVAL_SEC,
    AdaptiveSlotController,
    SandboxOutput,
    SandboxStats,
    execute_code_in_sandbox,
)
from data_designer_sandbox_piston.config import CodeSandboxColumnConfig

if TYPE_CHECKING:
    from pandas import DataFrame

logger = logging.getLogger(__name__)
AsyncResultT = TypeVar("AsyncResultT")


@dataclass(frozen=True)
class SandboxRowRequest:
    """One DataFrame row prepared for sandbox execution."""

    position: int
    index: Hashable
    code: str | None


def stringify_code_cell(value: object) -> str | None:
    """Convert a DataFrame cell into source code or ``None`` for empty input.

    Args:
        value: Raw DataFrame cell value.

    Returns:
        Source code text, or ``None`` when the value is missing or blank.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    if not text.strip():
        return None
    return text


def build_row_requests(data: DataFrame, target_column: str) -> list[SandboxRowRequest]:
    """Create ordered sandbox requests from a DataFrame.

    Args:
        data: Input DataFrame.
        target_column: Column containing code to execute.

    Returns:
        One request per row in original DataFrame order.
    """
    requests: list[SandboxRowRequest] = []
    for position, (df_index, row) in enumerate(data.iterrows()):
        series = row
        requests.append(
            SandboxRowRequest(
                position=position,
                index=df_index,
                code=stringify_code_cell(series[target_column]),
            )
        )
    return requests


def empty_code_output() -> SandboxOutput:
    """Return the standard result for empty source code."""
    return SandboxOutput(
        stdout="",
        stderr="Code is empty or missing",
        exit_code=EXIT_CODE_EMPTY_INPUT,
        message="Code is empty or missing",
    )


def api_error_output(error: Exception) -> SandboxOutput:
    """Return the standard result for sandbox API failures.

    Args:
        error: The final exception after retries are exhausted.

    Returns:
        Structured sandbox output for a failed request.
    """
    return SandboxOutput(
        stdout="",
        stderr=f"Execution failed: {error}",
        exit_code=-1,
        message=f"Execution failed after {SANDBOX_MAX_RETRIES} retries: {error}",
    )


async def log_stats_periodically(stats: SandboxStats, slots: AdaptiveSlotController) -> None:
    """Log interval sandbox execution metrics until cancelled.

    Args:
        stats: Shared execution stats.
        slots: Shared adaptive slot controller.
    """
    while True:
        await asyncio.sleep(STATS_LOG_INTERVAL_SEC)
        code_successes, code_errors, api_errors, rate_limits = await stats.snapshot_interval()
        api_successes = code_successes + code_errors
        interval_requests = api_successes + api_errors
        if interval_requests > 0 or rate_limits > 0:
            code_success_rate = 100 * code_successes / api_successes if api_successes else 0
            slot_info = slots.stats
            logger.info(
                "Sandbox stats: %d requests, %d total (%.1f%% code success, %d code errors, "
                "%d api errors, %d rate-limited), slots=%d/%d",
                interval_requests,
                stats.request_count,
                code_success_rate,
                code_errors,
                api_errors,
                rate_limits,
                slot_info["inflight"],
                slot_info["limit"],
            )


async def execute_row_request(
    request: SandboxRowRequest,
    config: CodeSandboxColumnConfig,
    session: aiohttp.ClientSession,
    slots: AdaptiveSlotController,
    stats: SandboxStats,
) -> tuple[int, SandboxOutput]:
    """Execute one prepared row request.

    Args:
        request: Prepared row request.
        config: Column configuration.
        session: Shared HTTP session.
        slots: Shared adaptive slot controller.
        stats: Shared stats tracker.

    Returns:
        The request position and structured sandbox output.
    """
    if request.code is None:
        await stats.record_result(EXIT_CODE_EMPTY_INPUT)
        return request.position, empty_code_output()

    try:
        output = await execute_code_in_sandbox(
            session=session,
            slots=slots,
            sandbox_url=config.sandbox_url or "",
            code=request.code,
            language=config.language,
            version=config.version,
            run_timeout=config.run_timeout,
            run_cpu_time=config.run_cpu_time,
            compile_timeout=config.compile_timeout,
            compile_cpu_time=config.compile_cpu_time,
            stdin=config.stdin,
            args=config.args,
            stats=stats,
            row_id=request.index,
        )
    except Exception as err:
        await stats.record_api_error()
        logger.exception("Failed to execute code for row %s after all retries", request.index)
        output = api_error_output(err)
    return request.position, output


async def execute_row_requests(
    requests: list[SandboxRowRequest],
    config: CodeSandboxColumnConfig,
    stats: SandboxStats,
    slots: AdaptiveSlotController,
) -> list[SandboxOutput]:
    """Execute all row requests asynchronously while preserving input order.

    Args:
        requests: Prepared row requests.
        config: Column configuration.
        stats: Shared stats tracker.
        slots: Shared adaptive slot controller.

    Returns:
        Structured outputs in DataFrame row order.
    """
    results: list[SandboxOutput | None] = [None] * len(requests)
    if not requests:
        return []

    connector = aiohttp.TCPConnector(limit=SLOTS_MAX, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        stats_task = asyncio.create_task(log_stats_periodically(stats, slots))
        try:
            tasks = [
                asyncio.create_task(execute_row_request(request, config, session, slots, stats)) for request in requests
            ]
            for position, output in await asyncio.gather(*tasks):
                results[position] = output
        finally:
            stats_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stats_task

    return [
        result if result is not None else api_error_output(RuntimeError("missing sandbox result")) for result in results
    ]


def run_coroutine_sync(coro: Coroutine[Any, Any, AsyncResultT]) -> AsyncResultT:
    """Run a coroutine from sync generator code, including inside notebooks.

    Args:
        coro: Coroutine to execute.

    Returns:
        The coroutine result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[AsyncResultT] = []
    errors: list[BaseException] = []

    def run_in_thread() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as err:  # noqa: BLE001
            errors.append(err)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0]


class CodeSandboxColumnGenerator(ColumnGeneratorFullColumn[CodeSandboxColumnConfig]):
    """Generate a column by executing source code through a Piston sandbox."""

    @property
    def can_generate_from_scratch(self) -> bool:
        """Return whether this generator can run without upstream data."""
        return False

    def log_pre_generation(self) -> None:
        """Log the configured sandbox target before generation starts."""
        logger.info("Preparing code-sandbox column generation for %s", self.config.name)

    def generate(self, data: DataFrame) -> DataFrame:
        """Execute code from the target column and write structured results.

        Args:
            data: Input DataFrame.

        Returns:
            The same DataFrame with the configured output column added.

        Raises:
            ValueError: If ``sandbox_url`` is missing or the target column does
                not exist.
        """
        sandbox_url = self.config.sandbox_url
        if not sandbox_url:
            raise ValueError("sandbox_url must be set to a running Piston API endpoint.")
        if self.config.target_column not in data.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in dataframe")

        logger.info("Generating column %s using Piston sandbox at %s", self.config.name, sandbox_url)

        requests = build_row_requests(data, self.config.target_column)
        stats = SandboxStats(column_name=self.config.name)
        slots = AdaptiveSlotController()
        start_time = time.perf_counter()
        outputs = run_coroutine_sync(execute_row_requests(requests, self.config, stats, slots))
        end_time = time.perf_counter()

        data[self.config.name] = [output.model_dump(mode="json") for output in outputs]
        stats.set_walltime(end_time - start_time)
        logger.info("Sandbox execution complete for column '%s': %s", self.config.name, stats.to_json())
        return data
