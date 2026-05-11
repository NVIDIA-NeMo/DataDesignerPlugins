# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Hashable, Sequence
from enum import Enum

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SANDBOX_MAX_RETRIES = 5
SANDBOX_INITIAL_RETRY_DELAY_SEC = 0.5
SANDBOX_MAX_RETRY_DELAY_SEC = 30.0
SANDBOX_REQUEST_TIMEOUT_SEC = 60.0

SLOTS_INITIAL = 64
SLOTS_MIN = 16
SLOTS_MAX = 2048
SLOTS_SHRINK_FACTOR = 0.95
SLOTS_SUCCESS_THRESHOLD = 16
SLOTS_GROW_STEP = 1
SLOTS_JITTER_SEC = 0.001

EXIT_CODE_EMPTY_INPUT = -2

HTTP_STATUS_RATE_LIMITED = 429
HTTP_STATUS_CLIENT_ERROR = 400
HTTP_STATUS_SERVER_ERROR = 500

ERROR_TEXT_MAX_LENGTH = 200
STATS_LOG_INTERVAL_SEC = 5.0


class SandboxStats:
    """Async-safe stats tracker for sandbox execution."""

    def __init__(self, column_name: str = "") -> None:
        self.column_name = column_name
        self.code_success_count = 0
        self.code_error_count = 0
        self.api_error_count = 0
        self.rate_limit_count = 0
        self.walltime_sec = 0.0
        self._interval_code_successes = 0
        self._interval_code_errors = 0
        self._interval_api_errors = 0
        self._interval_rate_limits = 0
        self._lock = asyncio.Lock()

    async def record_result(self, exit_code: int) -> None:
        """Record a completed API call based on the code exit code.

        Args:
            exit_code: Process exit code from Piston.
        """
        async with self._lock:
            if exit_code == 0:
                self.code_success_count += 1
                self._interval_code_successes += 1
            else:
                self.code_error_count += 1
                self._interval_code_errors += 1

    async def record_api_error(self) -> None:
        """Record a sandbox API-level failure."""
        async with self._lock:
            self.api_error_count += 1
            self._interval_api_errors += 1

    async def record_rate_limit(self) -> None:
        """Record a sandbox 429 response."""
        async with self._lock:
            self.rate_limit_count += 1
            self._interval_rate_limits += 1

    async def snapshot_interval(self) -> tuple[int, int, int, int]:
        """Return and reset interval counters.

        Returns:
            Counts for code successes, code errors, API errors, and rate limits.
        """
        async with self._lock:
            snapshot = (
                self._interval_code_successes,
                self._interval_code_errors,
                self._interval_api_errors,
                self._interval_rate_limits,
            )
            self._interval_code_successes = 0
            self._interval_code_errors = 0
            self._interval_api_errors = 0
            self._interval_rate_limits = 0
            return snapshot

    def set_walltime(self, walltime_sec: float) -> None:
        """Set total wall-clock execution time in seconds."""
        self.walltime_sec = walltime_sec

    @property
    def request_count(self) -> int:
        """Return total completed requests."""
        return self.code_success_count + self.code_error_count + self.api_error_count

    @property
    def api_success_count(self) -> int:
        """Return API calls that returned a valid Piston response."""
        return self.code_success_count + self.code_error_count

    @property
    def requests_per_minute(self) -> float:
        """Return completed request throughput."""
        if self.walltime_sec <= 0:
            return 0.0
        return 60 * (self.request_count / self.walltime_sec)

    @property
    def api_success_rate(self) -> float:
        """Return percentage of requests where the API call succeeded."""
        if self.request_count <= 0:
            return 0.0
        return 100 * (self.api_success_count / self.request_count)

    @property
    def code_success_rate(self) -> float:
        """Return percentage of API-successful calls where code exited with zero."""
        if self.api_success_count <= 0:
            return 0.0
        return 100 * (self.code_success_count / self.api_success_count)

    @property
    def avg_time_per_request_ms(self) -> float:
        """Return average wall-clock milliseconds per completed request."""
        if self.request_count <= 0:
            return 0.0
        return 1000 * (self.walltime_sec / self.request_count)

    def to_dict(self) -> dict[str, str | int | float]:
        """Return stats as a JSON-serializable dictionary."""
        return {
            "column_name": self.column_name,
            "request_count": self.request_count,
            "code_success_count": self.code_success_count,
            "code_error_count": self.code_error_count,
            "api_error_count": self.api_error_count,
            "rate_limit_count": self.rate_limit_count,
            "walltime_sec": round(self.walltime_sec, 2),
            "requests_per_minute": round(self.requests_per_minute, 1),
            "api_success_rate": round(self.api_success_rate, 1),
            "code_success_rate": round(self.code_success_rate, 1),
            "avg_time_per_request_ms": round(self.avg_time_per_request_ms, 1),
        }

    def to_json(self) -> str:
        """Return stats as compact JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)


class PistonStatus(str, Enum):
    """Piston execution status codes."""

    RUNTIME_ERROR = "RE"
    SIGNAL = "SG"
    TIMEOUT = "TO"
    OUTPUT_EXCEEDED = "OL"
    ERROR_EXCEEDED = "EL"
    INTERNAL_ERROR = "XX"

    @classmethod
    def from_string(cls, value: str | None) -> PistonStatus | None:
        """Convert a raw Piston status string to an enum.

        Args:
            value: Raw Piston status value.

        Returns:
            Matching status enum, or ``None`` for unknown or empty values.
        """
        if value is None:
            return None
        try:
            return cls(value)
        except ValueError:
            logger.warning("Unknown Piston status code: %r", value)
            return None


class SandboxOutput(BaseModel):
    """Structured output from Piston code sandbox execution."""

    stdout: str = Field(default="", description="Standard output from code execution.")
    stderr: str = Field(default="", description="Standard error from code execution.")
    output: str = Field(default="", description="Combined standard output and error.")
    exit_code: int = Field(default=-1, description="Process exit code.")
    signal: str | None = Field(default=None, description="Signal that terminated the process.")
    message: str | None = Field(default=None, description="Human-readable failure message.")
    status: PistonStatus | None = Field(default=None, description="Piston status code.")
    cpu_time: float | None = Field(default=None, description="CPU time in milliseconds.")
    wall_time: float | None = Field(default=None, description="Wall time in milliseconds.")
    memory: int | None = Field(default=None, description="Memory used in bytes.")


class AdaptiveSlotController:
    """Adaptive concurrency controller for Piston traffic."""

    def __init__(
        self,
        initial_slots: int = SLOTS_INITIAL,
        min_slots: int = SLOTS_MIN,
        max_slots: int = SLOTS_MAX,
        shrink_factor: float = SLOTS_SHRINK_FACTOR,
        grow_step: int = SLOTS_GROW_STEP,
        success_threshold: int = SLOTS_SUCCESS_THRESHOLD,
    ) -> None:
        self._current_limit = max(min_slots, min(initial_slots, max_slots))
        self._min_slots = min_slots
        self._max_slots = max_slots
        self._shrink_factor = shrink_factor
        self._grow_step = grow_step
        self._success_threshold = success_threshold
        self._success_streak = 0
        self._inflight = 0
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        """Acquire one execution slot."""
        async with self._cond:
            while self._inflight >= self._current_limit:
                await self._cond.wait()
            self._inflight += 1

    async def release(self) -> None:
        """Release one execution slot."""
        async with self._cond:
            self._inflight = max(0, self._inflight - 1)
            self._cond.notify_all()

    async def on_overload(self) -> int:
        """Shrink concurrency after an overload signal.

        Returns:
            The new concurrency limit.
        """
        async with self._cond:
            new_limit = max(self._min_slots, int(self._current_limit * self._shrink_factor))
            self._current_limit = max(self._min_slots, new_limit)
            self._success_streak = 0
            self._cond.notify_all()
            return self._current_limit

    async def on_success(self) -> int:
        """Grow concurrency slowly after sustained success.

        Returns:
            The current concurrency limit.
        """
        async with self._cond:
            self._success_streak += 1
            if self._success_streak >= self._success_threshold:
                if self._current_limit < self._max_slots:
                    self._current_limit = min(self._max_slots, self._current_limit + self._grow_step)
                self._success_streak = 0
                self._cond.notify_all()
            return self._current_limit

    @property
    def stats(self) -> dict[str, int]:
        """Return current slot controller counters."""
        return {
            "limit": self._current_limit,
            "inflight": self._inflight,
            "success_streak": self._success_streak,
        }


def normalize_piston_url(sandbox_url: str) -> str:
    """Normalize a Piston API base URL.

    Args:
        sandbox_url: HTTP(S) base URL.

    Returns:
        URL without trailing slash.
    """
    return sandbox_url.rstrip("/")


def build_execute_payload(
    code: str,
    language: str,
    version: str = "*",
    run_timeout: int = 10000,
    run_cpu_time: int = 3000,
    compile_timeout: int = 10000,
    compile_cpu_time: int = 3000,
    stdin: str = "",
    args: list[str] | None = None,
) -> dict[str, object]:
    """Build a Piston ``/api/v2/execute`` request payload."""
    return {
        "language": language,
        "version": version,
        "files": [{"content": code}],
        "stdin": stdin,
        "args": args or [],
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
        "compile_cpu_time": compile_cpu_time,
        "run_cpu_time": run_cpu_time,
        "compile_memory_limit": -1,
        "run_memory_limit": -1,
    }


def parse_execute_response(response_data: dict[str, object]) -> SandboxOutput:
    """Parse a Piston execute response into ``SandboxOutput``.

    Args:
        response_data: JSON response from Piston.

    Returns:
        Structured sandbox output.
    """
    raw_run_info = response_data.get("run", {}) or {}
    run_info = raw_run_info if isinstance(raw_run_info, dict) else {}
    exit_code = run_info.get("code")
    final_exit_code = exit_code if isinstance(exit_code, int) else -1
    return SandboxOutput(
        stdout=str(run_info.get("stdout", "")),
        stderr=str(run_info.get("stderr", "")),
        output=str(run_info.get("output", "")),
        exit_code=final_exit_code,
        signal=run_info.get("signal") if isinstance(run_info.get("signal"), str) else None,
        message=run_info.get("message") if isinstance(run_info.get("message"), str) else None,
        status=PistonStatus.from_string(run_info.get("status") if isinstance(run_info.get("status"), str) else None),
        cpu_time=run_info.get("cpu_time") if isinstance(run_info.get("cpu_time"), int | float) else None,
        wall_time=run_info.get("wall_time") if isinstance(run_info.get("wall_time"), int | float) else None,
        memory=run_info.get("memory") if isinstance(run_info.get("memory"), int) else None,
    )


def response_has_exit_or_signal(response_data: dict[str, object]) -> bool:
    """Return whether a Piston response contains a terminal execution marker."""
    raw_run_info = response_data.get("run", {}) or {}
    if not isinstance(raw_run_info, dict):
        return False
    return raw_run_info.get("code") is not None or raw_run_info.get("signal") is not None


def serialize_sandbox_output(result: SandboxOutput, fields: Sequence[str]) -> str:
    """Serialize selected output fields as JSON.

    Args:
        result: Sandbox output.
        fields: Field names to include.

    Returns:
        JSON object string containing selected fields.
    """
    payload = result.model_dump(mode="json")
    filtered = {field: payload[field] for field in fields}
    return json.dumps(filtered)


async def execute_code_in_sandbox(
    session: aiohttp.ClientSession,
    slots: AdaptiveSlotController,
    sandbox_url: str,
    code: str,
    language: str,
    version: str = "*",
    run_timeout: int = 10000,
    run_cpu_time: int = 3000,
    compile_timeout: int = 10000,
    compile_cpu_time: int = 3000,
    stdin: str = "",
    args: list[str] | None = None,
    stats: SandboxStats | None = None,
    row_id: Hashable = "<unknown>",
) -> SandboxOutput:
    """Execute code in a Piston sandbox with retry and backpressure handling.

    Args:
        session: Shared aiohttp session.
        slots: Shared adaptive slot controller.
        sandbox_url: Piston API base URL.
        code: Source code to execute.
        language: Piston runtime language.
        version: Piston runtime version selector.
        run_timeout: Run wall-time limit in milliseconds.
        run_cpu_time: Run CPU-time limit in milliseconds.
        compile_timeout: Compile wall-time limit in milliseconds.
        compile_cpu_time: Compile CPU-time limit in milliseconds.
        stdin: Standard input text.
        args: Command-line arguments.
        stats: Optional stats tracker.
        row_id: Identifier used in log messages.

    Returns:
        Structured sandbox output.
    """
    payload = build_execute_payload(
        code=code,
        language=language,
        version=version,
        run_timeout=run_timeout,
        run_cpu_time=run_cpu_time,
        compile_timeout=compile_timeout,
        compile_cpu_time=compile_cpu_time,
        stdin=stdin,
        args=args,
    )
    url = f"{normalize_piston_url(sandbox_url)}/api/v2/execute"
    last_exception: Exception | None = None
    retry_delay = SANDBOX_INITIAL_RETRY_DELAY_SEC
    attempt = 0

    while attempt < SANDBOX_MAX_RETRIES:
        try:
            await slots.acquire()
            await asyncio.sleep(SLOTS_JITTER_SEC)

            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=SANDBOX_REQUEST_TIMEOUT_SEC),
            ) as response:
                if response.status == HTTP_STATUS_RATE_LIMITED:
                    attempt += 1
                    slot_limit = await slots.on_overload()
                    if stats:
                        await stats.record_rate_limit()
                    error_text = await response.text()
                    logger.warning(
                        "Rate limited for row %s (attempt %s/%s); slots -> %s: %s",
                        row_id,
                        attempt,
                        SANDBOX_MAX_RETRIES,
                        slot_limit,
                        error_text[:ERROR_TEXT_MAX_LENGTH],
                    )
                    last_exception = aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=error_text,
                    )
                    if attempt < SANDBOX_MAX_RETRIES:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, SANDBOX_MAX_RETRY_DELAY_SEC)
                        continue
                    raise last_exception

                if response.status >= HTTP_STATUS_SERVER_ERROR:
                    attempt += 1
                    error_text = await response.text()
                    logger.warning(
                        "Server error %s for row %s (attempt %s/%s): %s",
                        response.status,
                        row_id,
                        attempt,
                        SANDBOX_MAX_RETRIES,
                        error_text[:ERROR_TEXT_MAX_LENGTH],
                    )
                    last_exception = aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=error_text,
                    )
                    if attempt < SANDBOX_MAX_RETRIES:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, SANDBOX_MAX_RETRY_DELAY_SEC)
                        continue
                    raise last_exception

                if response.status >= HTTP_STATUS_CLIENT_ERROR:
                    error_text = await response.text()
                    logger.error(
                        "Client error %s for row %s: %s",
                        response.status,
                        row_id,
                        error_text[:ERROR_TEXT_MAX_LENGTH],
                    )
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=error_text,
                    )

                response_data = await response.json()
                if not response_has_exit_or_signal(response_data):
                    attempt += 1
                    error_msg = f"Sandbox API returned invalid data for row {row_id}"
                    logger.warning("%s (attempt %s/%s)", error_msg, attempt, SANDBOX_MAX_RETRIES)
                    last_exception = RuntimeError(error_msg)
                    if attempt < SANDBOX_MAX_RETRIES:
                        continue
                    raise last_exception

                await slots.on_success()
                output = parse_execute_response(response_data)
                if stats:
                    await stats.record_result(output.exit_code)
                return output

        except aiohttp.ClientError as err:
            attempt += 1
            last_exception = err
            logger.warning(
                "Request error for row %s (attempt %s/%s): %r",
                row_id,
                attempt,
                SANDBOX_MAX_RETRIES,
                err,
            )
            if attempt < SANDBOX_MAX_RETRIES:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, SANDBOX_MAX_RETRY_DELAY_SEC)
                continue
            raise
        except TimeoutError as err:
            attempt += 1
            last_exception = err
            slot_limit = await slots.on_overload()
            logger.warning(
                "Timeout for row %s (attempt %s/%s); slots -> %s", row_id, attempt, SANDBOX_MAX_RETRIES, slot_limit
            )
            if attempt < SANDBOX_MAX_RETRIES:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, SANDBOX_MAX_RETRY_DELAY_SEC)
                continue
            raise
        finally:
            await slots.release()

    if last_exception:
        raise last_exception
    raise RuntimeError("Sandbox request failed without exception")
