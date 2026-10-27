# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from data_designer_gym.conversion import export_scenarios, normalize_rollouts


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone plugin CLI parser."""
    parser = argparse.ArgumentParser(prog="data-designer-gym")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Translate canonical scenarios to Gym task JSONL")
    export.add_argument("input", type=Path)
    export.add_argument("--output", type=Path, required=True)

    ingest = subparsers.add_parser("ingest", help="Join Gym outputs back to Data Designer scenario IDs")
    ingest.add_argument("--tasks", type=Path, required=True)
    ingest.add_argument("--rollouts", type=Path, required=True)
    ingest.add_argument("--failures", type=Path)
    ingest.add_argument("--output", type=Path, required=True)
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """Run one CLI command and return its process status."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export":
            print(export_scenarios(args.input, args.output))
        else:
            print(
                normalize_rollouts(
                    args.tasks,
                    args.rollouts,
                    args.output,
                    failures_path=args.failures,
                )
            )
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def main() -> None:
    """Console-script entry point."""
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
