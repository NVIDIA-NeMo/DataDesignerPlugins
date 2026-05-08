# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print release-eligible owners from a per-plugin CODEOWNERS file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ddp.codeowners import github_release_owners, owner_tokens_for_codeowners_path


def main(args: list[str] | None = None) -> int:
    """Print parsed CODEOWNERS tokens for release workflows.

    Args:
        args: CLI arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="release-owners",
        description="Print owner tokens parsed from a per-plugin CODEOWNERS file.",
    )
    parser.add_argument("codeowners_path", type=Path, help="Path to a per-plugin CODEOWNERS file")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print all owner tokens instead of only GitHub user/team tokens.",
    )
    parsed = parser.parse_args(args)

    owners = owner_tokens_for_codeowners_path(parsed.codeowners_path)
    output_owners = owners if parsed.all else github_release_owners(owners)
    print(" ".join(output_owners))
    return 0


if __name__ == "__main__":
    sys.exit(main())
