# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared repository utilities for monorepo management tools."""

from __future__ import annotations

import sys
from pathlib import Path

SPDX_HEADER = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0"""


def find_repo_root() -> Path:
    """Walk up from CWD to find the monorepo root.

    The root is identified by the presence of both a ``plugins/``
    directory and a ``pyproject.toml`` file.

    Returns:
        Path to the monorepo root directory.
    """
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "plugins").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return cwd


def load_toml(path: Path) -> dict:
    """Load a TOML file using ``tomllib`` (3.11+) or ``tomli`` fallback.

    Args:
        path: Path to the TOML file.

    Returns:
        Parsed TOML data as a dictionary.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            print(
                "Error: Python 3.11+ (tomllib) or the 'tomli' package is required.\nInstall with: pip install tomli",
                file=sys.stderr,
            )
            sys.exit(1)
    with open(path, "rb") as f:
        return tomllib.load(f)
