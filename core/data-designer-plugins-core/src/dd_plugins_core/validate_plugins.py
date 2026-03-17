# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate all installed Data Designer plugins via assert_valid_plugin."""

from __future__ import annotations

import importlib.metadata
import sys


def main() -> None:
    """Discover and validate all installed data_designer.plugins entry points."""
    from data_designer.engine.testing.utils import assert_valid_plugin

    eps = importlib.metadata.entry_points(group="data_designer.plugins")
    if not eps:
        print("ERROR: no plugins found in data_designer.plugins group")
        sys.exit(1)

    failed = False
    for ep in eps:
        try:
            plugin = ep.load()
            assert_valid_plugin(plugin)
            print(f"OK: {ep.name}")
        except Exception as exc:
            print(f"FAIL: {ep.name} — {exc}")
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
