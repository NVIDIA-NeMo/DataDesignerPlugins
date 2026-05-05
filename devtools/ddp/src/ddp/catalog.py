# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible alias for generated plugin documentation."""

from __future__ import annotations

import sys

from ddp.plugin_docs import main

if __name__ == "__main__":
    sys.exit(main())
