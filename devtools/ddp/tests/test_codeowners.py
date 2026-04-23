# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ddp.codeowners."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from ddp.codeowners import main


def test_main_produces_infrastructure_and_plugins_sections() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    output = buf.getvalue()
    assert "# Infrastructure" in output
    assert "# Plugins" in output
    assert "/devtools/ @NVIDIA-NeMo/data_designer_reviewers" in output


def test_main_output_does_not_reference_tools_dir() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    output = buf.getvalue()
    assert "/tools/" not in output
