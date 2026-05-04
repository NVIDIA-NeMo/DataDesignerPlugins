# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ddp.catalog."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from ddp.catalog import main


def test_main_produces_markdown_table() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    output = buf.getvalue()
    assert "# Plugin Catalog" in output
    assert 'class="plugin-catalog"' in output
    assert "Auto-generated from plugin metadata" not in output


def test_main_includes_template_plugin() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    output = buf.getvalue()
    assert "data-designer-template" in output
