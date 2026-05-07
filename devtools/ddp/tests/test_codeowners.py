# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ddp.codeowners."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from ddp.codeowners import github_release_owners, main, owner_tokens_from_codeowners_text


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


def test_owner_tokens_from_codeowners_text_reads_supported_owner_forms() -> None:
    output = owner_tokens_from_codeowners_text(
        """
        # Plugin owners
        * @acme/platform release@example.test @octocat
        fallback@example.test
        """
    )

    assert output == ["@acme/platform", "release@example.test", "@octocat", "fallback@example.test"]


def test_github_release_owners_filters_email_owners() -> None:
    owners = github_release_owners(["release@example.test", "@octocat", "@acme/platform"])

    assert owners == ["@octocat", "@acme/platform"]
