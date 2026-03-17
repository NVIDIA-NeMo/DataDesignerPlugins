# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dd_plugins_core.validate_release."""

from __future__ import annotations

import sys

from dd_plugins_core.validate_release import REQUIRED_FIELDS, main


def test_valid_template_plugin_passes(monkeypatch: object) -> None:
    """validate_release should pass for the template plugin with its actual version."""
    from dd_plugins_core._repo import find_repo_root, load_toml

    root = find_repo_root()
    data = load_toml(root / "plugins" / "data-designer-template" / "pyproject.toml")
    version = data["project"]["version"]

    monkeypatch.setattr(sys, "argv", ["validate-release", "data-designer-template", version])  # type: ignore[attr-defined]
    result = main()
    assert result == 0


def test_wrong_version_fails(monkeypatch: object) -> None:
    monkeypatch.setattr(sys, "argv", ["validate-release", "data-designer-template", "99.99.99"])  # type: ignore[attr-defined]
    result = main()
    assert result == 1


def test_required_fields_are_present() -> None:
    assert "description" in REQUIRED_FIELDS
    assert "license" in REQUIRED_FIELDS
    assert "readme" in REQUIRED_FIELDS
    assert "authors" in REQUIRED_FIELDS
