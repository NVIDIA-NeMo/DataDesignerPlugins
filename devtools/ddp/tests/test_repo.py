# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ddp._repo shared utilities."""

from __future__ import annotations

from pathlib import Path

from ddp._repo import SPDX_HEADER, find_repo_root, load_toml


def test_find_repo_root_returns_directory_with_plugins() -> None:
    root = find_repo_root()
    assert (root / "plugins").is_dir()
    assert (root / "pyproject.toml").is_file()


def test_load_toml_parses_known_file() -> None:
    root = find_repo_root()
    data = load_toml(root / "pyproject.toml")
    assert data["project"]["name"] == "data-designer-plugins-workspace"


def test_spdx_header_contains_expected_tags() -> None:
    assert "SPDX-FileCopyrightText" in SPDX_HEADER
    assert "SPDX-License-Identifier" in SPDX_HEADER
    assert "Apache-2.0" in SPDX_HEADER


def test_spdx_header_is_two_comment_lines() -> None:
    lines = SPDX_HEADER.strip().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("#") for line in lines)


def test_find_repo_root_is_absolute() -> None:
    root = find_repo_root()
    assert root.is_absolute()


def test_load_toml_returns_dict() -> None:
    root = find_repo_root()
    ddp_toml = root / "devtools" / "ddp" / "pyproject.toml"
    data = load_toml(ddp_toml)
    assert isinstance(data, dict)
    assert data["project"]["name"] == "ddp"


def test_load_toml_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.toml"
    try:
        load_toml(missing)
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass
