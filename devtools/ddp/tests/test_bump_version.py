# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ddp.bump_version."""

from __future__ import annotations

from pathlib import Path

import pytest

from ddp._repo import find_repo_root
from ddp.bump_version import (
    bump_semver,
    main,
    parse_semver,
    read_version,
    replace_version_in_file,
)


class TestParseSemver:
    """Tests for parse_semver."""

    def test_simple_version(self) -> None:
        assert parse_semver("1.2.3") == (1, 2, 3)

    def test_zero_version(self) -> None:
        assert parse_semver("0.0.0") == (0, 0, 0)

    def test_large_numbers(self) -> None:
        assert parse_semver("10.20.300") == (10, 20, 300)

    def test_prerelease_suffix_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_semver("1.2.3-rc1")

    def test_prerelease_plus_suffix_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_semver("1.2.3+build42")

    def test_prerelease_dot_suffix_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_semver("1.2.3.dev0")

    def test_malformed_version_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_semver("not-a-version")

    def test_two_part_version_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_semver("1.2")


class TestBumpSemver:
    """Tests for bump_semver."""

    def test_patch_bump(self) -> None:
        assert bump_semver(1, 2, 3, "patch") == (1, 2, 4)

    def test_minor_bump_resets_patch(self) -> None:
        assert bump_semver(1, 2, 3, "minor") == (1, 3, 0)

    def test_major_bump_resets_minor_and_patch(self) -> None:
        assert bump_semver(1, 2, 3, "major") == (2, 0, 0)

    def test_patch_from_zero(self) -> None:
        assert bump_semver(0, 0, 0, "patch") == (0, 0, 1)

    def test_minor_from_zero(self) -> None:
        assert bump_semver(0, 0, 0, "minor") == (0, 1, 0)

    def test_major_from_zero(self) -> None:
        assert bump_semver(0, 0, 0, "major") == (1, 0, 0)


class TestReplaceVersionInFile:
    """Tests for replace_version_in_file."""

    def test_replaces_version_preserving_format(self, tmp_path: Path) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text('[project]\nname = "my-plugin"\nversion = "0.1.0"\ndescription = "A plugin"\n')
        replace_version_in_file(toml, "0.1.0", "0.2.0")
        content = toml.read_text()
        assert 'version = "0.2.0"' in content
        assert 'name = "my-plugin"' in content
        assert 'description = "A plugin"' in content

    def test_wrong_version_raises(self, tmp_path: Path) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text('[project]\nversion = "0.1.0"\n')
        with pytest.raises(SystemExit):
            replace_version_in_file(toml, "9.9.9", "9.9.10")

    def test_preserves_other_version_strings(self, tmp_path: Path) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text('[project]\nversion = "1.0.0"\n\n[tool.something]\nversion = "2.0.0"\n')
        replace_version_in_file(toml, "1.0.0", "1.0.1")
        content = toml.read_text()
        assert 'version = "1.0.1"' in content
        assert 'version = "2.0.0"' in content


class TestReadVersion:
    """Tests for read_version."""

    def test_reads_template_plugin_version(self) -> None:
        root = find_repo_root()
        toml_path = root / "plugins" / "data-designer-template" / "pyproject.toml"
        version = read_version(toml_path)
        assert parse_semver(version) is not None

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            read_version(tmp_path / "nonexistent.toml")

    def test_missing_version_key_exits(self, tmp_path: Path) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text('[project]\nname = "no-version"\n')
        with pytest.raises(SystemExit):
            read_version(toml)


class TestMainCli:
    """End-to-end tests for the main CLI entry point."""

    def _make_plugin(self, tmp_path: Path, name: str, version: str) -> Path:
        """Create a minimal plugin directory structure for testing."""
        plugin_dir = tmp_path / "plugins" / name
        plugin_dir.mkdir(parents=True)
        toml = plugin_dir / "pyproject.toml"
        toml.write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n')
        # Repo root marker
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        return toml

    def test_patch_bump(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        toml = self._make_plugin(tmp_path, "data-designer-test", "1.0.0")
        monkeypatch.setattr("ddp.bump_version.find_repo_root", lambda: tmp_path)
        result = main(["data-designer-test", "patch"])
        assert result == 0
        assert 'version = "1.0.1"' in toml.read_text()

    def test_minor_bump(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        toml = self._make_plugin(tmp_path, "data-designer-test", "0.3.7")
        monkeypatch.setattr("ddp.bump_version.find_repo_root", lambda: tmp_path)
        result = main(["data-designer-test", "minor"])
        assert result == 0
        assert 'version = "0.4.0"' in toml.read_text()

    def test_major_bump(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        toml = self._make_plugin(tmp_path, "data-designer-test", "2.5.1")
        monkeypatch.setattr("ddp.bump_version.find_repo_root", lambda: tmp_path)
        result = main(["data-designer-test", "major"])
        assert result == 0
        assert 'version = "3.0.0"' in toml.read_text()

    def test_missing_plugin_exits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "plugins").mkdir()
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        monkeypatch.setattr("ddp.bump_version.find_repo_root", lambda: tmp_path)
        with pytest.raises(SystemExit):
            main(["data-designer-nonexistent", "patch"])
