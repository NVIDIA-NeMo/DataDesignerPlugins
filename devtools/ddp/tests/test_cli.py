# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified ``ddp`` CLI dispatcher."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from ddp.cli import _run_new, build_parser


class TestBuildParser:
    """Tests for build_parser producing correct subcommands."""

    EXPECTED_COMMANDS = (
        "new",
        "plugin-docs",
        "sync",
        "codeowners",
        "license-headers",
        "validate",
        "check-release",
        "bump",
    )

    def test_all_subcommands_registered(self) -> None:
        parser = build_parser()
        # Parse --help would exit, so check the subparsers action directly
        subparsers_actions = [a for a in parser._subparsers._actions if hasattr(a, "_parser_class")]
        assert len(subparsers_actions) == 1
        choices = subparsers_actions[0].choices
        for cmd in self.EXPECTED_COMMANDS:
            assert cmd in choices, f"Missing subcommand: {cmd}"

    def test_new_requires_name(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["new"])

    def test_new_parses_name(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["new", "my-plugin"])
        assert args.name == "my-plugin"
        assert args.plugin_type == "column-generator"

    @pytest.mark.parametrize(
        "plugin_type",
        ["column-generator", "seed-reader", "processor"],
    )
    def test_new_parses_valid_plugin_types(self, plugin_type: str) -> None:
        parser = build_parser()
        args = parser.parse_args(["new", "my-plugin", "--type", plugin_type])
        assert args.name == "my-plugin"
        assert args.plugin_type == plugin_type

    def test_new_rejects_invalid_plugin_type(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["new", "my-plugin", "--type", "not-a-type"])
        assert exc_info.value.code == 2

    def test_new_help_mentions_column_generator_default(self) -> None:
        parser = build_parser()
        subparsers_actions = [a for a in parser._subparsers._actions if hasattr(a, "_parser_class")]
        new_parser = subparsers_actions[0].choices["new"]
        assert "default: column-generator" in new_parser.format_help()

    def test_bump_parses_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["bump", "data-designer-test", "minor"])
        assert args.plugin == "data-designer-test"
        assert args.part == "minor"

    def test_check_release_parses_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["check-release", "data-designer-test", "1.0.0"])
        assert args.plugin_name == "data-designer-test"
        assert args.tag_version == "1.0.0"

    def test_license_headers_check_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["license-headers", "--check"])
        assert args.check is True

    def test_license_headers_default_no_check(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["license-headers"])
        assert args.check is False

    def test_plugin_docs_check_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["plugin-docs", "--check"])
        assert args.check is True

    def test_sync_catalog_parses_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["sync", "catalog", "--check"])
        assert args.command == "sync"
        assert args.sync_target == "catalog"
        assert args.check is True

    def test_no_command_prints_help(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert not hasattr(args, "func")


class TestDispatch:
    """Tests for subcommand dispatch routing."""

    @patch("ddp.cli._run_new")
    def test_new_dispatches(self, mock_run: MagicMock) -> None:
        mock_run.return_value = 0
        parser = build_parser()
        args = parser.parse_args(["new", "test-plugin", "--type", "seed-reader"])
        args.func(args)
        mock_run.assert_called_once_with(args)
        assert args.plugin_type == "seed-reader"

    @patch("ddp.scaffold.main")
    def test_run_new_passes_plugin_type_to_scaffold(self, mock_scaffold: MagicMock) -> None:
        result = _run_new(Namespace(name="test-plugin", plugin_type="processor"))

        assert result == 0
        mock_scaffold.assert_called_once_with(["test-plugin", "--type", "processor"])

    @patch("ddp.cli._run_plugin_docs")
    def test_plugin_docs_dispatches(self, mock_run: MagicMock) -> None:
        mock_run.return_value = 0
        parser = build_parser()
        args = parser.parse_args(["plugin-docs"])
        args.func(args)
        mock_run.assert_called_once_with(args)

    @patch("ddp.cli._run_sync_catalog")
    def test_sync_catalog_dispatches(self, mock_run: MagicMock) -> None:
        mock_run.return_value = 0
        parser = build_parser()
        args = parser.parse_args(["sync", "catalog"])
        args.func(args)
        mock_run.assert_called_once_with(args)

    @patch("ddp.cli._run_validate")
    def test_validate_dispatches(self, mock_run: MagicMock) -> None:
        mock_run.return_value = 0
        parser = build_parser()
        args = parser.parse_args(["validate"])
        args.func(args)
        mock_run.assert_called_once_with(args)

    @patch("ddp.cli._run_bump")
    def test_bump_dispatches(self, mock_run: MagicMock) -> None:
        mock_run.return_value = 0
        parser = build_parser()
        args = parser.parse_args(["bump", "data-designer-test", "patch"])
        args.func(args)
        mock_run.assert_called_once_with(args)
