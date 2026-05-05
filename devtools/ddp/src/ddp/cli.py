# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI for data-designer-plugins monorepo management.

Usage::

    ddp --help              # List all subcommands
    ddp new my-plugin       # Scaffold a new plugin
    ddp plugin-docs         # Generate plugin documentation pages
    ddp validate            # Validate all installed plugins
    ddp bump <plugin> patch # Bump a plugin version
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands.

    Returns:
        Configured ``ArgumentParser`` with subcommands registered.
    """
    parser = argparse.ArgumentParser(
        prog="ddp",
        description="Data Designer Plugins — monorepo management CLI.",
    )
    sub = parser.add_subparsers(dest="command")

    # ddp new <name>
    p_new = sub.add_parser(
        "new",
        help="Scaffold a new plugin",
        description=(
            "Create a new Data Designer plugin from the canonical scaffold template. "
            "Generates a complete package under plugins/data-designer-<name>/ with "
            "pyproject.toml, config, implementation, plugin wiring, tests, and CODEOWNERS."
        ),
    )
    p_new.add_argument("name", help="Plugin name in kebab-case (e.g., my-cool-thing)")
    p_new.set_defaults(func=_run_new)

    # ddp plugin-docs
    p_plugin_docs = sub.add_parser(
        "plugin-docs",
        help="Generate plugin documentation pages",
        description=(
            "Generate docs/plugins/ from each plugin's docs/ directory and "
            "package metadata, then update the generated plugin navigation in zensical.toml."
        ),
    )
    p_plugin_docs.add_argument(
        "--check",
        action="store_true",
        help="Check generated plugin docs without modifying files",
    )
    p_plugin_docs.set_defaults(func=_run_plugin_docs)

    # ddp codeowners
    p_codeowners = sub.add_parser(
        "codeowners",
        help="Aggregate CODEOWNERS to stdout",
        description=(
            "Aggregate per-plugin CODEOWNERS files into a single unified CODEOWNERS "
            "file and print it to stdout. Typically redirected to the root CODEOWNERS."
        ),
    )
    p_codeowners.set_defaults(func=_run_codeowners)

    # ddp license-headers [--check]
    p_license = sub.add_parser(
        "license-headers",
        help="Add or check SPDX license headers",
        description=(
            "Add or update SPDX license headers in all Python and shell files under "
            "devtools/ and plugins/. In --check mode, reports files that are missing or "
            "have outdated headers and exits non-zero if any need updating."
        ),
    )
    p_license.add_argument(
        "--check",
        action="store_true",
        help="Check headers without modifying files",
    )
    p_license.set_defaults(func=_run_license_headers)

    # ddp validate
    p_validate = sub.add_parser(
        "validate",
        help="Validate all installed plugins",
        description=(
            "Discover all installed data_designer.plugins entry points and run "
            "assert_valid_plugin on each one. Exits non-zero if any plugin fails "
            "validation."
        ),
    )
    p_validate.set_defaults(func=_run_validate)

    # ddp check-release <plugin_name> <tag_version>
    p_check_release = sub.add_parser(
        "check-release",
        help="Validate plugin metadata for release",
        description=(
            "Validate that a plugin's pyproject.toml is release-ready: checks that "
            "the file version matches the expected tag version and that all required "
            "PyPI metadata fields (description, license, readme, authors) are present."
        ),
    )
    p_check_release.add_argument("plugin_name", help="Plugin name (e.g. data-designer-my-plugin)")
    p_check_release.add_argument("tag_version", help="Expected version from the git tag")
    p_check_release.set_defaults(func=_run_check_release)

    # ddp bump <plugin> <part>
    p_bump = sub.add_parser(
        "bump",
        help="Bump a plugin's semantic version",
        description=(
            "Bump the semantic version of a plugin's pyproject.toml in place. "
            "Supports major, minor, and patch bumps on strict X.Y.Z versions. "
            "Pre-release versions are not supported; edit pyproject.toml manually for those."
        ),
    )
    p_bump.add_argument("plugin", help="Plugin name (e.g. data-designer-my-plugin)")
    p_bump.add_argument(
        "part",
        choices=("major", "minor", "patch"),
        help="Version component to bump",
    )
    p_bump.set_defaults(func=_run_bump)

    return parser


def _run_new(args: argparse.Namespace) -> int:
    from ddp.scaffold import main as scaffold_main

    scaffold_main([args.name])
    return 0


def _run_plugin_docs(args: argparse.Namespace) -> int:
    from ddp.plugin_docs import main as plugin_docs_main

    argv = ["--check"] if args.check else []
    return plugin_docs_main(argv)


def _run_codeowners(args: argparse.Namespace) -> int:
    from ddp.codeowners import main as codeowners_main

    codeowners_main()
    return 0


def _run_license_headers(args: argparse.Namespace) -> int:
    from ddp.license_headers import cli as license_cli

    argv = ["--check"] if args.check else []
    license_cli(argv)
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    from ddp.validate_plugins import main as validate_main

    validate_main()
    return 0


def _run_check_release(args: argparse.Namespace) -> int:
    from ddp.validate_release import main as release_main

    return release_main([args.plugin_name, args.tag_version])


def _run_bump(args: argparse.Namespace) -> int:
    from ddp.bump_version import main as bump_main

    return bump_main([args.plugin, args.part])


def main() -> None:
    """Entry point for the ``ddp`` CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
