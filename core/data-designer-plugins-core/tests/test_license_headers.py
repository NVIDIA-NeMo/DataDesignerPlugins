# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dd_plugins_core.license_headers helper functions."""

from __future__ import annotations

from pathlib import Path

from dd_plugins_core.license_headers import (
    extract_license_header,
    generate_license_header,
    parse_header_start_year,
    should_process_file,
)


class TestExtractLicenseHeader:
    """Tests for extract_license_header."""

    def test_standard_two_line_header(self) -> None:
        lines = [
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA\n",
            "# SPDX-License-Identifier: Apache-2.0\n",
            "\n",
            "import os\n",
        ]
        header, consumed = extract_license_header(lines, 0)
        assert "SPDX-FileCopyrightText" in header
        assert "SPDX-License-Identifier" in header
        assert consumed == 3  # 2 header lines + trailing blank

    def test_no_header_returns_empty(self) -> None:
        lines = ["import os\n", "print('hello')\n"]
        header, consumed = extract_license_header(lines, 0)
        assert header == ""
        assert consumed == 0

    def test_header_after_shebang(self) -> None:
        lines = [
            "#!/usr/bin/env python\n",
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA\n",
            "# SPDX-License-Identifier: Apache-2.0\n",
            "\n",
        ]
        header, consumed = extract_license_header(lines, 1)
        assert "SPDX-FileCopyrightText" in header
        assert consumed == 3


class TestParseHeaderStartYear:
    """Tests for parse_header_start_year."""

    def test_single_year(self) -> None:
        header = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA\n"
        assert parse_header_start_year(header) == 2026

    def test_year_range(self) -> None:
        header = "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA\n"
        assert parse_header_start_year(header) == 2025

    def test_no_year_returns_none(self) -> None:
        header = "# Some random comment\n"
        assert parse_header_start_year(header) is None


class TestShouldProcessFile:
    """Tests for should_process_file."""

    def test_python_file(self) -> None:
        assert should_process_file(Path("/foo/bar.py")) is True

    def test_shell_file(self) -> None:
        assert should_process_file(Path("/foo/bar.sh")) is True

    def test_non_supported_extension(self) -> None:
        assert should_process_file(Path("/foo/bar.txt")) is False

    def test_skip_pycache(self) -> None:
        assert should_process_file(Path("/foo/__pycache__/bar.py")) is False

    def test_skip_version_file(self) -> None:
        assert should_process_file(Path("/foo/_version.py")) is False

    def test_skip_venv(self) -> None:
        assert should_process_file(Path("/foo/.venv/lib/bar.py")) is False


class TestGenerateLicenseHeader:
    """Tests for generate_license_header."""

    def test_single_year(self) -> None:
        header = generate_license_header("2026")
        assert "Copyright (c) 2026" in header
        assert header.endswith("\n\n")

    def test_year_range(self) -> None:
        header = generate_license_header("2025-2026")
        assert "Copyright (c) 2025-2026" in header

    def test_contains_spdx_tags(self) -> None:
        header = generate_license_header("2026")
        assert "SPDX-FileCopyrightText" in header
        assert "SPDX-License-Identifier: Apache-2.0" in header
