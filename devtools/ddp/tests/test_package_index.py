# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for static package-index tooling."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from ddp import package_index

VALID_HASH = "sha256=" + ("a" * 64)


def write_fake_wheel(path: Path) -> None:
    """Write a minimal wheel-like zip with core metadata.

    Args:
        path: Wheel path to write.
    """
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data_designer_example-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: data-designer-example\nVersion: 0.1.0\nRequires-Python: >=3.10\n",
        )


def write_fake_wheel_with_requires_python(path: Path, requires_python: str) -> None:
    """Write a minimal wheel-like zip with configurable Requires-Python.

    Args:
        path: Wheel path to write.
        requires_python: Requires-Python metadata value.
    """
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data_designer_example-0.1.0.dist-info/METADATA",
            (
                "Metadata-Version: 2.4\n"
                "Name: data-designer-example\n"
                "Version: 0.1.0\n"
                f"Requires-Python: {requires_python}\n"
            ),
        )


def test_read_package_rows_validates_json_lines(tmp_path: Path) -> None:
    package_list = tmp_path / "packages.json"
    package_list.write_text(
        json.dumps(
            {
                "filename": "data_designer_example-0.1.0-py3-none-any.whl",
                "hash": VALID_HASH,
                "requires_python": ">=3.10",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert package_index.read_package_rows(package_list) == [
        {
            "filename": "data_designer_example-0.1.0-py3-none-any.whl",
            "hash": VALID_HASH,
            "requires_python": ">=3.10",
        }
    ]


def test_read_package_rows_rejects_invalid_hash(tmp_path: Path) -> None:
    package_list = tmp_path / "packages.json"
    package_list.write_text(
        json.dumps(
            {
                "filename": "data_designer_example-0.1.0-py3-none-any.whl",
                "hash": "sha256=not-a-real-hash",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(package_index.PackageIndexError) as exc_info:
        package_index.read_package_rows(package_list)

    assert "invalid hash" in str(exc_info.value)


def test_package_row_for_file_extracts_hash_and_requires_python(tmp_path: Path) -> None:
    wheel = tmp_path / "data_designer_example-0.1.0-py3-none-any.whl"
    write_fake_wheel(wheel)

    row = package_index.package_row_for_file(wheel)

    assert row["filename"] == wheel.name
    assert isinstance(row["hash"], str)
    assert row["hash"].startswith("sha256=")
    assert row["requires_python"] == ">=3.10"


def test_package_row_for_file_rejects_invalid_requires_python(tmp_path: Path) -> None:
    wheel = tmp_path / "data_designer_example-0.1.0-py3-none-any.whl"
    write_fake_wheel_with_requires_python(wheel, "not a specifier")

    with pytest.raises(package_index.PackageIndexError) as exc_info:
        package_index.package_row_for_file(wheel)

    assert "invalid Requires-Python" in str(exc_info.value)


def test_merge_package_rows_refuses_hash_conflicts() -> None:
    existing = [
        {
            "filename": "data_designer_example-0.1.0-py3-none-any.whl",
            "hash": VALID_HASH,
        }
    ]
    new = [
        {
            "filename": "data_designer_example-0.1.0-py3-none-any.whl",
            "hash": "sha256=" + ("b" * 64),
        }
    ]

    with pytest.raises(package_index.PackageIndexError) as exc_info:
        package_index.merge_package_rows(existing, new)

    assert "different SHA256" in str(exc_info.value)


def test_sync_package_index_site_copies_dumb_pypi_outputs_and_catalog(monkeypatch, tmp_path: Path) -> None:
    package_list = tmp_path / "packages.json"
    package_index.write_package_rows(package_list, [])
    catalog_path = tmp_path / "catalog" / "plugins.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text('{"schema_version":2,"packages":[]}\n', encoding="utf-8")

    def fake_run_dumb_pypi(package_list_path: Path, packages_url: str, output_dir: Path) -> None:
        assert package_list_path.is_file()
        assert packages_url == "https://packages.example.test/"
        (output_dir / "simple").mkdir(parents=True)
        (output_dir / "simple" / "index.html").write_text("<html></html>", encoding="utf-8")
        (output_dir / "pypi").mkdir(parents=True)
        (output_dir / "pypi" / "index.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(package_index, "run_dumb_pypi", fake_run_dumb_pypi)
    monkeypatch.setattr(package_index.catalog, "PLUGINS_CATALOG_PATH", catalog_path)

    site_dir = tmp_path / "site"
    package_index.sync_package_index_site(
        package_list_path=package_list,
        packages_url="https://packages.example.test/",
        site_dir=site_dir,
    )

    assert (site_dir / "simple" / "index.html").is_file()
    assert (site_dir / "pypi" / "index.html").is_file()
    assert (site_dir / "packages.json").is_file()
    assert (site_dir / "catalog" / "plugins.json").read_text(encoding="utf-8") == '{"schema_version":2,"packages":[]}\n'


def test_sync_package_index_site_removes_stale_generated_index_files(monkeypatch, tmp_path: Path) -> None:
    package_list = tmp_path / "packages.json"
    package_index.write_package_rows(package_list, [])

    def fake_run_dumb_pypi(package_list_path: Path, packages_url: str, output_dir: Path) -> None:
        (output_dir / "simple").mkdir(parents=True)
        (output_dir / "simple" / "index.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(package_index, "run_dumb_pypi", fake_run_dumb_pypi)

    site_dir = tmp_path / "site"
    (site_dir / "simple" / "stale.html").parent.mkdir(parents=True)
    (site_dir / "simple" / "stale.html").write_text("stale", encoding="utf-8")
    (site_dir / "pypi" / "stale.html").parent.mkdir(parents=True)
    (site_dir / "pypi" / "stale.html").write_text("stale", encoding="utf-8")

    package_index.sync_package_index_site(
        package_list_path=package_list,
        packages_url="https://packages.example.test/",
        site_dir=site_dir,
    )

    assert (site_dir / "simple" / "index.html").is_file()
    assert not (site_dir / "simple" / "stale.html").exists()
    assert not (site_dir / "pypi" / "stale.html").exists()
