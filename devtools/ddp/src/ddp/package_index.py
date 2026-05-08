# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and validate the static package index used by the plugin catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

from ddp import catalog
from ddp._repo import find_repo_root, load_toml
from ddp.catalog_config import load_catalog_config

PACKAGE_LIST_FILENAME = "packages.json"
PACKAGE_FILE_SUFFIXES = (".whl", ".tar.gz", ".zip")
PACKAGE_INDEX_DIRS = ("simple", "pypi")
HASH_PREFIX = "sha256="


class PackageIndexError(RuntimeError):
    """Raised when the package index cannot be generated or validated."""


@dataclass(frozen=True)
class PackageFileInfo:
    """Package identity parsed from a distribution filename.

    Args:
        name: Canonical package distribution name.
        version: Parsed package version.
        filename: Distribution filename.
    """

    name: str
    version: Version
    filename: str


def main(args: list[str] | None = None) -> int:
    """Run the package-index CLI.

    Args:
        args: CLI arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    parsed = parser.parse_args(args)
    try:
        return parsed.func(parsed) or 0
    except PackageIndexError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the package-index argument parser.

    Returns:
        Configured argument parser.
    """
    catalog_config = load_catalog_config(find_repo_root())
    parser = argparse.ArgumentParser(
        prog="ddp package-index",
        description="Build and validate the static Python package index.",
    )
    sub = parser.add_subparsers(dest="package_index_command", required=True)

    p_build = sub.add_parser("build", help="Build package index files into a site directory")
    add_package_list_args(p_build, catalog_config.package_assets_url)
    p_build.add_argument("--site-dir", default="site", type=Path, help="Site output directory to update.")
    p_build.set_defaults(func=_run_build)

    p_check = sub.add_parser("check", help="Validate package rows and generated static index files")
    add_package_list_args(p_check, catalog_config.package_assets_url)
    p_check.set_defaults(func=_run_check)

    p_merge = sub.add_parser("merge", help="Merge built distribution files into a packages.json JSON-lines file")
    p_merge.add_argument("--package-list", required=True, type=Path, help="Existing packages.json file, if present.")
    p_merge.add_argument("--dist-dir", required=True, type=Path, help="Directory containing built wheels and sdists.")
    p_merge.add_argument("--output", required=True, type=Path, help="Output packages.json path.")
    p_merge.set_defaults(func=_run_merge)

    p_qa = sub.add_parser("qa", help="Run local scratch QA against a built static package index")
    p_qa.add_argument("--plugin", default="data-designer-template", help="Plugin package name to build and install.")
    p_qa.add_argument("--scratch-dir", type=Path, help="Scratch directory. Defaults to a new temporary directory.")
    p_qa.add_argument("--force", action="store_true", help="Delete the scratch directory first if it exists.")
    p_qa.set_defaults(func=_run_qa)

    return parser


def add_package_list_args(parser: argparse.ArgumentParser, default_packages_url: str) -> None:
    """Add package-list and package URL arguments to a parser.

    Args:
        parser: Parser to modify.
        default_packages_url: Default package file base URL.
    """
    parser.add_argument(
        "--package-list",
        default=Path(".cache/package-index/packages.json"),
        type=Path,
        help="JSON-lines package list. Missing files are treated as an empty list.",
    )
    parser.add_argument(
        "--packages-url",
        default=default_packages_url,
        help="Base URL where package files named by packages.json are served.",
    )


def _run_build(args: argparse.Namespace) -> int:
    sync_package_index_site(
        package_list_path=args.package_list,
        packages_url=args.packages_url,
        site_dir=args.site_dir,
    )
    print(f"Synced package index into {args.site_dir}")
    return 0


def _run_check(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="ddp-package-index-check-") as tmp:
        site_dir = Path(tmp) / "site"
        sync_package_index_site(
            package_list_path=args.package_list,
            packages_url=args.packages_url,
            site_dir=site_dir,
        )
        required_paths = [
            site_dir / "simple",
            site_dir / "pypi",
            site_dir / PACKAGE_LIST_FILENAME,
        ]
        missing = [path for path in required_paths if not path.exists()]
        if missing:
            missing_text = ", ".join(path.as_posix() for path in missing)
            raise PackageIndexError(f"generated package index is missing expected path(s): {missing_text}")
    print("Package index is valid.")
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    rows = merge_package_rows(
        existing_rows=read_package_rows(args.package_list),
        new_rows=package_rows_from_dist_dir(args.dist_dir),
    )
    write_package_rows(args.output, rows)
    print(f"Wrote package list: {args.output}")
    return 0


def _run_qa(args: argparse.Namespace) -> int:
    scratch_dir = run_scratch_qa(
        plugin_name=args.plugin,
        scratch_dir=args.scratch_dir,
        force=args.force,
    )
    print(f"Package index QA passed in {scratch_dir}")
    return 0


def sync_package_index_site(package_list_path: Path, packages_url: str, site_dir: Path) -> None:
    """Build package index files and copy them into a documentation site.

    Args:
        package_list_path: JSON-lines package list path. Missing files are
            treated as an empty package list.
        packages_url: Base URL for package file links.
        site_dir: Site output directory to update.

    Raises:
        PackageIndexError: If package rows are invalid or ``dumb-pypi`` fails.
    """
    rows = read_package_rows(package_list_path)
    site_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ddp-package-index-") as tmp:
        scratch_dir = Path(tmp)
        normalized_package_list = scratch_dir / PACKAGE_LIST_FILENAME
        write_package_rows(normalized_package_list, rows)
        raw_index_dir = scratch_dir / "raw-index"
        run_dumb_pypi(normalized_package_list, packages_url, raw_index_dir)

        for directory_name in PACKAGE_INDEX_DIRS:
            source_dir = raw_index_dir / directory_name
            destination_dir = site_dir / directory_name
            destination_dir.parent.mkdir(parents=True, exist_ok=True)
            if destination_dir.exists():
                shutil.rmtree(destination_dir)
            if source_dir.exists():
                shutil.copytree(source_dir, destination_dir)
            else:
                destination_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(normalized_package_list, site_dir / PACKAGE_LIST_FILENAME)
        sync_catalog_to_site(site_dir)


def sync_catalog_to_site(site_dir: Path) -> None:
    """Copy the generated catalog into the site artifact.

    Args:
        site_dir: Documentation site output directory.
    """
    catalog_source = catalog.PLUGINS_CATALOG_PATH
    if not catalog_source.exists():
        return
    catalog_destination = site_dir / "catalog" / catalog.PLUGINS_CATALOG_FILENAME
    catalog_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(catalog_source, catalog_destination)


def run_dumb_pypi(package_list_path: Path, packages_url: str, output_dir: Path) -> None:
    """Run ``dumb-pypi`` to generate static package index files.

    Args:
        package_list_path: JSON-lines package list path.
        packages_url: Base URL where package files are served.
        output_dir: Output directory for generated files.

    Raises:
        PackageIndexError: If the command fails.
    """
    command = [
        "dumb-pypi",
        "--package-list-json",
        package_list_path.as_posix(),
        "--packages-url",
        packages_url,
        "--output-dir",
        output_dir.as_posix(),
        "--no-generate-timestamp",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise PackageIndexError("dumb-pypi is not installed; run `make sync`") from exc
    except subprocess.CalledProcessError as exc:
        raise PackageIndexError(f"dumb-pypi failed with exit code {exc.returncode}") from exc


def read_package_rows(package_list_path: Path) -> list[dict[str, object]]:
    """Read and validate package-list JSON lines.

    Args:
        package_list_path: JSON-lines package list. Missing files are treated as
            an empty list.

    Returns:
        Validated package row dictionaries.

    Raises:
        PackageIndexError: If a row is malformed.
    """
    if not package_list_path.exists():
        return []

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(package_list_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PackageIndexError(f"{package_list_path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise PackageIndexError(f"{package_list_path}:{line_number} must be a JSON object")
        validate_package_row(row, f"{package_list_path}:{line_number}")
        rows.append(row)
    return sort_package_rows(rows)


def write_package_rows(package_list_path: Path, rows: list[dict[str, object]]) -> None:
    """Write package rows as deterministic JSON lines.

    Args:
        package_list_path: Output package-list path.
        rows: Package rows to write.
    """
    package_list_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{json.dumps(row, sort_keys=True, separators=(',', ':'))}\n" for row in sort_package_rows(rows))
    package_list_path.write_text(content, encoding="utf-8")


def validate_package_row(row: dict[str, object], context: str) -> None:
    """Validate one ``dumb-pypi`` JSON package-list row.

    Args:
        row: Row object to validate.
        context: Error context.

    Raises:
        PackageIndexError: If the row is malformed.
    """
    filename = row.get("filename")
    if not isinstance(filename, str) or not filename:
        raise PackageIndexError(f"{context} has invalid filename; expected a non-empty string")
    parse_package_filename(filename, context)

    hash_value = row.get("hash")
    if not isinstance(hash_value, str) or not valid_sha256_hash(hash_value):
        raise PackageIndexError(f"{context} has invalid hash; expected sha256=<64 hex characters>")

    requires_python = row.get("requires_python")
    if requires_python is not None:
        if not isinstance(requires_python, str) or not requires_python:
            raise PackageIndexError(f"{context} has invalid requires_python; expected a non-empty string")
        try:
            SpecifierSet(requires_python)
        except InvalidSpecifier as exc:
            raise PackageIndexError(f"{context} has invalid requires_python {requires_python!r}: {exc}") from exc


def valid_sha256_hash(value: str) -> bool:
    """Return whether a value is a well-formed SHA256 package hash.

    Args:
        value: Hash value to check.

    Returns:
        ``True`` when the value has ``sha256=`` plus 64 hex characters.
    """
    if not value.startswith(HASH_PREFIX) or len(value) != len(HASH_PREFIX) + 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value[len(HASH_PREFIX) :])


def parse_package_filename(filename: str, context: str | None = None) -> PackageFileInfo:
    """Parse a wheel or source distribution filename.

    Args:
        filename: Distribution filename.
        context: Optional error context.

    Returns:
        Parsed package file information.

    Raises:
        PackageIndexError: If the filename is not a supported distribution file.
    """
    if "/" in filename or "\\" in filename:
        raise PackageIndexError(f"{context or filename} has invalid filename; path separators are not allowed")
    try:
        if filename.endswith(".whl"):
            name, version, _build, _tags = parse_wheel_filename(filename)
            return PackageFileInfo(name=canonicalize_name(name), version=version, filename=filename)
        if filename.endswith((".tar.gz", ".zip")):
            name, version = parse_sdist_filename(filename)
            return PackageFileInfo(name=canonicalize_name(name), version=version, filename=filename)
    except (InvalidSdistFilename, InvalidVersion, InvalidWheelFilename) as exc:
        raise PackageIndexError(f"{context or filename} has invalid package filename {filename!r}: {exc}") from exc
    raise PackageIndexError(f"{context or filename} has unsupported package filename {filename!r}")


def package_rows_from_dist_dir(dist_dir: Path) -> list[dict[str, object]]:
    """Return package-list rows for wheel and sdist files in a directory.

    Args:
        dist_dir: Directory containing built package files.

    Returns:
        Package rows with filename, hash, and Requires-Python metadata.

    Raises:
        PackageIndexError: If no package files are present.
    """
    if not dist_dir.is_dir():
        raise PackageIndexError(f"{dist_dir} is not a directory")
    package_files = [
        path
        for path in sorted(dist_dir.iterdir())
        if path.is_file() and any(path.name.endswith(suffix) for suffix in PACKAGE_FILE_SUFFIXES)
    ]
    if not package_files:
        raise PackageIndexError(f"{dist_dir} contains no wheel or source distribution files")
    return sort_package_rows([package_row_for_file(path) for path in package_files])


def package_row_for_file(package_file: Path) -> dict[str, object]:
    """Build a ``dumb-pypi`` package-list row for one distribution file.

    Args:
        package_file: Wheel or source distribution file.

    Returns:
        Package-list row.
    """
    parse_package_filename(package_file.name)
    row: dict[str, object] = {
        "filename": package_file.name,
        "hash": f"{HASH_PREFIX}{sha256_file(package_file)}",
    }
    requires_python = package_requires_python(package_file)
    if requires_python:
        row["requires_python"] = requires_python
    validate_package_row(row, package_file.as_posix())
    return row


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest for a file.

    Args:
        path: File to hash.

    Returns:
        SHA256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_requires_python(package_file: Path) -> str | None:
    """Extract ``Requires-Python`` metadata from a wheel or sdist.

    Args:
        package_file: Package distribution file.

    Returns:
        Requires-Python value, or ``None`` when absent.
    """
    if package_file.name.endswith(".whl"):
        metadata = wheel_metadata(package_file)
    else:
        metadata = sdist_metadata(package_file)
    value = metadata.get("Requires-Python")
    if value:
        try:
            SpecifierSet(value)
        except InvalidSpecifier as exc:
            raise PackageIndexError(f"{package_file} has invalid Requires-Python {value!r}: {exc}") from exc
    return value


def wheel_metadata(package_file: Path) -> Any:
    """Read core metadata from a wheel file.

    Args:
        package_file: Wheel file.

    Returns:
        Parsed email-style metadata object.

    Raises:
        PackageIndexError: If metadata cannot be found.
    """
    with zipfile.ZipFile(package_file) as archive:
        metadata_members = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if not metadata_members:
            raise PackageIndexError(f"{package_file} does not contain .dist-info/METADATA")
        return BytesParser().parsebytes(archive.read(sorted(metadata_members)[0]))


def sdist_metadata(package_file: Path) -> Any:
    """Read core metadata from a source distribution file.

    Args:
        package_file: Source distribution file.

    Returns:
        Parsed email-style metadata object.

    Raises:
        PackageIndexError: If metadata cannot be found.
    """
    if package_file.name.endswith(".zip"):
        with zipfile.ZipFile(package_file) as archive:
            metadata_members = [name for name in archive.namelist() if name.endswith("/PKG-INFO")]
            if not metadata_members:
                raise PackageIndexError(f"{package_file} does not contain PKG-INFO")
            return BytesParser().parsebytes(archive.read(sorted(metadata_members)[0]))

    with tarfile.open(package_file) as archive:
        metadata_members = [
            member for member in archive.getmembers() if member.isfile() and member.name.endswith("/PKG-INFO")
        ]
        if not metadata_members:
            raise PackageIndexError(f"{package_file} does not contain PKG-INFO")
        handle = archive.extractfile(sorted(metadata_members, key=lambda item: item.name)[0])
        if handle is None:
            raise PackageIndexError(f"{package_file} PKG-INFO could not be read")
        return BytesParser().parsebytes(handle.read())


def merge_package_rows(
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge newly built package rows into an existing package list.

    Args:
        existing_rows: Existing rows from the package-list asset.
        new_rows: Rows for newly built distribution files.

    Returns:
        Merged rows sorted deterministically.

    Raises:
        PackageIndexError: If an existing filename has different bytes or
            conflicting metadata.
    """
    merged_by_filename = {required_row_filename(row): dict(row) for row in existing_rows}
    for new_row in new_rows:
        filename = required_row_filename(new_row)
        existing_row = merged_by_filename.get(filename)
        if existing_row is None:
            merged_by_filename[filename] = dict(new_row)
            continue
        if existing_row.get("hash") != new_row.get("hash"):
            raise PackageIndexError(f"package asset {filename!r} already exists with a different SHA256")
        existing_requires_python = existing_row.get("requires_python")
        new_requires_python = new_row.get("requires_python")
        if existing_requires_python is not None and new_requires_python is not None:
            if existing_requires_python != new_requires_python:
                raise PackageIndexError(f"package asset {filename!r} already exists with different Requires-Python")
        if existing_requires_python is None and new_requires_python is not None:
            existing_row["requires_python"] = new_requires_python
    return sort_package_rows(list(merged_by_filename.values()))


def required_row_filename(row: dict[str, object]) -> str:
    """Return a validated package-list row filename.

    Args:
        row: Package-list row.

    Returns:
        Filename string.
    """
    filename = row.get("filename")
    if not isinstance(filename, str):
        raise PackageIndexError("package row is missing filename")
    return filename


def sort_package_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Sort package-list rows by package identity and filename.

    Args:
        rows: Rows to sort.

    Returns:
        Sorted rows.
    """
    return sorted(rows, key=package_row_sort_key)


def package_row_sort_key(row: dict[str, object]) -> tuple[str, Version, str]:
    """Return the deterministic sort key for one package row.

    Args:
        row: Package-list row.

    Returns:
        Sort key tuple.
    """
    info = parse_package_filename(required_row_filename(row))
    return (info.name, info.version, info.filename)


def run_scratch_qa(plugin_name: str, scratch_dir: Path | None, force: bool = False) -> Path:
    """Run an end-to-end local package-index QA workflow.

    Args:
        plugin_name: Plugin package to build and install.
        scratch_dir: Optional scratch directory. When omitted, a new temporary
            directory is created and retained for inspection.
        force: Whether to delete an existing scratch directory first.

    Returns:
        Scratch directory used for QA.

    Raises:
        PackageIndexError: If any QA step fails.
    """
    scratch_root = prepare_scratch_dir(scratch_dir, force)
    repo_root = find_repo_root()
    plugin_dir = repo_root / "plugins" / plugin_name
    if not plugin_dir.is_dir():
        raise PackageIndexError(f"plugin directory not found: {plugin_dir}")

    project = load_toml(plugin_dir / "pyproject.toml")["project"]
    package_name = project["name"]
    if not isinstance(package_name, str):
        raise PackageIndexError(f"{plugin_dir / 'pyproject.toml'} has invalid project name")

    packages_dir = scratch_root / "packages"
    site_dir = scratch_root / "site"
    venv_dir = scratch_root / "venv"
    packages_dir.mkdir(parents=True, exist_ok=True)

    run_command(["uv", "build", plugin_dir.as_posix(), "--out-dir", packages_dir.as_posix()])
    package_list_path = scratch_root / PACKAGE_LIST_FILENAME
    write_package_rows(package_list_path, package_rows_from_dist_dir(packages_dir))
    sync_package_index_site(
        package_list_path=package_list_path,
        packages_url=f"{packages_dir.resolve().as_uri()}/",
        site_dir=site_dir,
    )
    run_command(["uv", "venv", venv_dir.as_posix()])
    python_path = venv_dir / "bin" / "python"
    catalog_target = catalog_install_target_for_package(package_name)
    run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python_path.as_posix(),
            "--default-index",
            "https://pypi.org/simple/",
            "--index",
            f"{(site_dir / 'simple').resolve().as_uri()}/",
            catalog_target.target,
        ]
    )
    verify_installed_plugin(python_path, package_name)
    return scratch_root


def prepare_scratch_dir(scratch_dir: Path | None, force: bool) -> Path:
    """Prepare a scratch directory for local QA.

    Args:
        scratch_dir: Requested scratch directory, or ``None`` for a new temp
            directory.
        force: Whether an existing requested directory should be removed first.

    Returns:
        Prepared scratch path.
    """
    if scratch_dir is None:
        return Path(tempfile.mkdtemp(prefix="ddp-package-index-qa-"))
    if scratch_dir.exists():
        if not force:
            raise PackageIndexError(f"{scratch_dir} already exists; pass --force to replace it")
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
    return scratch_dir


def verify_installed_plugin(python_path: Path, package_name: str) -> None:
    """Verify the installed package exposes valid Data Designer entry points.

    Args:
        python_path: Python interpreter in the scratch virtual environment.
        package_name: Distribution package name expected to own entry points.
    """
    script = f"""
import importlib.metadata as metadata
from data_designer.engine.testing.utils import assert_valid_plugin

package_name = {package_name!r}
entry_points = [
    entry_point
    for entry_point in metadata.entry_points(group="data_designer.plugins")
    if entry_point.dist is not None and entry_point.dist.metadata.get("Name") == package_name
]
if not entry_points:
    raise SystemExit(f"no data_designer.plugins entry points found for {{package_name}}")
for entry_point in entry_points:
    assert_valid_plugin(entry_point.load())
print("validated", len(entry_points), "entry point(s)")
"""
    run_command([python_path.as_posix(), "-c", script])


def catalog_install_target_for_package(package_name: str) -> catalog.InstallTarget:
    """Return the checked-in catalog install target for a package.

    Args:
        package_name: Package name expected in the catalog.
    Returns:
        Install target derived from ``catalog/plugins.json``.
    """
    document = json.loads(catalog.PLUGINS_CATALOG_PATH.read_text(encoding="utf-8"))
    packages = document.get("packages")
    if not isinstance(packages, list):
        raise PackageIndexError("catalog/plugins.json has invalid packages field")
    matching_package = next(
        (package for package in packages if isinstance(package, dict) and package.get("name") == package_name),
        None,
    )
    if matching_package is None:
        raise PackageIndexError(f"catalog/plugins.json has no package entry for {package_name!r}")
    install = matching_package.get("install")
    target = catalog.install_target_for_install_metadata(package_name, install)
    if target.index_url is None or not target.index_url.rstrip("/").endswith("/simple"):
        raise PackageIndexError(f"catalog install index_url has unexpected shape: {target.index_url!r}")
    return target


def run_command(command: list[str]) -> None:
    """Run a subprocess and convert failures into package-index errors.

    Args:
        command: Command and arguments.

    Raises:
        PackageIndexError: If the command exits non-zero.
    """
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise PackageIndexError(f"command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise PackageIndexError(f"command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


if __name__ == "__main__":
    sys.exit(main())
