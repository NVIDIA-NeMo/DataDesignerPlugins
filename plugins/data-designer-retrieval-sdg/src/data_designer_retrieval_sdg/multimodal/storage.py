# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable, checksummed workflow artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    """Hash a file in bounded blocks."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def fingerprint(value: Any) -> str:
    """Hash a JSON-compatible value independently of mapping insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_bytes(path: Path, content: bytes) -> None:
    """Publish immutable bytes atomically; an existing different artifact is an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sdg-", dir=path.parent) as directory:
        temporary = Path(directory) / "content"
        temporary.write_bytes(content)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Artifact changed; use a new run directory: {path}") from None


def write_json(path: Path, value: Any) -> None:
    """Persist JSON without replacing an existing different value."""
    write_bytes(
        path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Persist an ordered, possibly empty JSONL collection."""
    write_bytes(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in records
        ).encode(),
    )


def artifact(path: Path, root: Path) -> dict:
    """Describe one portable artifact's relative location and integrity."""
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
