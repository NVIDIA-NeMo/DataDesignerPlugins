# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from fastapi import HTTPException


def digest(value: dict) -> str:
    """Hash canonical JSON without storing credentials in a manifest."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    """Publish a private JSON file atomically, flushing its contents and directory."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with open(temporary, "x", opener=private_file) as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def private_file(path: str, flags: int) -> int:
    return os.open(path, flags, 0o600)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass
class Turn:
    directory: Path
    number: int
    request_hash: str
    manifest: dict
    receipt: dict | None = None
    native: dict | None = None

    @property
    def workspace(self) -> Path:
        return self.directory / "workspace"


class SessionStore:
    """Single-writer, completed-turn checkpoints; interrupted tools are never replayed."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def turn(self, key: str, number: int, request: dict, fingerprint: str) -> Iterator[Turn]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", key) or number < 1:
            raise HTTPException(422, "Expected a safe conversation ID and positive turn")
        directory = self.root / key
        directory.mkdir(exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise HTTPException(409, "Conversation directory must not be a symlink")
        sync_directory(self.root)
        with open(directory / "writer.lock", "a", opener=private_file) as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise HTTPException(409, "Conversation has an active writer") from error
            manifest_path = directory / "manifest.json"
            manifest = (
                json.loads(manifest_path.read_text())
                if manifest_path.exists()
                else {
                    "schema_version": 1,
                    "conversation_id": key,
                    "turn": 0,
                    "status": "new",
                    "fingerprint": fingerprint,
                }
            )
            if manifest.get("schema_version") != 1 or manifest.get("fingerprint") != fingerprint:
                raise HTTPException(409, "Session schema or model/agent configuration changed")
            turn = Turn(directory, number, digest(request), manifest)
            receipt_path = directory / f"turn-{number}.json"
            if number <= manifest["turn"]:
                turn.receipt = json.loads(receipt_path.read_text())
                if turn.receipt["request_hash"] != turn.request_hash:
                    raise HTTPException(409, "Committed turn has different input")
                yield turn
                return
            if manifest["status"] not in ("new", "parked") or number != manifest["turn"] + 1:
                raise HTTPException(409, "No committed checkpoint for this turn; interrupted turns require inspection")
            if manifest["turn"]:
                session = Path(manifest["session_file"])
                if (
                    not session.is_file()
                    or hashlib.sha256(session.read_bytes()).hexdigest() != manifest["session_sha256"]
                ):
                    raise HTTPException(409, "Committed native Pi history changed")
            (directory / "sessions").mkdir(exist_ok=True, mode=0o700)
            turn.workspace.mkdir(exist_ok=True, mode=0o700)
            manifest.update(status="running", requested_turn=number)
            save_json(manifest_path, manifest)
            try:
                yield turn
            except BaseException:
                manifest["status"] = "interrupted"
                save_json(manifest_path, manifest)
                raise

    def commit(self, turn: Turn, response: dict, observations: dict, *, final: bool = False) -> None:
        """Publish the receipt before the manifest makes it replayable."""
        if turn.native is None:
            raise ValueError("Missing native Pi state")
        session = Path(turn.native["state"]["sessionFile"]).resolve()
        if not session.is_relative_to((turn.directory / "sessions").resolve()) or not session.is_file():
            raise ValueError("Pi session file must be inside this conversation")
        with session.open("rb") as stream:
            os.fsync(stream.fileno())
        sync_directory(session.parent)
        save_json(
            turn.directory / f"turn-{turn.number}.json",
            {
                "request_hash": turn.request_hash,
                "response": response,
                "observations": observations,
                "native": turn.native,
            },
        )
        committed = dict(turn.manifest)
        committed.update(
            status="completed" if final else "parked",
            turn=turn.number,
            session_file=str(session),
            session_id=turn.native["state"]["sessionId"],
            session_sha256=hashlib.sha256(session.read_bytes()).hexdigest(),
        )
        save_json(turn.directory / "manifest.json", committed)
        turn.manifest.update(committed)
