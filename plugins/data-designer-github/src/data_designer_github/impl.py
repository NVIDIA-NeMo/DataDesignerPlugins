# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import pandas as pd
from data_designer.engine.resources.seed_reader import (
    FileSystemSeedReader,
    SeedReaderError,
    SeedReaderFileSystemContext,
)

from data_designer_github.config import GitHubSeedSource

logger = logging.getLogger(__name__)


LANGUAGE_BY_EXTENSION = {
    ".bash": "bash",
    ".c": "c",
    ".cc": "cpp",
    ".cfg": "config",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".lua": "lua",
    ".md": "markdown",
    ".php": "php",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".swift": "swift",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "zsh",
}

LANGUAGE_BY_FILENAME = {
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
}


@dataclass(frozen=True)
class RepositoryRoot:
    """Prepared repository root available for manifest building."""

    repo_id: str
    repo_url: str | None
    root_path: Path
    source_kind: str
    commit_sha: str | None


class GitHubSeedReader(FileSystemSeedReader[GitHubSeedSource]):
    """Read code files from GitHub clones and local git repositories."""

    output_columns: ClassVar[list[str] | None] = [
        "repo_id",
        "repo_url",
        "commit_sha",
        "source_kind",
        "repository_path",
        "source_path",
        "relative_path",
        "file_name",
        "file_extension",
        "code_lang",
        "size_bytes",
        "content_sha256",
        "content",
    ]

    def _reset_attachment_state(self) -> None:
        super()._reset_attachment_state()
        temp_dir = getattr(self, "_temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._repository_roots: list[RepositoryRoot] | None = None

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> pd.DataFrame | list[dict[str, Any]]:
        """Build a cheap file manifest across every configured repository."""
        records: list[dict[str, Any]] = []
        for repository in self._get_repository_roots(context):
            records.extend(self._build_repository_manifest(repository))
        return records

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Read file content and add it to a manifest row."""
        del context
        source_path = Path(str(manifest_row["source_path"]))
        try:
            content_bytes = source_path.read_bytes()
            content = content_bytes.decode(self.source.encoding)
        except UnicodeDecodeError as error:
            logger.warning(
                "Skipping file %s because it cannot be decoded as %s: %s",
                source_path,
                self.source.encoding,
                error,
            )
            return []
        except OSError as error:
            raise SeedReaderError(f"Failed to read repository file {source_path}: {error}") from error

        record = dict(manifest_row)
        record["content_sha256"] = hashlib.sha256(content_bytes).hexdigest()
        record["content"] = content
        return record

    def _get_filesystem_context(self) -> SeedReaderFileSystemContext:
        self._ensure_attached()
        context = getattr(self, "_filesystem_context", None)
        if context is not None:
            return context

        runtime_root = self._prepare_runtime_root()
        context = self.create_filesystem_context(runtime_root)
        self._filesystem_context = context
        return context

    def _prepare_runtime_root(self) -> Path:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="data-designer-github-")
        runtime_root = Path(self._temp_dir.name).resolve()

        repository_roots = self._prepare_local_repositories()
        clone_root = runtime_root / "github"
        clone_root.mkdir(parents=True, exist_ok=True)
        repository_roots.extend(self._clone_github_repositories(clone_root))

        if not repository_roots:
            raise SeedReaderError("GitHub seed source did not resolve any repositories.")

        self.source._runtime_path = str(runtime_root)
        self._repository_roots = repository_roots
        return runtime_root

    def _get_repository_roots(self, context: SeedReaderFileSystemContext) -> list[RepositoryRoot]:
        del context
        repository_roots = getattr(self, "_repository_roots", None)
        if repository_roots is None:
            raise SeedReaderError("Repository roots are not prepared.")
        return repository_roots

    def _prepare_local_repositories(self) -> list[RepositoryRoot]:
        local_paths = _resolve_local_repository_paths(
            parent_path=self.source.path,
            repository_paths=self.source.repository_paths,
        )
        return [self._build_local_repository_root(path) for path in local_paths]

    def _clone_github_repositories(self, clone_root: Path) -> list[RepositoryRoot]:
        repository_roots: list[RepositoryRoot] = []
        for repository_spec in self.source.repositories:
            repo_id, repo_url = normalize_github_repository(repository_spec)
            destination = clone_root / _safe_repo_directory_name(repo_id)
            self._clone_repository(repo_url=repo_url, destination=destination)
            if self.source.ref is not None:
                _run_git(
                    ["checkout", "--quiet", self.source.ref],
                    cwd=destination,
                    timeout=self.source.clone_timeout_seconds,
                )
            repository_roots.append(
                RepositoryRoot(
                    repo_id=repo_id,
                    repo_url=repo_url,
                    root_path=destination,
                    source_kind="github",
                    commit_sha=_get_commit_sha(destination),
                )
            )
        return repository_roots

    def _clone_repository(self, *, repo_url: str, destination: Path) -> None:
        command = ["clone", "--quiet"]
        if self.source.ref is not None and not _looks_like_commit_sha(self.source.ref):
            command.extend(["--branch", self.source.ref])
        if self.source.clone_depth is not None:
            command.extend(["--depth", str(self.source.clone_depth)])
        command.extend([repo_url, str(destination)])
        _run_git(command, timeout=self.source.clone_timeout_seconds)

    def _build_local_repository_root(self, root_path: Path) -> RepositoryRoot:
        remote_url = _get_remote_url(root_path)
        return RepositoryRoot(
            repo_id=_repo_id_from_local_path(root_path, remote_url),
            repo_url=remote_url,
            root_path=root_path,
            source_kind="git_repository",
            commit_sha=_get_commit_sha(root_path),
        )

    def _build_repository_manifest(self, repository: RepositoryRoot) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for file_path in self._iter_matching_files(repository.root_path):
            relative_path = file_path.relative_to(repository.root_path).as_posix()
            stat = file_path.stat()
            records.append(
                {
                    "repo_id": repository.repo_id,
                    "repo_url": repository.repo_url,
                    "commit_sha": repository.commit_sha,
                    "source_kind": repository.source_kind,
                    "repository_path": str(repository.root_path),
                    "source_path": str(file_path),
                    "relative_path": relative_path,
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix.lower(),
                    "code_lang": _detect_language(file_path),
                    "size_bytes": stat.st_size,
                    "content_sha256": "",
                    "content": "",
                }
            )
        return records

    def _iter_matching_files(self, root_path: Path) -> list[Path]:
        paths = (
            root_path.rglob(self.source.file_pattern)
            if self.source.recursive
            else root_path.glob(self.source.file_pattern)
        )
        files = [path for path in paths if self._should_include_file(root_path=root_path, file_path=path)]
        files.sort(key=lambda path: path.relative_to(root_path).as_posix())
        return files

    def _should_include_file(self, *, root_path: Path, file_path: Path) -> bool:
        if not file_path.is_file():
            return False

        relative_path = file_path.relative_to(root_path).as_posix()
        if any(fnmatchcase(relative_path, pattern) for pattern in self.source.exclude_patterns):
            return False

        try:
            file_size = file_path.stat().st_size
        except OSError as error:
            logger.warning("Skipping file %s because it cannot be stat'ed: %s", file_path, error)
            return False

        if file_size > self.source.max_file_size_bytes:
            return False

        if file_path.name in self.source.include_file_names:
            return True

        include_extensions = self.source.include_extensions
        return include_extensions is None or file_path.suffix.lower() in include_extensions


def normalize_github_repository(repository: str) -> tuple[str, str]:
    """Normalize a GitHub repository spec to ``(owner/name, clone_url)``."""
    stripped = repository.strip()
    parsed = urlparse(stripped)

    if parsed.scheme in {"http", "https"}:
        if parsed.netloc.lower() != "github.com":
            raise SeedReaderError(f"Expected a github.com repository URL, got {repository!r}.")
        repo_id = parsed.path.strip("/").removesuffix(".git")
    elif stripped.startswith("git@github.com:"):
        repo_id = stripped.removeprefix("git@github.com:").removesuffix(".git").strip("/")
    else:
        repo_id = stripped.removesuffix(".git").strip("/")

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise SeedReaderError(f"GitHub repository {repository!r} must use 'owner/name' or a github.com repository URL.")

    return repo_id, f"https://github.com/{repo_id}.git"


def _resolve_local_repository_paths(*, parent_path: str | None, repository_paths: list[str]) -> list[Path]:
    roots: dict[Path, None] = {}
    if parent_path is not None:
        parent = Path(parent_path).expanduser().resolve()
        top_level = _get_git_toplevel(parent)
        if top_level is not None:
            roots[top_level] = None
        else:
            for child in sorted(parent.iterdir()):
                if child.is_dir():
                    child_top_level = _get_git_toplevel(child)
                    if child_top_level is not None:
                        roots[child_top_level] = None

    for repository_path in repository_paths:
        path = Path(repository_path).expanduser().resolve()
        top_level = _get_git_toplevel(path)
        if top_level is None:
            raise SeedReaderError(f"Repository path {path} is not a git repository.")
        roots[top_level] = None

    return list(roots)


def _get_git_toplevel(path: Path) -> Path | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _get_commit_sha(root_path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _get_remote_url(root_path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root_path), "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _run_git(command: list[str], *, cwd: Path | None = None, timeout: int) -> None:
    git = shutil.which("git")
    if git is None:
        raise SeedReaderError("git is required to read GitHub repositories, but it was not found on PATH.")

    try:
        result = subprocess.run(
            [git, *command],
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise SeedReaderError(f"git {' '.join(command)} timed out after {timeout} seconds") from error

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SeedReaderError(f"git {' '.join(command)} failed: {detail}")


def _repo_id_from_local_path(root_path: Path, remote_url: str | None) -> str:
    if remote_url:
        try:
            repo_id, _ = normalize_github_repository(remote_url)
            return repo_id
        except SeedReaderError:
            pass
    return root_path.name


def _safe_repo_directory_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _looks_like_commit_sha(ref: str) -> bool:
    return re.fullmatch(r"[0-9a-fA-F]{7,40}", ref) is not None


def _detect_language(file_path: Path) -> str:
    if file_path.name in LANGUAGE_BY_FILENAME:
        return LANGUAGE_BY_FILENAME[file_path.name]
    return LANGUAGE_BY_EXTENSION.get(file_path.suffix.lower(), file_path.suffix.lower().removeprefix("."))
