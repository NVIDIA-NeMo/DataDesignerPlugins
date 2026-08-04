# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration and file loading for retrieval SDG runs."""

from __future__ import annotations

import copy
import hashlib
import os
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, Mapping, TypeVar

import data_designer.config as dd
import yaml
from data_designer.config.base import ConfigBase
from data_designer.engine.storage.artifact_storage import ResumeMode
from pydantic import ConfigDict, Field, model_validator

from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

DEFAULT_CHAT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
DEFAULT_PROVIDER = "nvidia"
DEFAULT_ARTIFACT_PATH = Path("./artifacts")
DEFAULT_BUFFER_SIZE = 200
DEFAULT_MAX_ARTIFACTS_PER_TYPE = 2
DEFAULT_NUM_PAIRS = 7
DEFAULT_MIN_HOPS = 1
DEFAULT_MAX_HOPS = 3
DEFAULT_MIN_COMPLEXITY = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.9
DEFAULT_QUERY_COUNTS: dict[str, int] = {"multi_hop": 3, "structural": 2, "contextual": 2}
DEFAULT_REASONING_COUNTS: dict[str, int] = {
    "factual": 1,
    "relational": 1,
    "inferential": 1,
    "temporal": 1,
    "procedural": 1,
    "causal": 1,
    "visual": 1,
}
DEFAULT_GENERATION_CONFIG = "configs/generation/default.yaml"
DEFAULT_CONVERSION_CONFIG = "configs/conversion/default.yaml"

NonNegativeInt = Annotated[int, Field(ge=0)]
_QUERY_COUNT_KEYS = frozenset(DEFAULT_QUERY_COUNTS)
_REASONING_COUNT_KEYS = frozenset(DEFAULT_REASONING_COUNTS)
RunConfigT = TypeVar("RunConfigT", bound=ConfigBase)


def _validate_count_distribution(
    name: str,
    counts: dict[str, int],
    expected_keys: frozenset[str],
    num_pairs: int,
) -> None:
    """Validate one complete question-distribution mapping."""
    actual_keys = set(counts)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(f"{name} keys must match the expected set; missing={missing}, unexpected={unexpected}")

    total = sum(counts.values())
    if total != num_pairs:
        raise ValueError(f"{name} must sum to num_pairs ({num_pairs}); got {total}")


def _is_sensitive_key(key: str) -> bool:
    """Return whether a serialized configuration key can contain a credential."""
    normalized = key.lower().replace("-", "_")
    if normalized.endswith("_env"):
        return False
    exact_sensitive_keys = {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "password",
        "proxy_authorization",
        "refresh_token",
        "secret",
        "token",
    }
    sensitive_suffixes = (
        "_access_token",
        "_api_key",
        "_password",
        "_refresh_token",
        "_secret",
    )
    return normalized in exact_sensitive_keys or normalized.endswith(sensitive_suffixes)


def _redact_sensitive_values(value: Any) -> Any:
    """Return a recursively redacted copy of serialized configuration data."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, nested_value in value.items():
            if _is_sensitive_key(str(key)) and nested_value is not None:
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = _redact_sensitive_values(nested_value)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    return value


@dataclass(frozen=True)
class ConfigSource:
    """One configuration document that contributed to a resolved run."""

    location: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable provenance record."""
        return {"location": self.location, "sha256": self.sha256}


@dataclass(frozen=True)
class LoadedRunConfig(Generic[RunConfigT]):
    """A validated run config plus non-secret resolution provenance."""

    config: RunConfigT
    sources: tuple[ConfigSource, ...]
    override_paths: tuple[str, ...]
    environment_variables: tuple[str, ...]


class RunConfigBase(ConfigBase):
    """Shared behavior for top-level generation and conversion configs."""

    model_config = ConfigDict(frozen=True)

    def to_redacted_dict(self) -> dict[str, Any]:
        """Serialize all effective settings without exposing credential values."""
        return _redact_sensitive_values(self.model_dump(mode="json"))


class GenerationPipelineConfig(ConfigBase):
    """Explicit settings for the four-column retrieval generation pipeline."""

    model_config = ConfigDict(frozen=True)

    max_artifacts_per_type: int = Field(default=DEFAULT_MAX_ARTIFACTS_PER_TYPE, ge=1)
    num_pairs: int = Field(default=DEFAULT_NUM_PAIRS, ge=1)
    query_counts: dict[str, NonNegativeInt] = Field(default_factory=lambda: dict(DEFAULT_QUERY_COUNTS))
    min_hops: int = Field(default=DEFAULT_MIN_HOPS, ge=1)
    max_hops: int = Field(default=DEFAULT_MAX_HOPS, ge=1)
    reasoning_counts: dict[str, NonNegativeInt] = Field(default_factory=lambda: dict(DEFAULT_REASONING_COUNTS))
    min_complexity: int = Field(default=DEFAULT_MIN_COMPLEXITY, ge=1, le=5)
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0)
    max_parallel_requests_for_gen: int | None = Field(default=None, ge=1)
    artifact_extraction_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1)
    artifact_extraction_provider: str = Field(default=DEFAULT_PROVIDER, min_length=1)
    qa_generation_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1)
    qa_generation_provider: str = Field(default=DEFAULT_PROVIDER, min_length=1)
    quality_judge_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1)
    quality_judge_provider: str = Field(default=DEFAULT_PROVIDER, min_length=1)
    embed_model: str = Field(default=DEFAULT_EMBED_MODEL, min_length=1)
    embed_provider: str = Field(default=DEFAULT_PROVIDER, min_length=1)

    @model_validator(mode="after")
    def validate_pipeline_settings(self) -> GenerationPipelineConfig:
        """Reject inconsistent hop and question-distribution settings."""
        if self.min_hops > self.max_hops:
            raise ValueError("min_hops must be less than or equal to max_hops")
        _validate_count_distribution("query_counts", self.query_counts, _QUERY_COUNT_KEYS, self.num_pairs)
        _validate_count_distribution("reasoning_counts", self.reasoning_counts, _REASONING_COUNT_KEYS, self.num_pairs)
        return self

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``build_qa_generation_pipeline``."""
        return self.model_dump(mode="python")


class GenerationRunConfig(RunConfigBase):
    """Complete typed input for one resumable retrieval generation run."""

    schema_version: Literal[1] = 1
    seed_source: DocumentChunkerSeedSource
    output_dir: Path
    artifact_path: Path = DEFAULT_ARTIFACT_PATH
    dataset_name: str | None = None
    buffer_size: int = Field(default=DEFAULT_BUFFER_SIZE, ge=1)
    resume: ResumeMode = Field(default=ResumeMode.NEVER, validate_default=True)
    model_providers: list[dd.ModelProvider] | None = None
    pipeline: GenerationPipelineConfig = Field(default_factory=GenerationPipelineConfig)
    num_records: int | None = Field(
        default=None,
        ge=1,
        description="Maximum seed records to process; null processes all available records",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ConversionRunConfig(RunConfigBase):
    """Complete typed input for one retrieval data conversion run."""

    schema_version: Literal[1] = 1
    input_path: Path
    corpus_id: str = Field(min_length=1)
    output_dir: Path | None = None
    eval_only: bool = False
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = 42
    quality_threshold: float = 7.0
    max_pos_docs: int = Field(default=5, ge=1)
    use_group_id_in_eval: bool = False
    split_strategy: Literal["random", "dedupped", "cluster"] = "random"
    groups_json: list[Path] | None = None

    @model_validator(mode="after")
    def validate_split_ratios(self) -> ConversionRunConfig:
        """Reject split ratios that exceed the available dataset fraction."""
        if self.train_ratio + self.val_ratio > 1.0:
            raise ValueError("train_ratio plus val_ratio must be less than or equal to 1.0")
        return self

    def to_conversion_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``run_conversion``."""
        return {
            "input_path": str(self.input_path),
            "corpus_id": self.corpus_id,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "eval_only": self.eval_only,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
            "quality_threshold": self.quality_threshold,
            "max_pos_docs": self.max_pos_docs,
            "use_group_id_in_eval": self.use_group_id_in_eval,
            "split_strategy": self.split_strategy,
            "groups_json": [str(path) for path in self.groups_json] if self.groups_json is not None else None,
        }


def _sha256(raw: bytes) -> str:
    """Return a stable hexadecimal SHA-256 digest."""
    return hashlib.sha256(raw).hexdigest()


def config_source_from_path(path: str | Path) -> ConfigSource:
    """Build provenance for an auxiliary configuration file."""
    path = Path(path)
    raw = path.read_bytes()
    return ConfigSource(location=str(path.resolve()), sha256=_sha256(raw))


def _load_mapping(raw: bytes, location: str) -> dict[str, Any]:
    """Parse one YAML/JSON document and require a mapping at its root."""
    try:
        loaded = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not parse configuration {location}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration {location} must contain a mapping at its root")
    return loaded


def _load_packaged_mapping(relative_path: str) -> tuple[dict[str, Any], ConfigSource]:
    """Load one packaged default and record its content hash."""
    resource = files("data_designer_retrieval_sdg").joinpath(relative_path)
    raw = resource.read_bytes()
    location = f"package:data_designer_retrieval_sdg/{relative_path}"
    return _load_mapping(raw, location), ConfigSource(location=location, sha256=_sha256(raw))


def _load_user_mapping(path: Path) -> tuple[dict[str, Any], ConfigSource]:
    """Load a user YAML/JSON config and record its absolute location and hash."""
    raw = path.read_bytes()
    location = str(path.resolve())
    return _load_mapping(raw, location), ConfigSource(location=location, sha256=_sha256(raw))


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _leaf_paths(value: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Return dotted paths for every explicitly supplied leaf value."""
    paths: list[str] = []
    for key, nested_value in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(nested_value, Mapping):
            paths.extend(_leaf_paths(nested_value, path))
        else:
            paths.append(path)
    return paths


def _parse_set_override(expression: str) -> tuple[list[str], Any]:
    """Parse one ``key=value`` override using YAML scalar/list semantics."""
    if "=" not in expression:
        raise ValueError(f"Invalid --set override {expression!r}; expected key=value")
    dotted_path, raw_value = expression.split("=", 1)
    parts = dotted_path.split(".")
    if not dotted_path or any(not part for part in parts):
        raise ValueError(f"Invalid --set path {dotted_path!r}")
    try:
        value = "" if raw_value == "" else yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML value in --set {expression!r}: {exc}") from exc
    return parts, value


def _set_dotted_value(document: dict[str, Any], parts: list[str], value: Any) -> None:
    """Set a dotted path, creating intermediate mappings when necessary."""
    cursor = document
    for part in parts[:-1]:
        nested = cursor.get(part)
        if nested is None:
            nested = {}
            cursor[part] = nested
        if not isinstance(nested, dict):
            joined = ".".join(parts)
            raise ValueError(f"Cannot set {joined!r}; {part!r} is not a mapping")
        cursor = nested
    cursor[parts[-1]] = value


def _resolve_provider_environment(document: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Resolve explicit provider ``${VAR}`` references without exposing credentials."""
    resolved = copy.deepcopy(document)
    environment_variables: list[str] = []
    providers = resolved.get("model_providers")
    if not isinstance(providers, list):
        return resolved, environment_variables

    pattern = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        for key in ("endpoint", "api_key"):
            value = provider.get(key)
            match = pattern.fullmatch(value) if isinstance(value, str) else None
            if match is None:
                continue
            environment_variable = match.group(1)
            if environment_variable not in os.environ:
                raise ValueError(
                    f"Configuration references unset environment variable {environment_variable!r} "
                    f"for model_providers.{key}"
                )
            provider[key] = environment_variable if key == "api_key" else os.environ[environment_variable]
            environment_variables.append(environment_variable)
    return resolved, environment_variables


def _load_run_config(
    model: type[RunConfigT],
    default_path: str,
    *,
    config_path: str | Path | None,
    cli_overrides: Mapping[str, Any] | None,
    set_overrides: list[str] | tuple[str, ...],
) -> LoadedRunConfig[RunConfigT]:
    """Resolve defaults, a user file, CLI flags, and dotted overrides."""
    resolved, default_source = _load_packaged_mapping(default_path)
    sources = [default_source]
    override_paths: list[str] = []

    if config_path is not None:
        user_config, user_source = _load_user_mapping(Path(config_path))
        resolved = _deep_merge(resolved, user_config)
        sources.append(user_source)

    if cli_overrides:
        resolved = _deep_merge(resolved, cli_overrides)
        override_paths.extend(_leaf_paths(cli_overrides))

    for expression in set_overrides:
        parts, value = _parse_set_override(expression)
        _set_dotted_value(resolved, parts, value)
        override_paths.append(".".join(parts))

    resolved, environment_variables = _resolve_provider_environment(resolved)
    return LoadedRunConfig(
        config=model.model_validate(resolved),
        sources=tuple(sources),
        override_paths=tuple(dict.fromkeys(override_paths)),
        environment_variables=tuple(dict.fromkeys(environment_variables)),
    )


def load_generation_config(
    config_path: str | Path | None = None,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    set_overrides: list[str] | tuple[str, ...] = (),
) -> LoadedRunConfig[GenerationRunConfig]:
    """Load and validate one generation config using documented precedence."""
    return _load_run_config(
        GenerationRunConfig,
        DEFAULT_GENERATION_CONFIG,
        config_path=config_path,
        cli_overrides=cli_overrides,
        set_overrides=set_overrides,
    )


def load_conversion_config(
    config_path: str | Path | None = None,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    set_overrides: list[str] | tuple[str, ...] = (),
) -> LoadedRunConfig[ConversionRunConfig]:
    """Load and validate one conversion config using documented precedence."""
    return _load_run_config(
        ConversionRunConfig,
        DEFAULT_CONVERSION_CONFIG,
        config_path=config_path,
        cli_overrides=cli_overrides,
        set_overrides=set_overrides,
    )


def dump_resolved_config(config: RunConfigBase) -> str:
    """Serialize a fully resolved config as redacted YAML."""
    return yaml.safe_dump(config.to_redacted_dict(), sort_keys=False)
