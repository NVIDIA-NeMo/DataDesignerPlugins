# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus-independent query grouping before partitioning and positive unrolling."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher

from pydantic import BaseModel, ConfigDict, Field, field_validator

from data_designer_retrieval_sdg.multimodal.source_models import SplitName, SplitRatios

_SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "evaluation")


class QueryProvenance(BaseModel):
    """Optional caller-owned grouping and derivation metadata, never inferred from documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    query_group_id: str | None = None
    parent_query_ids: list[str] = Field(default_factory=list)
    seed_query_id: str | None = None

    @field_validator("query_group_id", "seed_query_id")
    @classmethod
    def nonblank_optional_id(cls, value: str | None) -> str | None:
        """Reject blank identifiers rather than accidentally joining unrelated queries."""
        if value is not None and not value.strip():
            raise ValueError("provenance IDs must be nonempty")
        return value

    @field_validator("parent_query_ids")
    @classmethod
    def nonblank_parent_ids(cls, values: list[str]) -> list[str]:
        """Validate caller-provided derivation links."""
        if any(not value.strip() for value in values):
            raise ValueError("parent query IDs must be nonempty")
        return values


class QueryGroupInput(QueryProvenance):
    """A unique query and optional provenance; no corpus-specific fields."""

    query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


def normalized_query(text: str) -> str:
    """Normalize Unicode, case, punctuation and whitespace, retaining words and numbers."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _find(parents: dict[str, str], key: str) -> str:
    """Resolve one connected component with path compression."""
    while parents[key] != key:
        parents[key] = parents[parents[key]]
        key = parents[key]
    return key


def _join(parents: dict[str, str], left: str, right: str) -> None:
    """Join components deterministically, independent of input ordering."""
    left, right = _find(parents, left), _find(parents, right)
    parents[max(left, right)] = min(left, right)


def _near_duplicate(left: str, right: str) -> bool:
    """Apply an optional conservative lexical heuristic, not semantic equivalence."""
    a, b = set(left.split()), set(right.split())
    return (
        min(len(a), len(b)) >= 5
        and len(a & b) / len(a | b) >= 0.9
        and SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.95
    )


def resolve_query_groups(queries: list[QueryGroupInput], *, group_near_duplicates: bool = False) -> dict[str, str]:
    """Resolve explicit groups, derivations and duplicates into disjoint components.

    Args:
        queries: Unique query IDs and texts with optional caller-provided provenance.
            Parent/seed IDs may refer to a query outside the accepted set.
        group_near_duplicates: Opt into lexical near-duplicate grouping. Exact
            normalized duplicates always join. No semantic guarantee is made.

    Returns:
        Query ID to resolved group ID. Unlinked unique queries are singleton groups.

    Raises:
        ValueError: If IDs repeat, are blank, or a query has no normalized text.
    """
    by_id = {row.query_id: row for row in queries}
    if len(by_id) != len(queries) or any(not key.strip() for key in by_id):
        raise ValueError("query IDs must be unique and nonempty before unrolling")
    parents = {key: key for key in by_id}
    seen: dict[tuple[str, str], str] = {}
    texts = {key: normalized_query(row.text) for key, row in by_id.items()}
    if any(not text for text in texts.values()):
        raise ValueError("queries must contain words or numbers")
    for key, row in sorted(by_id.items()):
        tokens = [("lineage", key), ("text", texts[key])]
        if row.query_group_id is not None:
            tokens.append(("group", row.query_group_id))
        tokens.extend(("lineage", parent) for parent in row.parent_query_ids)
        if row.seed_query_id is not None:
            tokens.append(("lineage", row.seed_query_id))
        for token in tokens:
            if token in seen:
                _join(parents, key, seen[token])
            else:
                seen[token] = key
    if group_near_duplicates:
        # An inverted index avoids comparing pairs with no shared words.
        postings: dict[str, set[str]] = {}
        for key in sorted(by_id):
            words = set(texts[key].split())
            candidates = set().union(*(postings.get(word, set()) for word in words))
            for other in sorted(candidates):
                if _find(parents, key) != _find(parents, other) and _near_duplicate(texts[key], texts[other]):
                    _join(parents, key, other)
            for word in words:
                postings.setdefault(word, set()).add(key)
    return {key: _find(parents, key) for key in sorted(by_id)}


def assign_query_group_splits(
    document_ids: Iterable[str],
    *,
    ratios: SplitRatios | None = None,
    seed: int = 42,
) -> dict[str, SplitName]:
    """Assign whole query groups to deterministic splits.

    Args:
        document_ids: Resolved query-group identifiers.
        ratios: Requested split ratios.
        seed: Stable split seed.

    Returns:
        Mapping from identifier to split.
    """
    effective_ratios = ratios or SplitRatios()
    unique_ids = sorted(set(document_ids), key=lambda value: _stable_order_key(value, seed))
    if not unique_ids:
        return {}
    values = {
        "train": effective_ratios.train,
        "validation": effective_ratios.validation,
        "evaluation": effective_ratios.evaluation,
    }
    raw_counts = {name: len(unique_ids) * values[name] for name in _SPLIT_NAMES}
    counts = {name: math.floor(raw_counts[name]) for name in _SPLIT_NAMES}
    remainder = len(unique_ids) - sum(counts.values())
    ranked = sorted(
        _SPLIT_NAMES,
        key=lambda name: (raw_counts[name] - counts[name], values[name], name),
        reverse=True,
    )
    for name in ranked[:remainder]:
        counts[name] += 1
    positive = [name for name in _SPLIT_NAMES if values[name] > 0]
    if len(unique_ids) >= len(positive):
        for name in positive:
            if counts[name] > 0:
                continue
            donors = [candidate for candidate in _SPLIT_NAMES if counts[candidate] > 1]
            donor = max(donors, key=lambda candidate: counts[candidate] - raw_counts[candidate])
            counts[donor] -= 1
            counts[name] += 1
    assignments: dict[str, SplitName] = {}
    cursor = 0
    for name in _SPLIT_NAMES:
        for document_id in unique_ids[cursor : cursor + counts[name]]:
            assignments[document_id] = name
        cursor += counts[name]
    return assignments


def _stable_order_key(identifier: str, seed: int) -> str:
    """Return a stable split-order key."""
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()
