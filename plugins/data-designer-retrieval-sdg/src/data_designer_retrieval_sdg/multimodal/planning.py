# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, deterministic context and summary selection helpers."""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from itertools import zip_longest

from data_designer_retrieval_sdg.multimodal.models import (
    GenerationContext,
    MultimodalSDGConfig,
    QueryInstruction,
)
from data_designer_retrieval_sdg.multimodal.storage import fingerprint
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource


def context_chars(ids: list[str], sources: dict[str, RetrievalSource]) -> int:
    """Measure serialized source text, excluding instruction and response overhead.

    Args:
        ids: Ordered source identities.
        sources: Complete source lookup.

    Returns:
        Unicode character count, not a model token estimate.
    """
    return len(json.dumps([{"unit_id": key, "text": sources[key].text} for key in ids], ensure_ascii=False))


def bounded_contexts(
    contexts: list[GenerationContext], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[GenerationContext]:
    """Partition oversized memberships without dropping text, images or source IDs.

    Args:
        contexts: Explicit or automatically constructed memberships.
        sources: Full source collection.
        config: Maximum units and serialized text characters per request.

    Returns:
        Stable bounded contexts. Multi-document memberships are interleaved.

    Raises:
        ValueError: A single indivisible retrieval unit exceeds the character bound.
    """
    by_id = {source.unit_id: source for source in sources}
    result = []
    for context in contexts:
        documents = defaultdict(list)
        for key in context.unit_ids:
            if context_chars([key], by_id) > config.max_context_chars:
                raise ValueError(f"Source unit {key!r} exceeds max_context_chars; prepare smaller canonical units")
            documents[by_id[key].document_id].append(key)
        ordered = [key for group in zip_longest(*documents.values()) for key in group if key is not None]
        chunks, current = [], []
        for key in ordered:
            if current and (
                len(current) == config.max_units_per_context
                or context_chars(current + [key], by_id) > config.max_context_chars
            ):
                chunks.append(current)
                current = []
            current.append(key)
        chunks.append(current)
        for ids in chunks:
            identity = context.context_id if len(chunks) == 1 else "ctx_" + fingerprint([context.context_id, ids])[:32]
            result.append(GenerationContext(context_id=identity, unit_ids=ids, language=context.language))
    return result


def automatic_contexts(sources: list[RetrievalSource], strategy: str) -> list[GenerationContext]:
    """Build unit contexts or contiguous document sections in input order.

    Args:
        sources: Canonical units, already in the desired reading order.
        strategy: ``unit`` or ``document``; no document text delimiter parsing.

    Returns:
        Unbounded memberships, to be partitioned by ``bounded_contexts``.
    """
    if strategy == "unit":
        return [GenerationContext(context_id=s.unit_id, unit_ids=[s.unit_id], language=s.language) for s in sources]
    grouped = defaultdict(list)
    for source in sources:
        grouped[(source.document_id, source.language)].append(source.unit_id)
    return [
        GenerationContext(context_id="doc_" + fingerprint(key)[:32], unit_ids=ids, language=key[1])
        for key, ids in grouped.items()
    ]


def summary_text(row: dict) -> str:
    """Return normalized textual and visual summary evidence for comparison."""
    return " ".join((row["summary"]["summary"] + " " + row["summary"]["visual_evidence"]).casefold().split())


def similarity(left: set, right: set) -> float:
    """Compute Jaccard similarity; two empty sets provide no evidence of similarity."""
    return len(left & right) / len(left | right) if left or right else 0.0


def related_contexts(
    rows: list[dict], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[GenerationContext]:
    """Propose bounded pairs/triples using lexical summary overlap across documents.

    Args:
        rows: Source-grounded summaries before selection.
        sources: Canonical units for document membership.
        config: Maximum neighbors, lexical threshold and request bounds.

    Returns:
        Additional contexts only; original contexts are never replaced.
    """
    rows = [row for row in rows if summary_passes(row, config)]
    by_id = {s.unit_id: s for s in sources}
    tokens = [set(re.findall(r"\w+", summary_text(row))) for row in rows]
    seen, result = set(), []
    for i, row in enumerate(rows):
        context = row["context"]
        docs = {by_id[key].document_id for key in context["unit_ids"]}
        scores = []
        for j, other in enumerate(rows):
            other_context = other["context"]
            other_docs = {by_id[key].document_id for key in other_context["unit_ids"]}
            score = similarity(tokens[i], tokens[j])
            if (
                docs.isdisjoint(other_docs)
                and context["language"] == other_context["language"]
                and score >= config.related_summary_similarity
            ):
                scores.append((-score, other_context["context_id"], j))
        members = list(context["unit_ids"])
        for _, _, j in sorted(scores)[: config.related_contexts_per_context]:
            members = list(dict.fromkeys(members + rows[j]["context"]["unit_ids"]))
            key = (context["language"], tuple(sorted(members)))
            if key in seen:
                continue
            seen.add(key)
            proposed = GenerationContext(
                context_id="related_" + fingerprint(key)[:32], unit_ids=members, language=context["language"]
            )
            for bounded in bounded_contexts([proposed], sources, config):
                if len({by_id[k].document_id for k in bounded.unit_ids}) > 1:
                    result.append(bounded)
    return result


def summary_passes(row: dict, config: MultimodalSDGConfig) -> bool:
    """Apply the configured summary gate without inventing missing judgments."""
    judgment = row["summary_judgment"]
    return judgment is None or min(judgment["fidelity"], judgment["usefulness"]) >= config.summary_quality_threshold


def summary_rank(row: dict) -> tuple:
    """Prefer higher-quality representatives, then stable context identity."""
    judgment = row["summary_judgment"]
    scores = [judgment["fidelity"], judgment["usefulness"]] if judgment else [0, 0]
    return (-sum(scores), -min(scores), row["context"]["context_id"])


def duplicate_summary(left: dict, right: dict, config: MultimodalSDGConfig) -> bool:
    """Deduplicate exact memberships or conservative near-identical evidence.

    Near matching protects numeric/negation tokens and requires source overlap;
    similar summaries of disjoint sources remain independent candidates.
    """
    a, b = set(left["context"]["unit_ids"]), set(right["context"]["unit_ids"])
    if a == b and left["context"]["language"] == right["context"]["language"]:
        return True
    threshold = config.summary_near_duplicate_threshold
    if threshold is None or similarity(a, b) < 0.9 or left["context"]["language"] != right["context"]["language"]:
        return False
    text_a, text_b = summary_text(left), summary_text(right)
    protected = r"\d+(?:[.,]\d+)*|\b(?:no|not|never|without)\b"
    if re.findall(protected, text_a) != re.findall(protected, text_b) or min(len(text_a), len(text_b)) < 40:
        return False
    return (
        similarity(
            {text_a[i : i + 5] for i in range(len(text_a) - 4)}, {text_b[i : i + 5] for i in range(len(text_b) - 4)}
        )
        >= threshold
    )


def select_summaries(rows: list[dict], config: MultimodalSDGConfig) -> list[dict]:
    """Record quality, deduplication and budget decisions without deleting evidence.

    Args:
        rows: All summaries and their optional actual judgments; updated in place.
        config: Quality floor and mutually exclusive count/fraction budget.

    Returns:
        Selected rows in stable quality order. Fractions apply after deduplication.
    """
    kept = []
    for row in sorted(rows, key=summary_rank):
        row["selection_reason"] = "selected"
        if not summary_passes(row, config):
            row["selection_reason"] = "summary_quality"
            continue
        duplicate = next((other for other in kept if duplicate_summary(row, other, config)), None)
        if duplicate is not None:
            row["selection_reason"] = "duplicate_summary"
            row["representative_context_id"] = duplicate["context"]["context_id"]
            continue
        kept.append(row)
    limit = config.summary_count or len(kept)
    if config.summary_fraction is not None:
        limit = math.ceil(len(kept) * config.summary_fraction)
    for row in kept[limit:]:
        row["selection_reason"] = "summary_budget"
    return kept[:limit]


def context_instructions(config: MultimodalSDGConfig, context_id: str) -> list[QueryInstruction]:
    """Select distinct query profiles with a context-local deterministic seed.

    Args:
        config: Explicit profile pool and optional requested count.
        context_id: Stable ID, independent of context enumeration order.

    Returns:
        All configured profiles or a reproducible sample without replacement.
    """
    if config.instructions_per_context is None:
        return config.instructions
    rng = random.Random(fingerprint([config.seed, context_id]))
    return rng.sample(config.instructions, config.instructions_per_context)
