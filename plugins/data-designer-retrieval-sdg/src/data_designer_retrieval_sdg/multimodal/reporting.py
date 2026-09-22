# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation coverage and quality diagnostics; never selection from held-out scores."""

from collections import Counter

from data_designer_retrieval_sdg.multimodal.models import Candidate
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource


def generation_report(sources: list[RetrievalSource], candidates: list[Candidate], outcomes: list[dict]) -> dict:
    """Count the source coverage, sequential funnel and independent diagnostic labels.

    Args:
        sources: Full canonical corpus, including distractors.
        candidates: Accepted and rejected records.
        outcomes: All selected, filtered and budgeted summaries.

    Returns:
        JSON-compatible diagnostic counts. Requested labels never gate acceptance.
    """
    by_id = {s.unit_id: s for s in sources}
    planned = {key for row in outcomes for key in row["context"]["unit_ids"]}
    generated = {
        key
        for row in outcomes
        if any(slot["query"] is not None for slot in row["slots"]["queries"])
        for key in row["context"]["unit_ids"]
    }
    accepted = [c for c in candidates if c.accepted]
    positives = {s.unit_id for c in accepted for s in c.localization.supports}
    survivors = candidates
    funnel = {"generated": len(candidates)}
    for name in ("insufficient_standalone_query", "query_contains_answer", "insufficient_relevance"):
        survivors = [c for c in survivors if name not in c.rejection_reasons]
        funnel["after_" + name] = len(survivors)
    funnel["accepted_localized"] = len(accepted)
    return {
        "summary_selection": dict(Counter(row.get("selection_reason", "selected") for row in outcomes)),
        "coverage": {
            stage: {
                "units": len(ids),
                "documents": len({by_id[key].document_id for key in ids}),
                "unit_fraction": len(ids) / len(sources) if sources else 0.0,
            }
            for stage, ids in {"planned": planned, "generated": generated, "accepted_positive": positives}.items()
        },
        "funnel": funnel,
        "retention_fraction": len(accepted) / len(candidates) if candidates else 0.0,
        "image_dependent_fraction": sum(any(s.modality != "text" for s in c.localization.supports) for c in accepted)
        / len(accepted)
        if accepted
        else 0.0,
        "multi_unit_positive_fraction": sum(len(c.localization.supports) > 1 for c in accepted) / len(accepted)
        if accepted
        else 0.0,
        "images_per_context": dict(
            Counter(
                str(sum(len(by_id[key].images) for key in row["context"]["unit_ids"]))
                for row in outcomes
                if row["slots"]["queries"]
            )
        ),
        "generated_modalities": dict(Counter(c.evidence_modality for c in candidates)),
        "requested_generated_modalities": dict(
            Counter(f"{c.requested_instruction.modality or 'unspecified'} -> {c.evidence_modality}" for c in candidates)
        ),
        "localized_modalities": dict(Counter(s.modality for c in accepted for s in c.localization.supports)),
        "observed_types": dict(Counter(c.query_judgment.observed_type for c in candidates)),
        "observed_formats": dict(Counter(c.query_judgment.observed_format for c in candidates)),
        "requested_observed": {
            field: dict(
                Counter(
                    f"{getattr(c.requested_instruction, field) or 'unspecified'} -> {getattr(c.query_judgment, observed)}"
                    for c in candidates
                )
            )
            for field, observed in (("query_type", "observed_type"), ("format", "observed_format"))
        },
        "quote_verification": dict(
            Counter(str(value).lower() for c in candidates for value in c.quote_verification.values())
        ),
    }
