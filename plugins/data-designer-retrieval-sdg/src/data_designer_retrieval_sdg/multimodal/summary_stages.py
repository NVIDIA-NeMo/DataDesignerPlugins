# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-preserving visual enrichment, sections and semantic summary planning."""

from __future__ import annotations

import json

from data_designer_retrieval_sdg.multimodal import prompts
from data_designer_retrieval_sdg.multimodal.inference import Request
from data_designer_retrieval_sdg.multimodal.models import (
    ContextSummary,
    Description,
    GenerationContext,
)
from data_designer_retrieval_sdg.multimodal.planning import bounded_contexts
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource


def visual_description(visual: dict, unit_id: str) -> str:
    """Use enrichment only when the model identified substantive visual content.

    Args:
        visual: Generated enrichment records keyed by source unit ID.
        unit_id: Source unit whose description is needed.

    Returns:
        The description when substantive visual content was identified, or an
        empty string when the record is absent or marked nonvisual.
    """
    record = visual.get(unit_id, {})
    return record.get("description", "") if record.get("has_visual_content", False) else ""


def enriched_text(context: GenerationContext, sources: dict[str, RetrievalSource], visual: dict) -> str:
    """Serialize separate source and generated-enrichment fields without altering corpus text."""
    return json.dumps(
        [
            {
                "unit_id": key,
                "text": sources[key].text,
                "visual_description": visual_description(visual, key),
            }
            for key in context.unit_ids
        ],
        ensure_ascii=False,
    )


def bounded_enriched_contexts(contexts, sources, visual, config):
    """Bound actual enriched text without changing the canonical source records."""
    enriched = [
        source.model_copy(
            update={
                "text": json.dumps(
                    {
                        "text": source.text,
                        "visual_description": visual_description(visual, source.unit_id),
                    },
                    ensure_ascii=False,
                )
            }
        )
        for source in sources
    ]
    # Nested serialization conservatively includes enrichment keys/escaping overhead.
    return bounded_contexts(
        contexts,
        enriched,
        config.model_copy(update={"max_units_per_context": len(sources)}),
    )


def summary_requests(context, sources, visual, descriptions=None):
    """Build the actual document/section prompt, including generated enrichment."""
    if descriptions is None:
        text = prompts.render(
            "document_description", content=enriched_text(context, sources, visual), language=context.language
        )
        schema = Description
    else:
        text = prompts.render(
            "section_summary",
            section=enriched_text(context, sources, visual),
            language=context.language,
            document_description="\n".join(
                dict.fromkeys(
                    descriptions.get((sources[k].document_id, sources[k].language), "") for k in context.unit_ids
                )
            ),
        )
        schema = ContextSummary
    return [Request(text, schema, "generator", context=context)]


def summary_row(context, summary, sources, visual, combined=False) -> dict:
    """Create a generic audit record used by selection and the unchanged export path."""
    return {
        "context": context.model_dump(),
        "summary": summary.model_dump(),
        "summary_judgment": None,
        "slots": {"queries": []},
        "combined": combined,
        "document_ids": list(dict.fromkeys(sources[key].document_id for key in context.unit_ids)),
        "has_visual_content": any(visual.get(key, {}).get("has_visual_content", False) for key in context.unit_ids),
    }


def combination_requests(context, rows):
    """Render combined summaries before assigning their source memberships."""
    return [
        Request(
            prompts.render(
                "combination",
                summaries=[rows[int(i)]["summary"]["summary"] for i in context.unit_ids],
                language=context.language,
            ),
            ContextSummary,
            "generator",
        )
    ]
