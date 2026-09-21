# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-preserving visual enrichment, sections and semantic summary planning."""

from __future__ import annotations

import json
import re
from collections import defaultdict

from data_designer_retrieval_sdg.multimodal import prompts
from data_designer_retrieval_sdg.multimodal.combinations import semantic_combinations
from data_designer_retrieval_sdg.multimodal.inference import Request, source_request
from data_designer_retrieval_sdg.multimodal.models import (
    ContextSummary,
    Description,
    GenerationContext,
    MultimodalSDGConfig,
    SummaryJudgment,
    VisualDescription,
)
from data_designer_retrieval_sdg.multimodal.planning import automatic_contexts, bounded_contexts
from data_designer_retrieval_sdg.multimodal.storage import fingerprint, write_json
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource


def section_contexts(sources: list[RetrievalSource], config: MultimodalSDGConfig) -> list[GenerationContext]:
    """Form heading-aware sections from whole generic units in caller-provided reading order.

    Unit identity is never parsed as a page number. Without heading boundaries,
    use the reference five-unit section size. A heading starting inside the last
    unit is also visible to the following section, retaining whole-unit evidence.
    """
    by_id = {s.unit_id: s for s in sources}
    output = []
    for document in automatic_contexts(sources, "document"):
        pending = []
        for key in document.unit_ids:
            pending.append(key)
            if len(pending) >= config.section_size:
                output.append(
                    GenerationContext(
                        context_id="section_" + fingerprint([document.context_id, pending])[:32],
                        unit_ids=pending,
                        language=document.language,
                    )
                )
                text = by_id[key].text
                pending = [key] if config.section_size > 1 and re.search(r"\n#{1,5} ", text) else []
        if pending and (not output or pending != output[-1].unit_ids):
            output.append(
                GenerationContext(
                    context_id="section_" + fingerprint([document.context_id, pending])[:32],
                    unit_ids=pending,
                    language=document.language,
                )
            )
    return output


def visual_description(visual: dict, unit_id: str) -> str:
    """Use enrichment only when the model identified substantive visual content."""
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


def visual_enrichment(sources, inference) -> dict:
    """Describe source images once; ordinary text and decorations are not visual evidence."""
    by_id = {s.unit_id: s for s in sources}
    selected = [s for s in sources if s.images]
    requests = [
        source_request(
            prompts.VISUAL_DESCRIPTION,
            VisualDescription,
            "generator",
            GenerationContext(context_id=s.unit_id, unit_ids=[s.unit_id], language=s.language),
            by_id,
        )
        for s in selected
    ]
    responses = inference.generate(requests)
    return {s.unit_id: r.model_dump() for s, r in zip(selected, responses, strict=True)}


def describe_documents(sources, visual, config, inference) -> dict:
    """Describe documents in bounded text requests, retaining all supplied units."""
    by_id = {s.unit_id: s for s in sources}
    documents = automatic_contexts(sources, "document")
    chunks = bounded_enriched_contexts(documents, sources, visual, config)
    responses = inference.generate(
        [
            Request(
                prompts.render(
                    "document_description",
                    content=enriched_text(c, by_id, visual),
                    language=c.language,
                ),
                Description,
                "generator",
            )
            for c in chunks
        ]
    )
    descriptions = defaultdict(list)
    for context, response in zip(chunks, responses, strict=True):
        descriptions[(by_id[context.unit_ids[0]].document_id, context.language)].append(response.description)
    return {key: "\n".join(values) for key, values in descriptions.items()}


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


def plan_summaries(sources, contexts, config, inference) -> list[dict]:
    """Run the reference-style planning stages over the existing generic input contracts.

    Args:
        sources: Complete canonical sources, whose text/assets are never rewritten.
        contexts: Validated caller contexts; automatic sections are used only without an override.
        config: Run settings including optional local semantic combinations.
        inference: Cached structured Data Designer calls.

    Returns:
        Single and combined summary records before grading/deduplication/budgeting.
    """
    root = config.output_dir / "planning"
    by_id = {s.unit_id: s for s in sources}
    visual = visual_enrichment(sources, inference)
    write_json(root / "visual_descriptions.json", visual)
    descriptions = describe_documents(sources, visual, config, inference)
    write_json(
        root / "document_descriptions.json",
        [{"document_id": key[0], "language": key[1], "description": value} for key, value in descriptions.items()],
    )
    # Corpus descriptions are enrichment only; batching avoids silently omitting documents.
    values = list(descriptions.values())
    corpus = inference.generate(
        [
            Request(
                prompts.render(
                    "corpus_description",
                    descriptions=values[i : i + 50],
                    language="the language of the documents",
                ),
                Description,
                "generator",
            )
            for i in range(0, len(values), 50)
        ]
    )
    write_json(root / "corpus_descriptions.json", [v.model_dump() for v in corpus])
    if config.contexts_file is None:
        contexts = section_contexts(sources, config)
    # Section summaries need text, not bounded image attachments. Keep larger memberships
    # for semantic combinations and deduplication; bound image requests after selection.
    contexts = bounded_enriched_contexts(contexts, sources, visual, config)
    summaries = inference.generate(
        [
            Request(
                prompts.render(
                    "section_summary",
                    document_description="\n".join(
                        dict.fromkeys(descriptions[(by_id[k].document_id, by_id[k].language)] for k in c.unit_ids)
                    ),
                    section=enriched_text(c, by_id, visual),
                    language=c.language,
                ),
                ContextSummary,
                "generator",
            )
            for c in contexts
        ]
    )
    rows = [summary_row(c, s, by_id, visual) for c, s in zip(contexts, summaries, strict=True)]
    combinations = semantic_combinations(rows, config)
    write_json(
        root / "combinations.json",
        {
            "section_summaries": len(rows),
            "iterations": config.combination_iterations,
            "small_input_skip": bool(config.combination_iterations and len(rows) < 12),
            "members": [[rows[i]["context"]["context_id"] for i in group] for group in combinations],
        },
    )
    combined = inference.generate(
        [
            Request(
                prompts.render(
                    "combination",
                    summaries=[rows[i]["summary"]["summary"] for i in group],
                    language=rows[group[0]]["context"]["language"],
                ),
                ContextSummary,
                "generator",
            )
            for group in combinations
        ]
    )
    for group, summary in zip(combinations, combined, strict=True):
        ids = list(dict.fromkeys(key for i in group for key in rows[i]["context"]["unit_ids"]))
        context = GenerationContext(
            context_id="combined_" + fingerprint([rows[i]["context"]["context_id"] for i in group])[:32],
            unit_ids=ids,
            language=rows[group[0]]["context"]["language"],
        )
        rows.append(summary_row(context, summary, by_id, visual, combined=True))
    if config.judge_summaries:
        judgments = inference.generate(
            [
                Request(
                    prompts.render(
                        "judgment",
                        summary=row["summary"]["summary"],
                        persona=config.persona,
                        language=row["context"]["language"],
                    ),
                    SummaryJudgment,
                    "judge",
                )
                for row in rows
            ]
        )
        for row, judgment in zip(rows, judgments, strict=True):
            row["summary_judgment"] = judgment.model_dump()
    write_json(root / "summaries.json", rows)
    return rows
