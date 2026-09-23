# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-preserving visual enrichment, sections and semantic summary planning."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import partial

from data_designer_retrieval_sdg.multimodal import prompts
from data_designer_retrieval_sdg.multimodal.bounds import fit_contexts
from data_designer_retrieval_sdg.multimodal.combinations import semantic_combinations
from data_designer_retrieval_sdg.multimodal.inference import Request, source_request
from data_designer_retrieval_sdg.multimodal.models import (
    ContextSummary,
    Description,
    GenerationContext,
    SummaryJudgment,
    VisualDescription,
)
from data_designer_retrieval_sdg.multimodal.planning import automatic_contexts, bounded_contexts
from data_designer_retrieval_sdg.multimodal.sections import section_contexts, verify_section_coverage
from data_designer_retrieval_sdg.multimodal.storage import fingerprint, write_json
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource


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
                dict.fromkeys(descriptions[(sources[k].document_id, sources[k].language)] for k in context.unit_ids)
            ),
        )
        schema = ContextSummary
    return [Request(text, schema, "generator")]


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
    builder = partial(summary_requests, sources=by_id, visual=visual)
    chunks = fit_contexts(chunks, builder, config)
    responses = inference.generate([builder(c)[0] for c in chunks])
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


def plan_summaries(sources, contexts, config, inference) -> list[dict]:
    """Run the reference-style planning stages over the existing generic input contracts.

    Args:
        sources: Complete canonical sources, whose text/assets are never rewritten.
        contexts: Validated caller contexts; automatic sections are used only without an override.
        config: Run settings including optional semantic combinations.
        inference: Cached structured Data Designer calls.

    Returns:
        Single and combined summary records before grading/deduplication/budgeting.
    """
    root = config.output_dir / "planning"
    by_id = {s.unit_id: s for s in sources}
    if config.contexts_file is None:
        contexts = section_contexts(sources, config, inference)
    visual = visual_enrichment(sources, inference)
    write_json(root / "visual_descriptions.json", visual)
    descriptions = describe_documents(sources, visual, config, inference)
    write_json(
        root / "document_descriptions.json",
        [{"document_id": key[0], "language": key[1], "description": value} for key, value in descriptions.items()],
    )
    # Section summaries need text, not bounded image attachments. Keep larger memberships
    # for semantic combinations and deduplication; bound image requests after selection.
    contexts = bounded_enriched_contexts(contexts, sources, visual, config)
    builder = partial(summary_requests, sources=by_id, visual=visual, descriptions=descriptions)
    contexts = fit_contexts(contexts, builder, config)
    if config.contexts_file is None:
        verify_section_coverage(contexts, sources, 2 * config.section_size)
    summaries = inference.generate([builder(c)[0] for c in contexts])
    rows = [summary_row(c, s, by_id, visual) for c, s in zip(contexts, summaries, strict=True)]
    combinations = semantic_combinations(rows, config)
    combination_builder = partial(combination_requests, rows=rows)
    combination_contexts = [
        GenerationContext(
            context_id="combination_" + fingerprint(group)[:32],
            unit_ids=[str(i) for i in group],
            language=rows[group[0]]["context"]["language"],
        )
        for group in combinations
    ]
    combination_contexts = fit_contexts(combination_contexts, combination_builder, config)
    combinations = [[int(i) for i in c.unit_ids] for c in combination_contexts]
    write_json(
        root / "combinations.json",
        {
            "section_summaries": len(rows),
            "iterations": config.combination_iterations,
            "small_input_skip": bool(config.combination_iterations and len(rows) < 12),
            "members": [[rows[i]["context"]["context_id"] for i in group] for group in combinations],
        },
    )
    combined = inference.generate([combination_builder(c)[0] for c in combination_contexts])
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
