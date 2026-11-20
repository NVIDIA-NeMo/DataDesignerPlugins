# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic planning and candidate boundaries for the Data Designer workflow."""

from __future__ import annotations

import math
from collections import defaultdict
from functools import partial

from data_designer_retrieval_sdg.multimodal import prompts
from data_designer_retrieval_sdg.multimodal.bounds import RequestTooLarge, check_request, fit_contexts, split_context
from data_designer_retrieval_sdg.multimodal.combinations import semantic_combinations
from data_designer_retrieval_sdg.multimodal.execution import ModelColumn, Stage
from data_designer_retrieval_sdg.multimodal.inference import Request, source_request
from data_designer_retrieval_sdg.multimodal.models import (
    Candidate,
    ContextSummary,
    Description,
    GenerationContext,
    Localization,
    MultimodalSDGConfig,
    QueryBatch,
    QueryJudgment,
    QueryMetadata,
    QuerySlot,
    RelevanceJudgment,
    SelfSufficiencyJudgment,
    SummaryJudgment,
    TOCConfirmation,
    VisualDescription,
)
from data_designer_retrieval_sdg.multimodal.planning import (
    automatic_contexts,
    context_chars,
    context_instructions,
    related_contexts,
    select_summaries,
)
from data_designer_retrieval_sdg.multimodal.sections import build_sections, toc_candidate, verify_section_coverage
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource
from data_designer_retrieval_sdg.multimodal.storage import fingerprint, write_json
from data_designer_retrieval_sdg.multimodal.summary_stages import (
    bounded_enriched_contexts,
    combination_requests,
    summary_requests,
    summary_row,
)
from data_designer_retrieval_sdg.multimodal.workflow import (
    child_rows,
    evidence_summary_requests,
    generation_context_rows,
    generation_requests,
    judgment_requests,
    localization_reasons,
    normalize_localization,
    quality_reasons,
    quote_verified,
)


def requests_for(requests):
    """Name the single structured response column consistently across planning stages."""
    return [{"response": request} for request in requests]


def contexts_from(state, key="active_contexts"):
    """Deserialize context memberships from stage state."""
    return [GenerationContext.model_validate(value) for value in state[key]]


def prepare_tocs(state: dict, sources: list[RetrievalSource], config: MultimodalSDGConfig) -> list[dict[str, Request]]:
    """Select complete, bounded TOC candidates without sending ordinary text for classification."""
    by_id = {source.unit_id: source for source in sources}
    candidates = [source for source in sources if toc_candidate(source.text)]
    selected = [
        source
        for source in candidates
        if config.max_context_chars is None or context_chars([source.unit_id], by_id) <= config.max_context_chars
    ]
    state["toc_ids"] = [source.unit_id for source in selected]
    state["toc_confirmations"] = {
        source.unit_id: {
            "has_table_of_contents": False,
            "explanation": "Candidate exceeds the text bound; no truncated confirmation request was made.",
        }
        for source in candidates
        if source.unit_id not in state["toc_ids"]
    }
    return requests_for(
        [
            Request(prompts.render("check_toc", page=source.text, language=source.language), TOCConfirmation, "judge")
            for source in selected
        ]
    )


def consume_tocs(
    state: dict, responses: list[dict | None], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> None:
    """Build whole-unit sections and their boundary audit from TOC confirmations."""
    state["toc_confirmations"].update(
        {
            key: value["response"].model_dump()
            for key, value in zip(state["toc_ids"], responses, strict=True)
            if value is not None
        }
    )
    contexts, audit = build_sections(sources, config, state["toc_confirmations"])
    state["contexts"] = [context.model_dump() for context in contexts]
    write_json(config.output_dir / "planning/section_boundaries.json", audit)


def prepare_visuals(
    state: dict, sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[dict[str, Request]]:
    """Request descriptions only for sources with actual images."""
    selected = [source for source in sources if source.images]
    state["visual_ids"] = [source.unit_id for source in selected]
    return requests_for(
        [
            source_request(
                prompts.VISUAL_DESCRIPTION,
                VisualDescription,
                "generator",
                GenerationContext(context_id=source.unit_id, unit_ids=[source.unit_id], language=source.language),
                {source.unit_id: source},
            )
            for source in selected
        ]
    )


def consume_visuals(
    state: dict, responses: list[dict | None], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> None:
    """Retain generated enrichment separately from canonical source text."""
    state["visual"] = {
        key: value["response"].model_dump()
        for key, value in zip(state["visual_ids"], responses, strict=True)
        if value is not None
    }
    write_json(config.output_dir / "planning/visual_descriptions.json", state["visual"])


def prepare_descriptions(
    state: dict, sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[dict[str, Request]]:
    """Prepare bounded document descriptions after visual enrichment."""
    by_id = {source.unit_id: source for source in sources}
    contexts = bounded_enriched_contexts(automatic_contexts(sources, "document"), sources, state["visual"], config)
    builder = partial(summary_requests, sources=by_id, visual=state["visual"])
    contexts = fit_contexts(contexts, builder, config)
    state["active_contexts"] = [context.model_dump() for context in contexts]
    return requests_for([builder(context)[0] for context in contexts])


def consume_descriptions(
    state: dict, responses: list[dict | None], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> None:
    """Join bounded descriptions in document order without dropping source memberships."""
    by_id = {source.unit_id: source for source in sources}
    descriptions = defaultdict(list)
    for context, response in zip(contexts_from(state), responses, strict=True):
        if response is None:
            continue
        descriptions[(by_id[context.unit_ids[0]].document_id, context.language)].append(
            response["response"].description
        )
    state["descriptions"] = [
        {"document_id": key[0], "language": key[1], "description": "\n".join(values)}
        for key, values in descriptions.items()
    ]
    write_json(config.output_dir / "planning/document_descriptions.json", state["descriptions"])


def prepare_summaries(
    state: dict, sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[dict[str, Request]]:
    """Build summaries from source evidence or section-planning enrichment."""
    by_id = {source.unit_id: source for source in sources}
    contexts = contexts_from(state, "contexts")
    if config.context_strategy == "sections":
        contexts = bounded_enriched_contexts(contexts, sources, state["visual"], config)
        descriptions = {(row["document_id"], row["language"]): row["description"] for row in state["descriptions"]}
        builder = partial(summary_requests, sources=by_id, visual=state["visual"], descriptions=descriptions)
    else:
        builder = partial(evidence_summary_requests, sources=by_id)
    contexts = fit_contexts(contexts, builder, config)
    if config.context_strategy == "sections" and config.contexts_file is None:
        verify_section_coverage(contexts, sources, 2 * config.section_size)
    state["active_contexts"] = [context.model_dump() for context in contexts]
    return requests_for([builder(context)[0] for context in contexts])


def consume_summaries(
    state: dict, responses: list[dict | None], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> None:
    """Record context summaries for selection and export."""
    by_id = {source.unit_id: source for source in sources}
    for context, response in zip(contexts_from(state), responses, strict=True):
        if response is None:
            continue
        summary = response["response"]
        row = (
            summary_row(context, summary, by_id, state["visual"])
            if config.context_strategy == "sections"
            else {
                "context": context.model_dump(),
                "summary": summary.model_dump(),
                "summary_judgment": None,
                "slots": {"queries": []},
            }
        )
        state["outcomes"].append(row)


def prepare_combinations(
    state: dict, sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[dict[str, Request]]:
    """Select optional summary combinations using the embedding planner."""
    rows = state["outcomes"]
    groups = semantic_combinations(rows, config)
    contexts = [
        GenerationContext(
            context_id="combination_" + fingerprint(group)[:32],
            unit_ids=[str(i) for i in group],
            language=rows[group[0]]["context"]["language"],
        )
        for group in groups
    ]
    builder = partial(combination_requests, rows=rows)
    contexts = fit_contexts(contexts, builder, config)
    state["combinations"] = [[int(i) for i in context.unit_ids] for context in contexts]
    write_json(
        config.output_dir / "planning/combinations.json",
        {
            "section_summaries": len(rows),
            "iterations": config.combination_iterations,
            "small_input_skip": bool(config.combination_iterations and len(rows) < 12),
            "members": [[rows[i]["context"]["context_id"] for i in group] for group in state["combinations"]],
        },
    )
    return requests_for([builder(context)[0] for context in contexts])


def consume_combinations(
    state: dict, responses: list[dict | None], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> None:
    """Associate combined summaries with their source unit IDs."""
    rows = state["outcomes"]
    for group, response in zip(state["combinations"], responses, strict=True):
        if response is None:
            continue
        context = GenerationContext(
            context_id="combined_" + fingerprint([rows[i]["context"]["context_id"] for i in group])[:32],
            unit_ids=list(dict.fromkeys(key for i in group for key in rows[i]["context"]["unit_ids"])),
            language=rows[group[0]]["context"]["language"],
        )
        rows.append(
            summary_row(
                context,
                response["response"],
                {source.unit_id: source for source in sources},
                state["visual"],
                combined=True,
            )
        )


def prepare_summary_judgments(
    state: dict, sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[dict[str, Request]]:
    """Judge each relevant summary once, including optional related contexts."""
    start = state.get("related_start", 0)
    state["summary_judge_indexes"] = list(range(start, len(state["outcomes"])))
    return requests_for(
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
                judge_stage="summary",
                context=GenerationContext.model_validate(row["context"]),
            )
            for row in state["outcomes"][start:]
        ]
    )


def consume_summary_judgments(
    state: dict, responses: list[dict | None], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> None:
    """Retain summary grades, including rejected summaries, for selection and reporting."""
    for index, response in zip(state["summary_judge_indexes"], responses, strict=True):
        if response is None:
            state["outcomes"][index]["summary_judgment_failed"] = True
        else:
            state["outcomes"][index]["summary_judgment"] = response["response"].model_dump()


def prepare_related(
    state: dict, sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[dict[str, Request]]:
    """Prepare optional lexical context links without changing source coverage."""
    if config.context_strategy == "sections":
        write_json(config.output_dir / "planning/summaries.json", state["outcomes"])
    additional = related_contexts(state["outcomes"], sources, config)
    builder = partial(evidence_summary_requests, sources={source.unit_id: source for source in sources})
    contexts = fit_contexts(additional, builder, config)
    state["related_start"] = len(state["outcomes"])
    state["active_contexts"] = [context.model_dump() for context in contexts]
    return requests_for([builder(context)[0] for context in contexts])


def consume_related(
    state: dict, responses: list[dict | None], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> None:
    """Record source-grounded summaries for related contexts."""
    for context, response in zip(contexts_from(state), responses, strict=True):
        if response is None:
            continue
        state["outcomes"].append(
            {
                "context": context.model_dump(),
                "summary": response["response"].model_dump(),
                "summary_judgment": None,
                "slots": {"queries": []},
            }
        )


def prepare_generation(
    state: dict, sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[dict[str, Request]]:
    """Prepare one bounded generation round; completed contexts are never regenerated."""
    by_id = {source.unit_id: source for source in sources}
    outcomes = state["outcomes"]
    if "pending" not in state:
        if config.context_strategy == "sections" and not config.related_contexts_per_context:
            write_json(config.output_dir / "planning/summaries.json", outcomes)
        selected = generation_context_rows(select_summaries(outcomes, config), outcomes, sources, config)
        state["pending"] = [outcomes.index(row) for row in selected]
        state["ready"] = []
    builder = partial(generation_requests, sources=by_id, config=config)
    rows = []
    for index in state["pending"]:
        row = outcomes[index]
        context = GenerationContext.model_validate(row["context"])
        rows.extend(child_rows(row, fit_contexts([context], builder, config), outcomes))
    state["active_indexes"] = [outcomes.index(row) for row in rows]
    for row in rows:
        row["instructions"] = [item.model_dump() for item in context_instructions(config, row["context"]["context_id"])]
    return requests_for([builder(GenerationContext.model_validate(row["context"]))[0] for row in rows])


def consume_generation(
    state: dict, responses: list[dict | None], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> None:
    """Replan at whole-unit boundaries if a generated query makes judging too large."""
    by_id = {source.unit_id: source for source in sources}
    state["pending"] = []
    for index, response in zip(state["active_indexes"], responses, strict=True):
        row = state["outcomes"][index]
        if response is None:
            row["selection_reason"] = "generation_failed"
            continue
        context = GenerationContext.model_validate(row["context"])
        batch = response["response"]
        row["slots"] = batch.model_dump()
        try:
            for slot in batch.queries:
                if slot.query is not None:
                    for request in judgment_requests(context, slot, by_id):
                        check_request(request, config)
        except RequestTooLarge:
            row["selection_reason"] = "request_size_replanned"
            children = child_rows(row, split_context(context), state["outcomes"])
            state["pending"].extend(state["outcomes"].index(child) for child in children)
        else:
            state["ready"].append(index)


def prepare_judgments(
    state: dict, sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> list[dict[str, Request]]:
    """Expand non-abstaining slots into Data Designer judge rows with original evidence."""
    if state["pending"]:
        raise RuntimeError("Generation exhausted its finite whole-unit replanning rounds")
    by_id = {source.unit_id: source for source in sources}
    state["candidate_slots"] = []
    requests = []
    for index in state["ready"]:
        row = state["outcomes"][index]
        context = GenerationContext.model_validate(row["context"])
        for slot in sorted(QueryBatch.model_validate(row["slots"]).queries, key=lambda item: item.slot):
            if slot.query is None:
                continue
            state["candidate_slots"].append({"outcome_index": index, "slot": slot.model_dump()})
            requests.append(
                dict(
                    zip(
                        ("metadata", "sufficiency", "relevance", "localization"),
                        judgment_requests(context, slot, by_id),
                        strict=True,
                    )
                )
            )
    return requests


def consume_judgments(
    state: dict, responses: list[dict | None], sources: list[RetrievalSource], config: MultimodalSDGConfig
) -> None:
    """Apply quality and evidence gates and record each candidate decision."""
    by_id = {source.unit_id: source for source in sources}
    candidates = []
    for record, response in zip(state["candidate_slots"], responses, strict=True):
        if response is None:
            continue
        row = state["outcomes"][record["outcome_index"]]
        context = GenerationContext.model_validate(row["context"])
        slot = QuerySlot.model_validate(record["slot"])
        metadata, sufficiency, relevance = (response[key] for key in ("metadata", "sufficiency", "relevance"))
        query = QueryJudgment(
            self_sufficiency=sufficiency.self_sufficiency,
            has_answer=metadata.has_answer,
            observed_type=metadata.actual_query_type,
            observed_format=metadata.actual_query_format,
            reasoning=sufficiency.reasoning + "\n" + metadata.classification_reasoning,
        )
        reasons = quality_reasons(query, relevance, config)
        localization = response["localization"]
        if localization is not None:
            localization = normalize_localization(localization, context, by_id, config.require_verbatim_quotes)
            reasons.extend(localization_reasons(localization, context, by_id, config.require_verbatim_quotes))
        elif not reasons:
            raise RuntimeError("Passing candidate is missing mandatory localization")
        candidates.append(
            Candidate(
                query_id="q_" + fingerprint([context.context_id, slot.slot, slot.query])[:32],
                context_id=context.context_id,
                query=slot.query,
                source_unit_ids=context.unit_ids,
                language=context.language,
                requested_instruction=context_instructions(config, context.context_id)[slot.slot],
                evidence_modality=slot.evidence_modality,
                quote_verification={
                    support.unit_id: quote_verified(support.quote, by_id[support.unit_id].text)
                    for support in localization.supports
                    if support.unit_id in by_id and support.modality != "image"
                }
                if localization
                else {},
                query_judgment=query,
                relevance_judgment=relevance,
                localization=localization,
                accepted=not reasons,
                rejection_reasons=reasons,
            ).model_dump(mode="json")
        )
    state["candidates"] = candidates


def workflow_stages(config: MultimodalSDGConfig) -> list[Stage]:
    """Declare the complete Data Designer graph, including every bounded generation round."""
    stages = []
    if config.context_strategy == "sections":
        if config.contexts_file is None:
            stages.append(
                Stage(
                    "toc-confirmation", (ModelColumn("response", TOCConfirmation, "judge"),), prepare_tocs, consume_tocs
                )
            )
        stages.extend(
            [
                Stage(
                    "visual-enrichment",
                    (ModelColumn("response", VisualDescription, "generator", images=True),),
                    prepare_visuals,
                    consume_visuals,
                ),
                Stage(
                    "document-description",
                    (ModelColumn("response", Description, "generator"),),
                    prepare_descriptions,
                    consume_descriptions,
                ),
            ]
        )
    stages.append(
        Stage(
            "summaries",
            (ModelColumn("response", ContextSummary, "generator", images=config.context_strategy != "sections"),),
            prepare_summaries,
            consume_summaries,
        )
    )
    if config.context_strategy == "sections":
        stages.append(
            Stage(
                "summary-combinations",
                (ModelColumn("response", ContextSummary, "generator"),),
                prepare_combinations,
                consume_combinations,
            )
        )
    if config.judge_summaries:
        stages.append(
            Stage(
                "summary-judgments",
                (ModelColumn("response", SummaryJudgment, "judge", judge_stage="summary"),),
                prepare_summary_judgments,
                consume_summary_judgments,
            )
        )
    if config.related_contexts_per_context:
        stages.append(
            Stage(
                "related-summaries",
                (ModelColumn("response", ContextSummary, "generator", images=True),),
                prepare_related,
                consume_related,
            )
        )
        if config.judge_summaries:
            stages.append(
                Stage(
                    "related-summary-judgments",
                    (ModelColumn("response", SummaryJudgment, "judge", judge_stage="summary"),),
                    prepare_summary_judgments,
                    consume_summary_judgments,
                )
            )
    slots = config.instructions_per_context or len(config.instructions)
    for index in range(1 + math.ceil(math.log2(config.max_units_per_context))):
        stages.append(
            Stage(
                f"queries-{index}",
                (ModelColumn("response", QueryBatch.for_slot_count(slots), "generator", images=True),),
                prepare_generation,
                consume_generation,
            )
        )
    skip = (
        "{{ _skip or metadata.has_answer or sufficiency.self_sufficiency < "
        + str(config.self_sufficiency_threshold)
        + " or relevance.relevance < "
        + str(config.relevance_threshold)
        + " }}"
    )
    stages.append(
        Stage(
            "query-judgments",
            (
                ModelColumn("metadata", QueryMetadata, "judge", judge_stage="query_metadata"),
                ModelColumn("sufficiency", SelfSufficiencyJudgment, "judge", judge_stage="self_sufficiency"),
                ModelColumn("relevance", RelevanceJudgment, "judge", images=True, judge_stage="relevance"),
                ModelColumn("localization", Localization, "judge", images=True, skip=skip, judge_stage="localization"),
            ),
            prepare_judgments,
            consume_judgments,
        )
    )
    return stages
