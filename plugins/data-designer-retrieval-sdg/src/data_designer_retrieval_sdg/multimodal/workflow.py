# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic text/image evidence -> queries -> judgments -> localized positives."""

from __future__ import annotations

import json
import os
from functools import partial
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from data_designer_retrieval_sdg.multimodal import prompts
from data_designer_retrieval_sdg.multimodal.bounds import (
    RequestTooLarge,
    check_request,
    fit_contexts,
    split_context,
    validate_deployment,
)
from data_designer_retrieval_sdg.multimodal.inference import DataDesignerInference, Request, source_request
from data_designer_retrieval_sdg.multimodal.models import (
    Candidate,
    ContextSummary,
    GenerationContext,
    Localization,
    MultimodalSDGConfig,
    QueryBatch,
    QueryJudgment,
    QueryMetadata,
    RelevanceJudgment,
    SelfSufficiencyJudgment,
    SummaryJudgment,
)
from data_designer_retrieval_sdg.multimodal.planning import (
    automatic_contexts,
    bounded_contexts,
    context_instructions,
    related_contexts,
    select_summaries,
)
from data_designer_retrieval_sdg.multimodal.storage import artifact, digest, fingerprint, write_bytes, write_json
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource
from data_designer_retrieval_sdg.retrieval.source_file import load_retrieval_sources


def load_contexts(config: MultimodalSDGConfig, sources: list[RetrievalSource]) -> list[GenerationContext]:
    """Build lossless bounded contexts, independent of query split groups.

    Args:
        config: Optional context JSONL and explicit request size bound.
        sources: Complete canonical source collection.

    Returns:
        Validated contexts; unselected source units remain corpus distractors.
    """
    if config.contexts_file is None:
        contexts = automatic_contexts(sources, config.context_strategy)
    else:
        contexts = [
            GenerationContext.model_validate_json(line)
            for line in config.contexts_file.read_text().splitlines()
            if line.strip()
        ]
    known = {source.unit_id for source in sources}
    if any(any(char in key for char in "\t\r\n") for key in known):
        raise ValueError("Source unit IDs must not contain TSV delimiters")
    if not contexts or len({context.context_id for context in contexts}) != len(contexts):
        raise ValueError("contexts must be nonempty with unique IDs")
    for context in contexts:
        if not set(context.unit_ids) <= known:
            raise ValueError(f"Unknown units in context: {context.context_id}")
    if config.context_strategy == "sections":
        return contexts
    return bounded_contexts(contexts, sources, config)


def source_text(context: GenerationContext, sources: dict[str, RetrievalSource]) -> str:
    """Supply exact source text to context-aware judgments, independently of generated summaries."""
    return json.dumps(
        [
            {"unit_id": key, "document_id": sources[key].document_id, "text": sources[key].text}
            for key in context.unit_ids
        ],
        ensure_ascii=False,
    )


def normalize_localization(localization: Localization, context: GenerationContext) -> Localization:
    """Keep valid source identities and the highest grade for each repeated support."""
    supports = {}
    for support in localization.supports:
        if support.unit_id in context.unit_ids and (
            support.unit_id not in supports or support.grade > supports[support.unit_id].grade
        ):
            supports[support.unit_id] = support
    return Localization(supports=list(supports.values()))


def localization_reasons(
    localization: Localization,
    context: GenerationContext,
    sources: dict[str, RetrievalSource],
    require_verbatim_quotes: bool = False,
) -> list[str]:
    """Validate source identities and inspectable evidence without repairing judge output."""
    if not localization.supports:
        return ["no_localized_positives"]
    seen = set()
    for support in localization.supports:
        if support.unit_id not in context.unit_ids or support.unit_id in seen:
            return ["invalid_localized_identity"]
        seen.add(support.unit_id)
        source = sources[support.unit_id]
        if support.modality in {"text", "text_and_image"}:
            quote = " ".join(support.quote.split())
            if not source.text.strip() or (require_verbatim_quotes and not quote):
                return ["missing_text_evidence"]
            if require_verbatim_quotes and not quote_verified(support.quote, source.text):
                return ["unverified_text_evidence"]
        if support.modality in {"image", "text_and_image"} and (not source.images or not support.visual_evidence):
            return ["missing_visual_evidence"]
    return []


def quote_verified(quote: str, text: str) -> bool:
    """Report quotation fidelity separately from the source-local relevance judgment."""
    normalized = " ".join(quote.casefold().split())
    return bool(normalized) and normalized in " ".join(text.casefold().split())


def evidence_summary_requests(c, sources):
    """Build the original-evidence summary request including image attachments."""
    return [
        source_request(
            prompts.render(
                "section_summary",
                document_description="",
                section="See supplied sources below.",
                language=c.language,
            ),
            ContextSummary,
            "generator",
            c,
            sources,
        )
    ]


def summarize_contexts(contexts, sources, config, inference) -> list[dict]:
    """Summarize original evidence and optionally judge each summary against it.

    Args:
        contexts: Bounded memberships.
        sources: Canonical unit lookup.
        config: Summary judging policy.
        inference: Cached structured inference implementation.

    Returns:
        Complete summary records, including rejected summaries and empty query slots.
    """
    builder = partial(evidence_summary_requests, sources=sources)
    contexts = fit_contexts(contexts, builder, config)
    summaries = inference.generate([builder(c)[0] for c in contexts])
    judgments = (
        inference.generate(
            [
                Request(
                    prompts.render("judgment", summary=s.summary, persona=config.persona, language=c.language),
                    SummaryJudgment,
                    "judge",
                )
                for c, s in zip(contexts, summaries, strict=True)
            ]
        )
        if config.judge_summaries
        else [None] * len(contexts)
    )
    return [
        {
            "context": c.model_dump(),
            "summary": s.model_dump(),
            "summary_judgment": j.model_dump() if j else None,
            "slots": {"queries": []},
        }
        for c, s, j in zip(contexts, summaries, judgments, strict=True)
    ]


def generation_context_rows(selected, outcomes, sources, config) -> list[dict]:
    """Bound selected summary memberships losslessly and remove duplicate generation requests."""
    seen, result = set(), []
    for row in selected:
        original = GenerationContext.model_validate(row["context"])
        for context in bounded_contexts([original], sources, config):
            key = (context.language, tuple(context.unit_ids))
            if key in seen:
                continue
            seen.add(key)
            if context.context_id == original.context_id:
                result.append(row)
            else:
                child = {
                    **row,
                    "context": context.model_dump(),
                    "slots": {"queries": []},
                    "selection_reason": "generation_context",
                    "parent_context_id": original.context_id,
                }
                outcomes.append(child)
                result.append(child)
    return result


def generation_requests(context, sources, config):
    """Construct the exact query prompt for deterministic slots on this membership."""
    instructions = context_instructions(config, context.context_id)
    return [
        source_request(
            prompts.render(
                "multi_section_query_generation"
                if len({sources[k].document_id for k in context.unit_ids}) > 1
                else "single_section_query_generation",
                language=context.language,
                query_modules=[item.model_dump() for item in instructions],
            )
            + "\nReturn one outcome per numbered slot (zero-based slot IDs). Return query=null and "
            'evidence_modality=none for unsupported slots. Use JSON null without quotes, never the string "null". '
            'Example abstention: {"slot": 1, "query": null, "evidence_modality": "none"}. '
            "Do not invent figures or tables. "
            "Use the supplied original text as well as images; text-only inputs require no images. "
            "No generated answers are requested.",
            QueryBatch.for_slot_count(len(instructions)),
            "generator",
            context,
            sources,
            language=context.language,
            instructions=[{"slot": i, **item.model_dump()} for i, item in enumerate(instructions)],
        )
    ]


def judgment_requests(context, slot, sources):
    """Build every downstream prompt using the actual generated query and evidence."""
    return [
        Request(prompts.render("query_metadata", query=slot.query), QueryMetadata, "judge"),
        Request(
            prompts.render("self_sufficiency", summary=source_text(context, sources), query=slot.query),
            SelfSufficiencyJudgment,
            "judge",
        ),
        source_request(
            prompts.render("relevancy", summary=source_text(context, sources), query=slot.query),
            RelevanceJudgment,
            "judge",
            context,
            sources,
            query=slot.query,
        ),
        source_request(prompts.LOCALIZE, Localization, "judge", context, sources, query=slot.query),
    ]


def bounded_query_batches(selected, outcomes, sources, config, inference):
    """Regenerate on smaller memberships when an actual downstream request cannot fit.

    Do not judge a multi-page query against only a fragment. Preserve superseded
    generated slots in outcomes, then generate new queries on the child contexts.
    Each split strictly reduces membership size, bounding replanning depth.
    """
    pending, accepted, batches = list(selected), [], []
    builder = partial(generation_requests, sources=sources, config=config)
    while pending:
        rows = []
        for row in pending:
            context = GenerationContext.model_validate(row["context"])
            children = fit_contexts([context], builder, config)
            rows.extend(child_rows(row, children, outcomes))
        for row in rows:
            row["instructions"] = [
                item.model_dump() for item in context_instructions(config, row["context"]["context_id"])
            ]
        responses = inference.generate([builder(GenerationContext.model_validate(row["context"]))[0] for row in rows])
        pending = []
        for row, batch in zip(rows, responses, strict=True):
            context = GenerationContext.model_validate(row["context"])
            try:
                for slot in batch.queries:
                    if slot.query is not None:
                        for request in judgment_requests(context, slot, sources):
                            check_request(request, config)
            except RequestTooLarge:
                row["slots"] = batch.model_dump()
                row["selection_reason"] = "request_size_replanned"
                pending.extend(child_rows(row, split_context(context), outcomes))
            else:
                accepted.append(row)
                batches.append(batch)
    return accepted, batches


def child_rows(row, contexts, outcomes):
    """Record every partition's lineage and retain the original row when it fits."""
    result = []
    for context in contexts:
        if context.context_id == row["context"]["context_id"]:
            result.append(row)
        else:
            child = {
                **row,
                "context": context.model_dump(),
                "slots": {"queries": []},
                "selection_reason": "generation_context",
                "parent_context_id": row["context"]["context_id"],
            }
            outcomes.append(child)
            result.append(child)
    return result


def generate_candidates(
    config: MultimodalSDGConfig,
    sources: list[RetrievalSource],
    contexts: list[GenerationContext],
    inference: DataDesignerInference,
) -> tuple[list[Candidate], list[dict]]:
    """Generate and judge bounded contexts, retaining abstentions and every rejected candidate."""
    by_id = {source.unit_id: source for source in sources}
    if config.context_strategy == "sections":
        from data_designer_retrieval_sdg.multimodal.summary_stages import plan_summaries

        outcomes = plan_summaries(sources, contexts, config, inference)
    else:
        outcomes = summarize_contexts(contexts, by_id, config, inference)
    if config.related_contexts_per_context:
        additional = related_contexts(outcomes, sources, config)
        outcomes.extend(summarize_contexts(additional, by_id, config, inference))
    selected = select_summaries(outcomes, config)
    selected = generation_context_rows(selected, outcomes, sources, config)
    selected, batches = bounded_query_batches(selected, outcomes, by_id, config, inference)
    contexts = [GenerationContext.model_validate(row["context"]) for row in selected]
    instructions = {c.context_id: context_instructions(config, c.context_id) for c in contexts}
    pending = []
    for context, row, batch in zip(contexts, selected, batches, strict=True):
        if sorted(slot.slot for slot in batch.queries) != list(range(len(instructions[context.context_id]))):
            raise ValueError("Generator did not return exactly one outcome for every requested slot")
        row["slots"] = batch.model_dump()
        row["instructions"] = [item.model_dump() for item in instructions[context.context_id]]
        for slot in sorted(batch.queries, key=lambda s: s.slot):
            if slot.query is None:
                if slot.evidence_modality != "none":
                    raise ValueError("Abstention must declare evidence_modality=none")
                continue
            if not slot.query or slot.evidence_modality == "none":
                raise ValueError("Generated query must be nonempty and declare evidence")
            pending.append((context, slot))
    judgments = [judgment_requests(c, slot, by_id) for c, slot in pending]
    metadata = inference.generate([requests[0] for requests in judgments])
    sufficiency = inference.generate([requests[1] for requests in judgments])
    query_judgments = [
        QueryJudgment(
            self_sufficiency=s.self_sufficiency,
            has_answer=m.has_answer,
            observed_type=m.actual_query_type,
            observed_format=m.actual_query_format,
            reasoning=s.reasoning + "\n" + m.classification_reasoning,
        )
        for s, m in zip(sufficiency, metadata, strict=True)
    ]
    relevance = inference.generate([requests[2] for requests in judgments])
    reasons_by_index = [quality_reasons(q, r, config) for q, r in zip(query_judgments, relevance, strict=True)]
    localized_indexes = [i for i, reasons in enumerate(reasons_by_index) if not reasons]
    localizations = inference.generate([judgments[i][3] for i in localized_indexes])
    by_index = dict(zip(localized_indexes, localizations, strict=True))
    candidates = []
    for i, ((context, slot), query_judge, relevance_judge) in enumerate(
        zip(pending, query_judgments, relevance, strict=True)
    ):
        reasons = reasons_by_index[i]
        localization = by_index.get(i)
        if localization is not None:
            localization = normalize_localization(localization, context)
            reasons.extend(localization_reasons(localization, context, by_id, config.require_verbatim_quotes))
        candidates.append(
            Candidate(
                query_id="q_" + fingerprint([context.context_id, slot.slot, slot.query])[:32],
                context_id=context.context_id,
                query=slot.query,
                source_unit_ids=context.unit_ids,
                language=context.language,
                requested_instruction=instructions[context.context_id][slot.slot],
                evidence_modality=slot.evidence_modality,
                quote_verification={
                    s.unit_id: quote_verified(s.quote, by_id[s.unit_id].text)
                    for s in localization.supports
                    if s.unit_id in by_id and s.modality != "image"
                }
                if localization
                else {},
                query_judgment=query_judge,
                relevance_judgment=relevance_judge,
                localization=localization,
                accepted=not reasons,
                rejection_reasons=reasons,
            )
        )
    return candidates, outcomes


def quality_reasons(query: QueryJudgment, relevance: RelevanceJudgment, config: MultimodalSDGConfig) -> list[str]:
    """Return the retrieval-first hard-gate failures, independent of requested style."""
    reasons = []
    if query.self_sufficiency < config.self_sufficiency_threshold:
        reasons.append("insufficient_standalone_query")
    if query.has_answer:
        reasons.append("query_contains_answer")
    if relevance.relevance < config.relevance_threshold:
        reasons.append("insufficient_relevance")
    return reasons


def run_multimodal_sdg(config: MultimodalSDGConfig) -> Path:
    """Run the EA workflow and return its validated-provenance portable handoff.

    Args:
        config: Explicit inputs, model roles, bounds, and split settings.

    Returns:
        Path to generation_result.json, published only after a complete export.

    Raises:
        ValueError: Input/config drift, invalid responses or no accepted queries.
        FileExistsError: Existing run unless resume was explicitly requested.
    """
    from data_designer_retrieval_sdg.multimodal.export import export_multimodal_bundle

    if config.combination_iterations and config.summary_embedding_endpoint:
        if not os.environ.get(config.summary_embedding_credential_env):
            raise ValueError(f"Set the credential environment variable {config.summary_embedding_credential_env}")
    root = config.output_dir.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock", timeout=0):
        sources = load_retrieval_sources(config.sources_file)
        validate_deployment(config, any(source.images for source in sources))
        contexts = load_contexts(config, sources)
        identity = {
            "config": config.model_dump(mode="json", exclude={"resume", "output_dir"}),
            "sources": [s.model_dump() for s in sources],
            "contexts": [c.model_dump() for c in contexts],
            "tokenizers": {
                role: digest(model.tokenizer_file)
                for role in ("generator", "judge")
                if (model := getattr(config, role)).tokenizer_file is not None
            },
            "images": {image: digest(Path(image)) for s in sources for image in s.images},
            "implementation": {
                p.relative_to(Path(__file__).parents[1]).as_posix(): digest(p)
                for p in sorted(Path(__file__).parents[1].rglob("*"))
                if p.is_file() and p.suffix in {".py", ".j2", ".json", ".txt"}
            },
            "dependencies": {
                name: version(name)
                for name in (
                    "data-designer",
                    "data-designer-engine",
                    "data-designer-config",
                    "pydantic",
                    "pyarrow",
                    "jinja2",
                )
                + (
                    (
                        "httpx" if config.summary_embedding_endpoint else "sentence-transformers",
                        "scikit-learn",
                        "umap-learn",
                        "numpy",
                    )
                    if config.combination_iterations
                    else ()
                )
            },
        }
        if root.exists() and not config.resume:
            raise FileExistsError(f"Run exists: {root}; inspect before explicitly resuming")
        if root.exists() and not (root / "identity.json").exists():
            raise ValueError("Existing run has no frozen identity")
        write_json(root / "identity.json", identity)
        if (root / "complete.json").exists():
            complete = json.loads((root / "complete.json").read_text())
            for record in complete["artifacts"]:
                path = root / record["path"]
                if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                    raise ValueError("Completed run integrity failure")
            return root / "bundle/generation_result.json"
        frozen = []
        for source in sources:
            images = []
            for image in source.images:
                path = Path(image)
                target = root / "inputs/assets" / (digest(path) + path.suffix.lower())
                write_bytes(target, path.read_bytes())
                images.append(str(target))
            frozen.append(source.model_copy(update={"images": images}))
        inference = DataDesignerInference(root / "inference", config)
        candidates, outcomes = generate_candidates(config, frozen, contexts, inference)
        write_json(root / "candidates.json", [c.model_dump(mode="json") for c in candidates])
        write_json(root / "context_outcomes.json", outcomes)
        handoff = export_multimodal_bundle(root / "bundle", frozen, candidates, outcomes, config)
        write_json(
            root / "complete.json",
            {"artifacts": [artifact(p, root) for p in sorted((root / "bundle").rglob("*")) if p.is_file()]},
        )
        return handoff
