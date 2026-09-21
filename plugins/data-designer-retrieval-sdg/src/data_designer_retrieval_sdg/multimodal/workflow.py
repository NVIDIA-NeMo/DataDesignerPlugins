# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic text/image evidence -> queries -> judgments -> localized positives."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from data_designer_retrieval_sdg.multimodal import prompts
from data_designer_retrieval_sdg.multimodal.inference import DataDesignerInference, Request
from data_designer_retrieval_sdg.multimodal.models import (
    Candidate,
    ContextSummary,
    GenerationContext,
    Localization,
    MultimodalSDGConfig,
    QueryBatch,
    QueryJudgment,
    RelevanceJudgment,
)
from data_designer_retrieval_sdg.multimodal.storage import artifact, digest, fingerprint, write_bytes, write_json
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource
from data_designer_retrieval_sdg.retrieval.source_file import load_retrieval_sources


def load_contexts(config: MultimodalSDGConfig, sources: list[RetrievalSource]) -> list[GenerationContext]:
    """Use explicit contexts or one per unit; never infer sections or split groups.

    Args:
        config: Optional context JSONL and explicit request size bound.
        sources: Complete canonical source collection.

    Returns:
        Validated contexts; unselected source units remain corpus distractors.
    """
    if config.contexts_file is None:
        contexts = [GenerationContext(context_id=s.unit_id, unit_ids=[s.unit_id], language=s.language) for s in sources]
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
        if not set(context.unit_ids) <= known or len(context.unit_ids) > config.max_units_per_context:
            raise ValueError(
                f"Unknown units or oversized context: {context.context_id}; prepare explicit bounded contexts"
            )
    return contexts


def source_request(
    instruction: str,
    schema: type,
    role: str,
    context: GenerationContext,
    sources: dict[str, RetrievalSource],
    **payload,
) -> Request:
    """Attach exact source text and aligned image pixels without truncation or templating source content."""
    selected = [sources[key] for key in context.unit_ids]
    data = {
        "sources": [{"unit_id": s.unit_id, "text": s.text} for s in selected],
        "image_unit_ids": [s.unit_id for s in selected for _ in s.images],
        **payload,
    }
    return Request(
        instruction + "\n" + json.dumps(data, ensure_ascii=False),
        schema,
        role,
        tuple(Path(image) for s in selected for image in s.images),
    )


def localization_reasons(
    localization: Localization, context: GenerationContext, sources: dict[str, RetrievalSource]
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
            if not quote or quote not in " ".join(source.text.split()):
                return ["unverified_text_evidence"]
        if support.modality in {"image", "text_and_image"} and (not source.images or not support.visual_evidence):
            return ["missing_visual_evidence"]
    return []


def generate_candidates(
    config: MultimodalSDGConfig,
    sources: list[RetrievalSource],
    contexts: list[GenerationContext],
    inference: DataDesignerInference,
) -> tuple[list[Candidate], list[dict]]:
    """Generate and judge bounded contexts, retaining abstentions and every rejected candidate."""
    by_id = {source.unit_id: source for source in sources}
    summaries = inference.generate(
        [source_request(prompts.SUMMARY, ContextSummary, "generator", c, by_id) for c in contexts]
    )
    batches = inference.generate(
        [
            source_request(
                prompts.GENERATE,
                QueryBatch,
                "generator",
                context,
                by_id,
                language=context.language,
                summary=summary.model_dump(),
                instructions=[{"slot": i, **item.model_dump()} for i, item in enumerate(config.instructions)],
            )
            for context, summary in zip(contexts, summaries, strict=True)
        ]
    )
    pending, outcomes = [], []
    for context, summary, batch in zip(contexts, summaries, batches, strict=True):
        if sorted(slot.slot for slot in batch.queries) != list(range(len(config.instructions))):
            raise ValueError("Generator did not return exactly one outcome for every requested slot")
        outcomes.append({"context": context.model_dump(), "summary": summary.model_dump(), "slots": batch.model_dump()})
        for slot in sorted(batch.queries, key=lambda s: s.slot):
            if slot.query is None:
                if slot.evidence_modality != "none":
                    raise ValueError("Abstention must declare evidence_modality=none")
                continue
            if not slot.query or slot.evidence_modality == "none":
                raise ValueError("Generated query must be nonempty and declare evidence")
            pending.append((context, slot))
    query_judgments = inference.generate(
        [
            Request(prompts.QUERY_JUDGE + "\n" + json.dumps({"query": slot.query}), QueryJudgment, "judge")
            for _, slot in pending
        ]
    )
    relevance = inference.generate(
        [
            source_request(prompts.RELEVANCE, RelevanceJudgment, "judge", c, by_id, query=slot.query)
            for c, slot in pending
        ]
    )
    reasons_by_index = [quality_reasons(q, r, config) for q, r in zip(query_judgments, relevance, strict=True)]
    localized_indexes = [i for i, reasons in enumerate(reasons_by_index) if not reasons]
    localizations = inference.generate(
        [
            source_request(prompts.LOCALIZE, Localization, "judge", pending[i][0], by_id, query=pending[i][1].query)
            for i in localized_indexes
        ]
    )
    by_index = dict(zip(localized_indexes, localizations, strict=True))
    candidates = []
    for i, ((context, slot), query_judge, relevance_judge) in enumerate(
        zip(pending, query_judgments, relevance, strict=True)
    ):
        reasons = reasons_by_index[i]
        localization = by_index.get(i)
        if localization is not None:
            reasons.extend(localization_reasons(localization, context, by_id))
        candidates.append(
            Candidate(
                query_id="q_" + fingerprint([context.context_id, slot.slot, slot.query])[:32],
                context_id=context.context_id,
                query=slot.query,
                source_unit_ids=context.unit_ids,
                language=context.language,
                requested_instruction=config.instructions[slot.slot],
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

    root = config.output_dir.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock", timeout=0):
        sources = load_retrieval_sources(config.sources_file)
        contexts = load_contexts(config, sources)
        identity = {
            "config": config.model_dump(mode="json", exclude={"resume", "output_dir"}),
            "sources": [s.model_dump() for s in sources],
            "contexts": [c.model_dump() for c in contexts],
            "images": {image: digest(Path(image)) for s in sources for image in s.images},
            "implementation": {
                p.relative_to(Path(__file__).parents[1]).as_posix(): digest(p)
                for p in sorted(Path(__file__).parents[1].rglob("*.py"))
            },
            "dependencies": {
                name: version(name)
                for name in ("data-designer", "data-designer-engine", "data-designer-config", "pydantic", "pyarrow")
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
