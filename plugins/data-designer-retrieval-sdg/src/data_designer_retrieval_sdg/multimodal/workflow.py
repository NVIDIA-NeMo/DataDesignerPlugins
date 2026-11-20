# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic text/image evidence -> queries -> judgments -> localized positives."""

from __future__ import annotations

import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from data_designer_retrieval_sdg.multimodal import prompts
from data_designer_retrieval_sdg.multimodal.bounds import (
    validate_deployment,
)
from data_designer_retrieval_sdg.multimodal.inference import Request, source_request
from data_designer_retrieval_sdg.multimodal.models import (
    ContextSummary,
    GenerationContext,
    Localization,
    MultimodalSDGConfig,
    QueryBatch,
    QueryJudgment,
    QueryMetadata,
    QuerySlot,
    RelevanceJudgment,
    SelfSufficiencyJudgment,
)
from data_designer_retrieval_sdg.multimodal.planning import (
    automatic_contexts,
    bounded_contexts,
    context_instructions,
)
from data_designer_retrieval_sdg.multimodal.preflight import check_models
from data_designer_retrieval_sdg.multimodal.source_directory import load_directory_sources
from data_designer_retrieval_sdg.multimodal.source_file import load_retrieval_sources
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource
from data_designer_retrieval_sdg.multimodal.storage import artifact, digest, write_bytes, write_json, write_jsonl


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


def normalize_localization(
    localization: Localization,
    context: GenerationContext,
    sources: dict[str, RetrievalSource],
    require_verbatim_quotes: bool = False,
) -> Localization:
    """Prefer valid duplicates while preserving invalid-only units for rejection.

    Args:
        localization: Unmodified judge response, possibly containing duplicates.
        context: Source membership the judge was allowed to cite.
        sources: Canonical records used to validate each claimed evidence carrier.
        require_verbatim_quotes: Whether text support requires a matching quote.

    Returns:
        One support per known unit, in first-unit-occurrence order. Select the
        highest-grade valid support when available, otherwise the highest-grade
        original support for downstream validation. Equal grades retain the first.
    """
    supports, valid_supports = {}, {}
    for support in localization.supports:
        if support.unit_id not in context.unit_ids:
            continue
        if support.unit_id not in supports or support.grade > supports[support.unit_id].grade:
            supports[support.unit_id] = support
        if not localization_reasons(Localization(supports=[support]), context, sources, require_verbatim_quotes) and (
            support.unit_id not in valid_supports or support.grade > valid_supports[support.unit_id].grade
        ):
            valid_supports[support.unit_id] = support
    return Localization(supports=[valid_supports.get(unit_id, support) for unit_id, support in supports.items()])


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


def judgment_requests(
    context: GenerationContext, slot: QuerySlot, sources: dict[str, RetrievalSource]
) -> list[Request]:
    """Build every downstream prompt using the actual generated query and evidence.

    Args:
        context: Membership of the source context used to generate the query.
        slot: Generated query to classify, judge, and localize.
        sources: Original source records keyed by unit ID.

    Returns:
        Requests for query metadata, self-sufficiency, relevance, and localization,
        in that order, with each request assigned to its configured judge stage.
    """
    return [
        Request(
            prompts.render("query_metadata", query=slot.query), QueryMetadata, "judge", judge_stage="query_metadata"
        ),
        Request(
            prompts.render("self_sufficiency", summary=source_text(context, sources), query=slot.query),
            SelfSufficiencyJudgment,
            "judge",
            judge_stage="self_sufficiency",
        ),
        source_request(
            prompts.render("relevancy", summary=source_text(context, sources), query=slot.query),
            RelevanceJudgment,
            "judge",
            context,
            sources,
            judge_stage="relevance",
            query=slot.query,
        ),
        source_request(
            prompts.LOCALIZE, Localization, "judge", context, sources, judge_stage="localization", query=slot.query
        ),
    ]


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
    """Run the Data Designer workflow and export a checksummed portable bundle.

    Args:
        config: Explicit inputs, model roles, bounds, and split settings.

    Returns:
        Path to generation_result.json, published only after a complete export.

    Raises:
        ValueError: Input/config drift, invalid responses or no accepted queries.
        FileExistsError: Existing run unless resume was explicitly requested.
    """
    from data_designer_retrieval_sdg.multimodal.execution import generate_candidates, verify_boundaries
    from data_designer_retrieval_sdg.multimodal.export import export_multimodal_bundle

    root = config.output_dir.resolve()
    if config.source_directory is not None and root.is_relative_to(config.source_directory.path.resolve()):
        raise ValueError("output_dir must be outside source_directory.path to avoid ingesting generated files")
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock", timeout=0):
        source_manifest = None
        if config.source_directory is not None:
            sources, source_manifest = load_directory_sources(config.source_directory)
        else:
            sources = load_retrieval_sources(config.sources_file)
        validate_deployment(config, any(source.images for source in sources))
        contexts = load_contexts(config, sources)
        identity = run_identity(config, sources, contexts)
        if source_manifest is not None:
            identity["source_files"] = source_manifest["files"]
            if "source_preparation" in source_manifest:
                identity["source_preparation"] = source_manifest["source_preparation"]
            source_root = config.source_directory.path.resolve()
            for record in source_manifest["files"]:
                image = str(source_root / record["path"])
                if image in identity["images"] and identity["images"][image] != record["sha256"]:
                    raise ValueError(f"Source image changed during preparation: {image}")
        if root.exists() and not config.resume:
            raise FileExistsError(f"Run exists: {root}; inspect before explicitly resuming")
        if root.exists() and not (root / "identity.json").exists():
            raise ValueError("Existing run has no frozen identity")
        if root.exists():
            write_json(root / "identity.json", identity)
        if (root / "complete.json").exists():
            verify_boundaries(root / "data_designer", require_complete=True)
            complete = json.loads((root / "complete.json").read_text())
            for record in complete["artifacts"]:
                path = root / record["path"]
                if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                    raise ValueError("Completed run integrity failure")
            return root / "bundle/generation_result.json"
        if (root / "data_designer").exists():
            verify_boundaries(root / "data_designer")
        check_models(config)
        write_json(root / "identity.json", identity)
        frozen = []
        for source in sources:
            images = []
            for image in source.images:
                path = Path(image)
                content = path.read_bytes()
                checksum = sha256(content).hexdigest()
                if source_manifest is not None and checksum != identity["images"][image]:
                    raise ValueError(f"Source image changed during preparation: {image}")
                target = root / "inputs/assets" / (checksum + path.suffix.lower())
                write_bytes(target, content)
                images.append(str(target))
            frozen.append(source.model_copy(update={"images": images}))
        if source_manifest is not None:
            write_json(root / "inputs/source_manifest.json", source_manifest)
            write_jsonl(
                root / "inputs/sources.jsonl",
                [
                    source.model_dump(mode="json")
                    | {"images": [Path(image).relative_to(root / "inputs").as_posix() for image in source.images]}
                    for source in frozen
                ],
            )
        candidates, outcomes = generate_candidates(config, frozen, contexts)
        write_json(root / "candidates.json", [c.model_dump(mode="json") for c in candidates])
        write_json(root / "context_outcomes.json", outcomes)
        handoff = export_multimodal_bundle(root / "bundle", frozen, candidates, outcomes, config)
        completed_paths = sorted((root / "bundle").rglob("*"))
        completed_paths.extend(sorted((root / "inputs").rglob("*")))
        if config.failed_record_policy == "skip":
            completed_paths.append(root / "generation_failures.json")
        write_json(
            root / "complete.json",
            {"artifacts": [artifact(p, root) for p in completed_paths if p.is_file()]},
        )
        return handoff


def run_identity(
    config: MultimodalSDGConfig, sources: list[RetrievalSource], contexts: list[GenerationContext]
) -> dict:
    """Bind resumable artifacts to configuration, evidence, code and dependencies."""
    return {
        "config": config.model_dump(mode="json", exclude={"resume", "output_dir"}),
        "sources": [s.model_dump() for s in sources],
        "contexts": [c.model_dump() for c in contexts],
        "tokenizers": {
            role: digest(model.tokenizer_file)
            for role, model in {"generator": config.generator, **config.active_judges()}.items()
            if model.context_window_tokens is not None and model.tokenizer_file is not None
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
