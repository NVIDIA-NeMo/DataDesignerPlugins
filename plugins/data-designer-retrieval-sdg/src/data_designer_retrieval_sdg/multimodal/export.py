# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable shared-corpus export of recorded retrieval-first judgments."""

from __future__ import annotations

import errno
import json
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from data_designer_retrieval_sdg.multimodal.models import Candidate, GenerationContext, MultimodalSDGConfig
from data_designer_retrieval_sdg.multimodal.query_groups import (
    QueryGroupInput,
    assign_query_group_splits,
    normalized_query,
    resolve_query_groups,
)
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource
from data_designer_retrieval_sdg.multimodal.storage import artifact, digest, write_bytes, write_json, write_jsonl

VIEWS = {
    "text": ("id", "TextQADataset"),
    "image": ("image_filename", "ColPaliDataset"),
    "image_and_text": ("docid", "WikiSSNQDataset"),
    "mixed": ("docid", "WikiSSNQDataset"),
}


def checked_candidates(
    sources: list[RetrievalSource], candidates: list[Candidate], config: MultimodalSDGConfig
) -> list[Candidate]:
    """Require acceptance to agree with recorded judgments, never synthesize gate labels."""
    from data_designer_retrieval_sdg.multimodal.workflow import localization_reasons, quality_reasons

    by_id = {source.unit_id: source for source in sources}
    if len(by_id) != len(sources) or len({c.query_id for c in candidates}) != len(candidates):
        raise ValueError("Source and query IDs must be unique")
    for candidate in candidates:
        if not set(candidate.source_unit_ids) <= by_id.keys():
            raise ValueError("Candidate references unknown source units")
        reasons = quality_reasons(candidate.query_judgment, candidate.relevance_judgment, config)
        if not reasons:
            context = GenerationContext(context_id=candidate.context_id, unit_ids=candidate.source_unit_ids)
            reasons = (
                localization_reasons(candidate.localization, context, by_id, config.require_verbatim_quotes)
                if candidate.localization
                else ["no_localized_positives"]
            )
        if candidate.accepted != (not reasons) or candidate.rejection_reasons != reasons:
            raise ValueError("Recorded acceptance disagrees with source evidence or quality judgments")
    return [candidate for candidate in candidates if candidate.accepted]


def view_candidates(view: str, units: list[dict], candidates: list[Candidate]) -> list[Candidate]:
    """Keep all positives or exclude the query; never discard the localized evidence modality."""
    ids = {unit["unit_id"] for unit in units}
    return [
        c
        for c in candidates
        if all(
            s.unit_id in ids and (view in {"image_and_text", "mixed"} or s.modality == view)
            for s in c.localization.supports
        )
    ]


def write_view(
    root: Path,
    view: str,
    units: list[dict],
    candidates: list[Candidate],
    groups: dict[str, str],
    assignments: dict[str, str],
    dataset_id: str,
) -> dict:
    """Write one modality view with a shared eligible corpus and every graded positive."""
    directory = root / "views" / view
    corpus = directory / "corpus/shared"
    corpus.mkdir(parents=True)
    id_column, loader = VIEWS[view]
    fields = [(id_column, pa.string())]
    if view != "image":
        fields.append(("text", pa.string()))
    if view != "text":
        fields.append(("image", pa.binary()))
    for offset in range(0, len(units), 32):
        rows = []
        for unit in units[offset : offset + 32]:
            row = {id_column: unit["unit_id"]}
            if view != "image":
                row["text"] = unit["text"]
            if view != "text":
                row["image"] = (root / unit["images"][0]).read_bytes() if unit["images"] else None
            rows.append(row)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=pa.schema(fields)), corpus / f"part-{offset // 32:05d}.parquet"
        )
    write_json(corpus / "merlin_metadata.json", {"class": loader, "format": "parquet", "corpus_id": dataset_id})
    by_id = {unit["unit_id"]: unit for unit in units}
    split_rows = {name: [] for name in ("train", "validation", "evaluation")}
    query_assignments, qrels = {}, ["query-id\tcorpus-id\tscore\n"]
    for candidate in candidates:
        partition = assignments[groups[candidate.query_id]]
        query_assignments[candidate.query_id] = {"split": partition, "query_group_id": groups[candidate.query_id]}
        supports = candidate.localization.supports
        row = {
            "question_id": candidate.query_id,
            "question": candidate.query,
            "query_group_id": groups[candidate.query_id],
            "corpus_id": dataset_id,
            "pos_doc": [{"id": s.unit_id, "score": s.grade} for s in supports],
            "neg_doc": [],
            "negative_scores": [],
            "positive_unit_ids": [s.unit_id for s in supports],
            "source_document_ids": sorted({by_id[s.unit_id]["document_id"] for s in supports}),
            "language": candidate.language,
            "relevance_judgements_complete": False,
            "negative_mining_performed": False,
            "unlisted_document_disposition": "unjudged",
        }
        split_rows[partition].append(row)
        if partition == "evaluation":
            qrels.extend(f"{candidate.query_id}\t{s.unit_id}\t{s.grade}\n" for s in supports)
    for name in ("train", "validation"):
        write_json(directory / f"{name}.json", {"corpus": {"path": "corpus/shared"}, "data": split_rows[name]})
    evaluation = root / "synthetic_eval" / view
    write_jsonl(
        evaluation / "queries.jsonl",
        [
            {
                "_id": row["question_id"],
                "text": row["question"],
                "query_group_id": row["query_group_id"],
                "metadata": {"language": row["language"]},
            }
            for row in split_rows["evaluation"]
        ],
    )
    documents = []
    for unit in units:
        row = {"_id": unit["unit_id"], "title": "", "metadata": {"source_document_id": unit["document_id"]}}
        if view != "image":
            row["text"] = unit["text"]
        if view != "text" and unit["images"]:
            row["image_path"] = unit["images"][0]
        documents.append(row)
    write_jsonl(evaluation / "corpus.jsonl", documents)
    write_bytes(evaluation / "qrels/test.tsv", "".join(qrels).encode())
    return query_assignments


def bundle_files(root: Path) -> dict[str, str]:
    """Hash regular bundle files and reject symbolic links before resume comparison.

    Args:
        root: A completed candidate bundle directory.

    Returns:
        Relative file paths and their content hashes.

    Raises:
        ValueError: The tree contains symbolic links or nonregular entries, or is not a directory.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Bundle must be a regular directory")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in paths):
        raise ValueError("Bundle must contain only regular files and directories, without symbolic links")
    return {path.relative_to(root).as_posix(): digest(path) for path in paths if path.is_file()}


def export_multimodal_bundle(
    root: Path,
    sources: list[RetrievalSource],
    candidates: list[Candidate],
    outcomes: list[dict],
    config: MultimodalSDGConfig,
) -> Path:
    """Publish a complete portable bundle, retaining an identical bundle on resume.

    Args:
        root: Destination that must be absent, or byte-identical on explicit resume.
        sources: Complete canonical corpus, including unselected distractors.
        candidates: Every terminal candidate and its original gate judgments.
        outcomes: Context summaries and generation slots, including abstentions.
        config: Generation and export policy, including explicit resume permission.

    Returns:
        The published portable generation_result.json path.

    Raises:
        FileExistsError: A non-resumed, partial, or different destination exists.
        ValueError: Candidates violate quality or source integrity requirements.
    """
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".export.lock", timeout=0):
        if root.exists() and not config.resume:
            raise FileExistsError(errno.EEXIST, "Export destination already exists", str(root))
        with tempfile.TemporaryDirectory(prefix=f".{root.name}-staging-", dir=root.parent) as directory:
            staged = Path(directory) / "bundle"
            write_multimodal_bundle(staged, sources, candidates, outcomes, config)
            if root.exists() or root.is_symlink():
                if not config.resume or bundle_files(root) != bundle_files(staged):
                    raise FileExistsError(errno.EEXIST, "Existing export is partial or differs", str(root))
            else:
                staged.rename(root)
    return root / "generation_result.json"


def write_multimodal_bundle(
    root: Path,
    sources: list[RetrievalSource],
    candidates: list[Candidate],
    outcomes: list[dict],
    config: MultimodalSDGConfig,
) -> Path:
    """Export a fresh schema-v2 bundle; never overwrite a partial or complete export.

    Args:
        root: New destination directory.
        sources: Complete canonical corpus, including unselected distractors.
        candidates: Every terminal candidate and its actual gate judgments.
        outcomes: Context summaries and all generation slots, including abstentions.
        config: Generation identity and corpus-independent group split policy.

    Returns:
        Portable generation_result.json path for the Nemotron recipe consumer.
    """
    accepted = checked_candidates(sources, candidates, config)
    if not accepted:
        raise ValueError("No candidates passed retrieval quality and localization gates; inspect candidates.json")
    groups = resolve_query_groups(
        [QueryGroupInput(query_id=c.query_id, text=c.query, **c.provenance.model_dump()) for c in accepted],
        group_near_duplicates=config.group_near_duplicates,
    )
    assignments = assign_query_group_splits(groups.values(), ratios=config.ratios, seed=config.seed)
    root.mkdir(parents=True, exist_ok=False)
    units = []
    for source in sources:
        images = []
        for image in source.images:
            path = Path(image)
            target = root / "assets" / (digest(path) + path.suffix.lower())
            write_bytes(target, path.read_bytes())
            images.append(target.relative_to(root).as_posix())
        units.append(
            {
                "unit_id": source.unit_id,
                "document_id": source.document_id,
                "text": source.text,
                "images": images,
                "split": "shared",
                "language": source.language,
            }
        )
    write_jsonl(root / "retrieval_units.jsonl", units)
    write_jsonl(root / "source_records.jsonl", [c.model_dump(mode="json") for c in candidates])
    write_jsonl(root / "candidate_diagnostics.jsonl", [c.model_dump(mode="json") for c in candidates])
    write_json(root / "context_outcomes.json", outcomes)
    query_assignments, view_counts = {}, {}
    for view in VIEWS:
        eligible = [
            unit
            for unit in units
            if view == "mixed" or (bool(unit["text"].strip()) if view == "text" else bool(unit["images"]))
        ]
        if not eligible:
            continue
        selected = view_candidates(view, eligible, accepted)
        query_assignments[view] = write_view(root, view, eligible, selected, groups, assignments, config.dataset_id)
        view_counts[view] = {
            "corpus_units": len(eligible),
            "accepted_queries": len(selected),
            "excluded_queries": len(accepted) - len(selected),
            "split_counts": dict(Counter(assignments[groups[c.query_id]] for c in selected)),
        }
    write_json(
        root / "split_manifest.json",
        {
            "split_protocol": "grouped_query_disjoint",
            "corpus_scope": "full_collection",
            "document_disjoint": False,
            "seed": config.seed,
            "ratios": config.ratios.model_dump(),
            "group_near_duplicates": config.group_near_duplicates,
            "query_assignments": query_assignments,
        },
    )
    counts = {
        "requested_slots": sum(len(outcome["slots"]["queries"]) for outcome in outcomes),
        "abstentions": sum(slot["query"] is None for outcome in outcomes for slot in outcome["slots"]["queries"]),
        "candidate_records": len(candidates),
        "accepted_query_ids": len(accepted),
        "accepted_normalized_texts": len({normalized_query(c.query) for c in accepted}),
        "query_groups": len(set(groups.values())),
        "corpus_units": len(units),
        "source_documents": len({u["document_id"] for u in units}),
        "positive_annotations": sum(len(c.localization.supports) for c in accepted),
        "rejection_reasons": dict(Counter(reason for c in candidates for reason in c.rejection_reasons)),
        "views": view_counts,
    }
    from data_designer_retrieval_sdg.multimodal.reporting import generation_report

    failed_query_ids = set()
    if config.failed_record_policy == "skip":
        failures = json.loads((config.output_dir / "generation_failures.json").read_text())
        write_json(root / "generation_failures.json", failures)
        failed_query_ids = {row["query_id"] for row in failures["records"] if "query_id" in row}
        failed_slots = sum(
            len(row["instructions"]) for row in outcomes if row.get("selection_reason") == "generation_failed"
        )
        counts["requested_slots"] += failed_slots
        counts["generation_failures"] = {
            "policy": "skip",
            "failed_records": len(failures["records"]),
            "failed_requested_slots": failed_slots,
            "excluded_query_ids": len(failed_query_ids),
            "by_stage": dict(Counter(row["stage"] for row in failures["records"])),
            "diagnostics": "generation_failures.json",
        }
    counts.update(generation_report(sources, candidates, outcomes, failed_query_ids))
    write_json(root / "report.json", counts)
    write_json(
        root / "run_manifest.json",
        {
            "schema_version": 2,
            "status": "completed",
            "portable": True,
            "producer": "data-designer-retrieval-sdg",
            "quality_contract": "retrieval_first_v1",
            "policy": {
                "relevance_threshold": config.relevance_threshold,
                "self_sufficiency_threshold": config.self_sufficiency_threshold,
                "instructions": [item.model_dump() for item in config.instructions],
                "require_verbatim_quotes": config.require_verbatim_quotes,
                "generation_config": config.model_dump(
                    mode="json",
                    exclude={
                        "generator",
                        "judge",
                        "sources_file",
                        "source_directory",
                        "contexts_file",
                        "output_dir",
                        "resume",
                    },
                ),
            },
            "models": {
                "generator": config.generator.model_dump(mode="json"),
                "judge": config.judge.model_dump(mode="json"),
            },
            "shared_generator_judge": config.generator.model == config.judge.model
            and config.generator.endpoint == config.judge.endpoint,
            "dataset_id": config.dataset_id,
            "split_protocol": "grouped_query_disjoint",
            "corpus_scope": "full_collection",
            "relevance_annotation_scope": "known_positives_only",
            "negative_mining_performed": False,
            "relevance_judgements_complete": False,
            "unlisted_document_disposition": "unjudged",
            "gold_evaluation_status": "unavailable",
            "counts": counts,
            "artifacts": [artifact(path, root) for path in sorted(root.rglob("*")) if path.is_file()],
        },
    )
    write_json(
        root / "generation_result.json",
        {
            "schema_version": 1,
            "dataset_name": config.dataset_id,
            "output_path": "source_records.jsonl",
            "portable_bundle_manifest": "run_manifest.json",
            "portable_bundle_sha256": digest(root / "run_manifest.json"),
        },
    )
    return root / "generation_result.json"
