# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe-oriented export for judged text and image retrieval data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from data_designer_retrieval_sdg.models import (
    DocumentArtifacts,
    EvidenceModality,
    QAEvaluation,
    QueryQualityEvaluation,
    QuestionAnswerPair,
    SourceAssessment,
)
from data_designer_retrieval_sdg.retrieval.models import (
    CandidateDiagnostic,
    ExportSummary,
    GeneratedRetrievalRecord,
    RetrievalUnit,
    SplitName,
    SplitRatios,
    ViewName,
)
from data_designer_retrieval_sdg.retrieval.quality import (
    deterministic_rejection_reason,
    normalize_query_text,
)

_SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "evaluation")
_VIEW_NAMES: tuple[ViewName, ...] = ("text", "image", "image_and_text")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class _AcceptedCandidate:
    record: GeneratedRetrievalRecord
    candidate_index: int
    candidate: QuestionAnswerPair
    positive_units: tuple[RetrievalUnit, ...]
    evidence_modality: EvidenceModality
    query_id: str


@dataclass(frozen=True)
class _CandidateDecision:
    record: GeneratedRetrievalRecord
    candidate_index: int
    candidate: QuestionAnswerPair
    positive_units: tuple[RetrievalUnit, ...]
    evidence_modality: EvidenceModality | None
    diagnostic: CandidateDiagnostic


class _UnionFind:
    """Disjoint sets for source documents linked by one generation row."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, value: str) -> str:
        """Return the representative for a source document."""
        if value not in self._parent:
            self._parent[value] = value
            self._rank[value] = 0
        if self._parent[value] != value:
            self._parent[value] = self.find(self._parent[value])
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        """Join two linked source-document sets."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1


def load_generated_retrieval_records(input_path: str | Path) -> list[GeneratedRetrievalRecord]:
    """Load completed shared-pipeline rows from JSONL or Parquet.

    Args:
        input_path: Data Designer export file.

    Returns:
        Validated source records.

    Raises:
        ValueError: If the format, rows, source IDs, or unit IDs are invalid.
    """
    path = Path(input_path).resolve()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".parquet":
        rows = pq.read_table(path).to_pylist()
    else:
        raise ValueError("retrieval export input must be a .jsonl or .parquet file")
    records = [GeneratedRetrievalRecord.model_validate(_normalize_nested_columns(row)) for row in rows]
    if not records:
        raise ValueError("retrieval export input contains no rows")
    source_ids = [record.source_id for record in records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("retrieval export input contains duplicate source_id values")
    unit_ids = [unit.unit_id for record in records for unit in record.retrieval_units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("retrieval export input contains duplicate retrieval unit IDs")
    return records


def assign_document_splits(
    document_ids: Iterable[str],
    *,
    ratios: SplitRatios | None = None,
    seed: int = 42,
) -> dict[str, SplitName]:
    """Assign whole source documents to deterministic splits.

    Args:
        document_ids: Source-document or connected-component identifiers.
        ratios: Requested split ratios.
        seed: Stable split seed.

    Returns:
        Mapping from identifier to split.
    """
    effective_ratios = ratios or SplitRatios()
    unique_ids = sorted(set(document_ids), key=lambda value: _stable_order_key(value, seed))
    if not unique_ids:
        return {}
    values = {
        "train": effective_ratios.train,
        "validation": effective_ratios.validation,
        "evaluation": effective_ratios.evaluation,
    }
    raw_counts = {name: len(unique_ids) * values[name] for name in _SPLIT_NAMES}
    counts = {name: math.floor(raw_counts[name]) for name in _SPLIT_NAMES}
    remainder = len(unique_ids) - sum(counts.values())
    ranked = sorted(
        _SPLIT_NAMES,
        key=lambda name: (raw_counts[name] - counts[name], values[name], name),
        reverse=True,
    )
    for name in ranked[:remainder]:
        counts[name] += 1
    positive = [name for name in _SPLIT_NAMES if values[name] > 0]
    if len(unique_ids) >= len(positive):
        for name in positive:
            if counts[name] > 0:
                continue
            donors = [candidate for candidate in _SPLIT_NAMES if counts[candidate] > 1]
            donor = max(donors, key=lambda candidate: counts[candidate] - raw_counts[candidate])
            counts[donor] -= 1
            counts[name] += 1
    assignments: dict[str, SplitName] = {}
    cursor = 0
    for name in _SPLIT_NAMES:
        for document_id in unique_ids[cursor : cursor + counts[name]]:
            assignments[document_id] = name
        cursor += counts[name]
    return assignments


def export_retrieval_data(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str,
    ratios: SplitRatios | None = None,
    seed: int = 42,
    strict_visual: bool = False,
    generator_model: str | None = None,
    judge_model: str | None = None,
    answer_model: str | None = None,
) -> ExportSummary:
    """Export accepted candidates into independently sampleable views.

    Args:
        input_path: Completed Data Designer JSONL or Parquet export.
        output_dir: New or empty destination directory.
        dataset_id: Corpus identifier written into recipe records.
        ratios: Source-document-level split ratios.
        seed: Stable split seed.
        strict_visual: Fail if image inputs produced no visual candidates.
        generator_model: Optional auditable generator model identity.
        judge_model: Optional auditable judge model identity.
        answer_model: Optional separate answer model; defaults to generator_model.

    Returns:
        Counts, warnings, and manifest location.

    Raises:
        ValueError: If the input or requested export is invalid.
    """
    if not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")
    if (generator_model is None) != (judge_model is None):
        raise ValueError("generator_model and judge_model must be supplied together")
    if answer_model is not None and (not answer_model.strip() or generator_model is None):
        raise ValueError("answer_model requires a non-empty identity and generator_model/judge_model")
    input_file = Path(input_path).resolve()
    records = load_generated_retrieval_records(input_file)
    decisions = _quarantine_cross_unit_collisions(_candidate_decisions(records))
    early_rejections = sum(len(record.generation_diagnostics) for record in records)
    accepted = _accepted_candidates(decisions)
    if not accepted:
        reasons = sorted(
            {
                item.diagnostic.rejection_reason.split(":", 1)[0]
                for item in decisions
                if item.diagnostic.rejection_reason is not None
            }
        )
        raise ValueError(
            "no candidates passed deterministic, query-quality, and grounding checks; "
            "rejection reasons: " + ", ".join(reasons)
        )

    warnings = _coverage_warnings(records, accepted)
    warnings.extend(_model_warnings(generator_model, judge_model))
    if answer_model is not None and answer_model == judge_model and answer_model != generator_model:
        warnings.append("SHARED_ANSWER_JUDGE_MODEL: an independent answer judge is recommended")
    if strict_visual and any(warning.startswith("NO_VISUAL_CANDIDATES") for warning in warnings):
        raise ValueError("strict visual coverage requires at least one accepted image-grounded candidate")

    destination = Path(output_dir).resolve()
    _require_empty_output(destination)
    destination.mkdir(parents=True)
    split_ratios = ratios or SplitRatios()
    assignments = _connected_document_splits(records, split_ratios, seed)
    candidate_split_counts = Counter(_candidate_split(item, assignments) for item in accepted)
    warnings.extend(_split_warnings(assignments, candidate_split_counts, split_ratios))
    assets = _materialize_assets(records, input_file.parent, destination)
    generated_paths = _write_audit_files(destination, records, decisions, assignments, assets)
    generated_paths.extend(_write_views(destination, records, accepted, assignments, assets, dataset_id))

    split_counts = Counter(assignments.values())
    surface_counts = Counter(item.candidate.query_surface for item in accepted)
    modality_counts = Counter(item.evidence_modality for item in accepted)
    units = [unit for record in records for unit in record.retrieval_units]
    manifest_path = destination / "run_manifest.json"
    manifest = {
        "schema_version": 2,
        "quality_contract": "independent_source_assessment_v1",
        "dataset_id": dataset_id,
        "models": {
            "generator": generator_model,
            "judge": judge_model,
            "answer": answer_model or generator_model,
            "distinct": generator_model != judge_model if generator_model is not None else None,
        },
        "source_record_count": len(records),
        "retrieval_unit_count": len(units),
        "accepted_candidate_count": len(accepted),
        "rejected_candidate_count": early_rejections + len(decisions) - len(accepted),
        "pre_grounding_rejected_candidate_count": early_rejections,
        "rejection_reason_counts": dict(
            Counter(
                reason.split(":", 1)[0]
                for reason in [
                    *(item.rejection_reason for record in records for item in record.generation_diagnostics),
                    *(item.diagnostic.rejection_reason for item in decisions if not item.diagnostic.accepted),
                ]
                if reason is not None
            )
        ),
        "source_suitability_counts": _source_suitability_counts(records),
        "document_split_counts": {name: split_counts[name] for name in _SPLIT_NAMES},
        "candidate_split_counts": {name: candidate_split_counts[name] for name in _SPLIT_NAMES},
        "query_surface_counts": {name: surface_counts[name] for name in ("question", "instruction", "keyword")},
        "evidence_modality_counts": {name: modality_counts[name] for name in ("text_only", "image_grounded")},
        "warnings": warnings,
        "artifacts": [_artifact_record(path, destination) for path in sorted(generated_paths)],
    }
    _write_json(manifest_path, manifest)
    return ExportSummary(
        output_dir=str(destination),
        run_manifest_path=str(manifest_path),
        source_record_count=len(records),
        retrieval_unit_count=len(units),
        accepted_candidate_count=len(accepted),
        rejected_candidate_count=manifest["rejected_candidate_count"],
        document_split_counts=manifest["document_split_counts"],
        candidate_split_counts=manifest["candidate_split_counts"],
        query_surface_counts=manifest["query_surface_counts"],
        evidence_modality_counts=manifest["evidence_modality_counts"],
        warnings=warnings,
    )


def _normalize_nested_columns(row: dict[str, Any]) -> dict[str, Any]:
    """Decode structured columns serialized as JSON strings."""
    normalized = dict(row)
    for name in (
        "retrieval_units",
        "deduplicated_qa_pairs",
        "query_quality_evaluations",
        "qa_evaluations",
        "source_assessments",
        "document_artifacts",
        "generation_diagnostics",
    ):
        value = normalized.get(name)
        if isinstance(value, str):
            normalized[name] = json.loads(value)
    artifacts = normalized.get("document_artifacts")
    if isinstance(artifacts, dict):
        # Arrow may fill omitted optional artifact fields with null when merging
        # row schemas. Restore only declared empty-list defaults, never evidence
        # or required quality/profile fields.
        optional_lists = {
            name for name, field in DocumentArtifacts.model_fields.items() if field.default_factory is list
        }
        normalized["document_artifacts"] = {
            name: value for name, value in artifacts.items() if value is not None or name not in optional_lists
        }
    if normalized.get("qa_evaluations") is None and normalized.get("deduplicated_qa_pairs") == []:
        normalized["qa_evaluations"] = {"evaluations": []}
    return normalized


def _source_suitability_counts(records: list[GeneratedRetrievalRecord]) -> dict[str, int]:
    """Count advisory source profiles without conflating them with grounding."""
    counts: Counter[str] = Counter()
    for record in records:
        profile = record.document_artifacts.retrieval_profile if record.document_artifacts else None
        if profile is None:
            counts["unknown"] += 1
        else:
            counts["visual_useful"] += int(profile.visual_retrieval_useful)
            counts["text_useful"] += int(profile.text_retrieval_useful)
            counts["neither_useful"] += int(not profile.visual_retrieval_useful and not profile.text_retrieval_useful)
    return dict(counts)


def _stable_order_key(identifier: str, seed: int) -> str:
    """Return a stable split-order key."""
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def _candidate_decisions(records: list[GeneratedRetrievalRecord]) -> list[_CandidateDecision]:
    """Apply positive-unit, deterministic, and independent-judge gates."""
    decisions: list[_CandidateDecision] = []
    for record in records:
        units = {unit.unit_id: unit for unit in record.retrieval_units}
        query_by_index = {item.candidate_index: item for item in record.query_quality_evaluations.evaluations}
        grounding_by_index = {item.candidate_index: item for item in record.qa_evaluations.evaluations}
        source_by_index = (
            {item.candidate_index: item for item in record.source_assessments.assessments}
            if record.source_assessments is not None
            else {}
        )
        for index, candidate in enumerate(record.deduplicated_qa_pairs):
            positive_units = tuple(units[unit_id] for unit_id in candidate.positive_unit_ids if unit_id in units)
            rejection = _positive_unit_rejection(candidate, units)
            grounding = grounding_by_index.get(index)
            if rejection is None:
                rejection = deterministic_rejection_reason(
                    candidate.question,
                    candidate.answer,
                    candidate.query_surface,
                )
            if rejection is None:
                rejection = _judge_rejection_reason(query_by_index.get(index), grounding)
            if rejection is None and grounding is not None:
                rejection = _modality_rejection_reason(grounding.evidence_modality, positive_units)
            if rejection is None and grounding is not None:
                rejection = _native_text_evidence_rejection(grounding, positive_units)
            if rejection is None and grounding is not None:
                rejection = _source_assessment_rejection(source_by_index.get(index), grounding, positive_units)
            evidence_modality = grounding.evidence_modality if grounding is not None else None
            decisions.append(
                _CandidateDecision(
                    record=record,
                    candidate_index=index,
                    candidate=candidate,
                    positive_units=positive_units,
                    evidence_modality=evidence_modality,
                    diagnostic=CandidateDiagnostic(
                        source_id=record.source_id,
                        candidate_index=index,
                        question=candidate.question,
                        answer=candidate.answer,
                        positive_unit_ids=candidate.positive_unit_ids,
                        query_surface=candidate.query_surface,
                        evidence_modality=evidence_modality,
                        accepted=rejection is None,
                        rejection_reason=rejection,
                    ),
                )
            )
    return decisions


def _positive_unit_rejection(
    candidate: QuestionAnswerPair,
    units: dict[str, RetrievalUnit],
) -> str | None:
    """Return a reason when declared positive units are invalid."""
    unknown = sorted(set(candidate.positive_unit_ids) - units.keys())
    if unknown:
        return "unknown_positive_unit_ids: " + ", ".join(unknown)
    return None


def _judge_rejection_reason(
    query: QueryQualityEvaluation | None,
    grounding: QAEvaluation | None,
) -> str | None:
    """Return a stable reason for a missing or failed model judgement."""
    if query is None:
        return "missing_query_quality_evaluation"
    if not query.passes:
        return f"query_quality_rejected: {query.reason}"
    if grounding is None:
        return "missing_grounding_evaluation"
    if not grounding.passes_grounding:
        return f"grounding_rejected: {grounding.overall.assessment}"
    return None


def _modality_rejection_reason(
    evidence_modality: EvidenceModality,
    positive_units: tuple[RetrievalUnit, ...],
) -> str | None:
    """Reject a judge label that the positive-unit content cannot support."""
    if evidence_modality == "text_only" and any(not unit.text.strip() for unit in positive_units):
        return "text_only_modality_requires_text_in_every_positive"
    if evidence_modality == "image_grounded" and any(not unit.images for unit in positive_units):
        return "image_grounded_modality_requires_images_in_every_positive"
    return None


def _native_text_evidence_rejection(
    grounding: QAEvaluation | SourceAssessment,
    positive_units: tuple[RetrievalUnit, ...],
) -> str | None:
    """Verify quote provenance, not semantic entailment, against positive text."""
    units = {unit.unit_id: unit for unit in positive_units}
    covered = set()
    for evidence in grounding.native_text_evidence:
        unit = units.get(evidence.unit_id)
        if unit is None:
            return "native_text_evidence_not_from_positive_unit"
        quote = " ".join(evidence.quote.split())
        if not quote or quote not in " ".join(unit.text.split()):
            return "native_text_evidence_not_in_supplied_text"
        covered.add(unit.unit_id)
    if (
        grounding.evidence_modality == "text_only"
        and any(unit.images for unit in positive_units)
        and covered != units.keys()
    ):
        return "text_only_modality_requires_native_text_evidence_in_every_positive"
    return None


def _source_assessment_rejection(
    assessment: SourceAssessment | None,
    grounding: QAEvaluation,
    positive_units: tuple[RetrievalUnit, ...],
) -> str | None:
    """Prevent final verification from rescuing or changing a source reading."""
    if assessment is None:
        return "missing_source_assessment"
    if set(assessment.positive_unit_ids) != {unit.unit_id for unit in positive_units}:
        return "source_assessment_positive_units_mismatch"
    if not assessment.passes_source_assessment:
        return "source_assessment_rejected"
    quote_rejection = _native_text_evidence_rejection(assessment, positive_units)
    if quote_rejection is not None:
        return quote_rejection
    if (
        assessment.positive_source_relevant != grounding.positive_source_relevant
        or assessment.source_language_preserved != grounding.source_language_preserved
        or assessment.evidence_modality != grounding.evidence_modality
        or _native_quote_set(assessment) != _native_quote_set(grounding)
    ):
        return "source_assessment_judgement_mismatch"
    return None


def _native_quote_set(evaluation: QAEvaluation | SourceAssessment) -> set[tuple[str, str]]:
    """Compare source-bound quotes without making order or whitespace semantic."""
    return {(item.unit_id, " ".join(item.quote.split())) for item in evaluation.native_text_evidence}


def _quarantine_cross_unit_collisions(decisions: list[_CandidateDecision]) -> list[_CandidateDecision]:
    """Reject an exact normalized query mapped to different positive units."""
    groups: dict[str, list[_CandidateDecision]] = defaultdict(list)
    for decision in decisions:
        if decision.diagnostic.accepted:
            groups[normalize_query_text(decision.candidate.question)].append(decision)
    colliding = {
        key
        for key, values in groups.items()
        if len({tuple(sorted(value.candidate.positive_unit_ids)) for value in values}) > 1
    }
    if not colliding:
        return decisions
    result: list[_CandidateDecision] = []
    for decision in decisions:
        key = normalize_query_text(decision.candidate.question)
        if decision.diagnostic.accepted and key in colliding:
            result.append(
                replace(
                    decision,
                    diagnostic=decision.diagnostic.model_copy(
                        update={"accepted": False, "rejection_reason": "cross_unit_query_collision"}
                    ),
                )
            )
        else:
            result.append(decision)
    return result


def _accepted_candidates(decisions: list[_CandidateDecision]) -> list[_AcceptedCandidate]:
    """Build stable accepted query records."""
    accepted: list[_AcceptedCandidate] = []
    for decision in decisions:
        if not decision.diagnostic.accepted:
            continue
        if decision.evidence_modality is None:
            raise ValueError("accepted candidate is missing an evidence-modality judgement")
        query_key = ":".join(
            (
                decision.record.source_id,
                ",".join(sorted(decision.candidate.positive_unit_ids)),
                normalize_query_text(decision.candidate.question),
            )
        )
        accepted.append(
            _AcceptedCandidate(
                record=decision.record,
                candidate_index=decision.candidate_index,
                candidate=decision.candidate,
                positive_units=decision.positive_units,
                evidence_modality=decision.evidence_modality,
                query_id="q_" + hashlib.sha256(query_key.encode()).hexdigest()[:24],
            )
        )
    return accepted


def _coverage_warnings(
    records: list[GeneratedRetrievalRecord],
    accepted: list[_AcceptedCandidate],
) -> list[str]:
    """Return warning-only query-surface and visual-coverage diagnostics."""
    warnings: list[str] = []
    has_image_input = any(unit.images for record in records for unit in record.retrieval_units)
    visual_count = sum(item.evidence_modality == "image_grounded" for item in accepted)
    if has_image_input and visual_count == 0:
        warnings.append("NO_VISUAL_CANDIDATES: inspect source suitability and generation prompts")
    surface_count = len({item.candidate.query_surface for item in accepted})
    expected_surface_count = min(3, len(accepted))
    if surface_count < expected_surface_count:
        warnings.append(
            "QUERY_SURFACE_DIVERSITY: "
            f"accepted candidates cover {surface_count} of {expected_surface_count} expected surfaces"
        )
    return warnings


def _model_warnings(generator_model: str | None, judge_model: str | None) -> list[str]:
    """Report absent or non-independent model provenance."""
    if generator_model is None:
        return ["MODEL_PROVENANCE_NOT_RECORDED: pass generator_model and judge_model during export"]
    if generator_model == judge_model:
        return ["GENERATOR_AND_JUDGE_MODELS_MATCH: independent models are recommended"]
    return []


def _connected_document_splits(
    records: list[GeneratedRetrievalRecord],
    ratios: SplitRatios,
    seed: int,
) -> dict[str, SplitName]:
    """Keep every document linked by a generation row in the same split."""
    groups = _UnionFind()
    document_ids: set[str] = set()
    for record in records:
        record_document_ids = sorted({unit.document_id for unit in record.retrieval_units})
        document_ids.update(record_document_ids)
        for document_id in record_document_ids:
            groups.find(document_id)
        for document_id in record_document_ids[1:]:
            groups.union(record_document_ids[0], document_id)
    components = {document_id: groups.find(document_id) for document_id in document_ids}
    component_splits = assign_document_splits(components.values(), ratios=ratios, seed=seed)
    return {document_id: component_splits[component] for document_id, component in components.items()}


def _candidate_split(item: _AcceptedCandidate, assignments: dict[str, SplitName]) -> SplitName:
    """Return the common split for a candidate's positives."""
    splits = {assignments[unit.document_id] for unit in item.positive_units}
    if len(splits) != 1:
        raise ValueError("candidate positive documents cross split boundaries")
    return next(iter(splits))


def _split_warnings(
    assignments: dict[str, SplitName],
    candidate_counts: Counter[SplitName],
    ratios: SplitRatios,
) -> list[str]:
    """Report requested splits without documents or accepted queries."""
    document_counts = Counter(assignments.values())
    values = {"train": ratios.train, "validation": ratios.validation, "evaluation": ratios.evaluation}
    warnings: list[str] = []
    for name in _SPLIT_NAMES:
        if values[name] <= 0:
            continue
        if document_counts[name] == 0:
            warnings.append(f"EMPTY_{name.upper()}_SPLIT: add source documents or adjust split ratios")
        elif candidate_counts[name] == 0:
            warnings.append(f"EMPTY_{name.upper()}_QUERIES: no accepted queries for assigned documents")
    return warnings


def _require_empty_output(output_dir: Path) -> None:
    """Protect an existing output directory from replacement."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must be new or empty")


def _materialize_assets(
    records: list[GeneratedRetrievalRecord],
    input_root: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Preflight and copy optional images into the portable bundle."""
    image_units = [unit for record in records for unit in record.retrieval_units if unit.images]
    sources = {unit.unit_id: _unit_image_source(unit, input_root) for unit in image_units}
    asset_dir = output_dir / "assets" / "units"
    asset_dir.mkdir(parents=True)
    assets: dict[str, str] = {}
    for unit in image_units:
        source = sources[unit.unit_id]
        filename = hashlib.sha256(unit.unit_id.encode()).hexdigest()[:24] + source.suffix.casefold()
        relative = Path("assets") / "units" / filename
        shutil.copyfile(source, output_dir / relative)
        assets[unit.unit_id] = relative.as_posix()
    return assets


def _unit_image_source(unit: RetrievalUnit, input_root: Path) -> Path:
    """Resolve and validate the single page image supported by one unit."""
    if len(unit.images) != 1:
        raise ValueError(f"retrieval unit {unit.unit_id!r} must declare exactly one page image")
    declared = Path(unit.images[0])
    source = (declared if declared.is_absolute() else input_root / declared).resolve()
    if not source.is_file():
        raise ValueError(f"retrieval-unit image does not exist: {unit.images[0]}")
    if source.suffix.casefold() not in _IMAGE_SUFFIXES:
        raise ValueError(f"unsupported retrieval-unit image format: {source.suffix}")
    return source


def _write_audit_files(
    output_dir: Path,
    records: list[GeneratedRetrievalRecord],
    decisions: list[_CandidateDecision],
    assignments: dict[str, SplitName],
    assets: dict[str, str],
) -> list[Path]:
    """Write portable source-unit, split, and decision audit files."""
    records_path = output_dir / "source_records.jsonl"
    _write_jsonl(
        records_path,
        (_portable_record(record) for record in records),
    )
    units_path = output_dir / "retrieval_units.jsonl"
    _write_jsonl(
        units_path,
        (
            _portable_unit(unit, assignments[unit.document_id], assets)
            for record in records
            for unit in record.retrieval_units
        ),
    )
    diagnostics_path = output_dir / "candidate_diagnostics.jsonl"
    diagnostics = [decision.diagnostic.model_dump(mode="json") for decision in decisions]
    diagnostics.extend(
        {"source_id": record.source_id, "stage": "pre_grounding", "accepted": False, **item.model_dump(mode="json")}
        for record in records
        for item in record.generation_diagnostics
    )
    _write_jsonl(diagnostics_path, diagnostics)
    split_path = output_dir / "split_manifest.json"
    _write_json(split_path, {"schema_version": 1, "assignments": assignments, "document_disjoint": True})
    return [
        records_path,
        units_path,
        diagnostics_path,
        split_path,
        *[output_dir / path for path in assets.values()],
    ]


def _portable_record(record: GeneratedRetrievalRecord) -> dict[str, Any]:
    """Serialize generated candidates and judgements without source asset paths."""
    row = record.model_dump(mode="json", exclude={"retrieval_units"})
    row["retrieval_unit_ids"] = [unit.unit_id for unit in record.retrieval_units]
    return row


def _portable_unit(
    unit: RetrievalUnit,
    split: SplitName,
    assets: dict[str, str],
) -> dict[str, Any]:
    """Serialize a unit using only bundle-relative image paths."""
    row = unit.model_dump(mode="json")
    row["images"] = [assets[unit.unit_id]] if unit.unit_id in assets else []
    row["split"] = split
    return row


def _write_views(
    output_dir: Path,
    records: list[GeneratedRetrievalRecord],
    accepted: list[_AcceptedCandidate],
    assignments: dict[str, SplitName],
    assets: dict[str, str],
    dataset_id: str,
) -> list[Path]:
    """Write independently sampleable training and evaluation views."""
    units = [unit for record in records for unit in record.retrieval_units]
    paths: list[Path] = []
    for view in _VIEW_NAMES:
        for split in _SPLIT_NAMES:
            split_units = [
                unit for unit in units if assignments[unit.document_id] == split and _unit_supports_view(unit, view)
            ]
            corpus_dir = output_dir / "views" / view / "corpus" / split
            corpus_dir.mkdir(parents=True, exist_ok=True)
            corpus_path = corpus_dir / "part-00000.parquet"
            _write_corpus(corpus_path, view, split_units, assets, output_dir)
            metadata_path = corpus_dir / "merlin_metadata.json"
            _write_json(metadata_path, _corpus_metadata(view, dataset_id))
            paths.extend([corpus_path, metadata_path])
        for split in ("train", "validation"):
            data = [
                _training_record(item, dataset_id)
                for item in accepted
                if _candidate_split(item, assignments) == split and _candidate_supports_view(item, view)
            ]
            path = output_dir / "views" / view / f"{split}.json"
            _write_json(path, {"corpus": {"path": f"corpus/{split}"}, "data": data})
            paths.append(path)
        paths.extend(_write_evaluation_view(output_dir, view, units, accepted, assignments, assets, dataset_id))
    return paths


def _unit_supports_view(unit: RetrievalUnit, view: ViewName) -> bool:
    """Return whether a retrieval unit can be represented in a view."""
    if view == "text":
        return bool(unit.text.strip())
    return bool(unit.images)


def _candidate_supports_view(item: _AcceptedCandidate, view: ViewName) -> bool:
    """Return whether all positives can be represented in a requested view."""
    if item.evidence_modality == "image_grounded" and view == "text":
        return False
    return all(_unit_supports_view(unit, view) for unit in item.positive_units)


def _corpus_schema(view: ViewName) -> pa.Schema:
    """Return the current recipe corpus schema for a view."""
    if view == "text":
        return pa.schema([("id", pa.string()), ("text", pa.string())])
    if view == "image":
        return pa.schema([("image_filename", pa.string()), ("image", pa.binary())])
    return pa.schema([("docid", pa.string()), ("text", pa.string()), ("image", pa.binary())])


def _write_corpus(
    path: Path,
    view: ViewName,
    units: list[RetrievalUnit],
    assets: dict[str, str],
    output_dir: Path,
) -> None:
    """Write one split corpus with a view-specific schema."""
    rows: list[dict[str, Any]] = []
    for unit in units:
        if view == "text":
            rows.append({"id": unit.unit_id, "text": unit.text})
            continue
        image = (output_dir / assets[unit.unit_id]).read_bytes()
        if view == "image":
            rows.append({"image_filename": unit.unit_id, "image": image})
        else:
            rows.append({"docid": unit.unit_id, "text": unit.text, "image": image})
    pq.write_table(pa.Table.from_pylist(rows, schema=_corpus_schema(view)), path)


def _corpus_metadata(view: ViewName, dataset_id: str) -> dict[str, Any]:
    """Return AutoModel-compatible corpus metadata."""
    classes = {"text": "TextQADataset", "image": "ColPaliDataset", "image_and_text": "WikiSSNQDataset"}
    return {
        "class": classes[view],
        "schema_version": 1,
        "view": view,
        "format": "parquet",
        "corpus_id": dataset_id,
    }


def _training_record(item: _AcceptedCandidate, dataset_id: str) -> dict[str, Any]:
    """Serialize one positive-only training query."""
    document_ids = sorted({unit.document_id for unit in item.positive_units})
    return {
        "question_id": item.query_id,
        "question": item.candidate.question,
        "corpus_id": dataset_id,
        "pos_doc": [{"id": unit.unit_id} for unit in item.positive_units],
        "neg_doc": [],
        "negative_scores": [],
        "answer": item.candidate.answer,
        "evidence": item.candidate.evidence,
        "evidence_modality": item.evidence_modality,
        "query_surface": item.candidate.query_surface,
        "language": item.record.language,
        "source_document_ids": document_ids,
        "positive_unit_ids": [unit.unit_id for unit in item.positive_units],
    }


def _write_evaluation_view(
    output_dir: Path,
    view: ViewName,
    units: list[RetrievalUnit],
    accepted: list[_AcceptedCandidate],
    assignments: dict[str, SplitName],
    assets: dict[str, str],
    dataset_id: str,
) -> list[Path]:
    """Write BEIR-style synthetic evaluation files for one view."""
    root = output_dir / "synthetic_eval" / view
    qrels_dir = root / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    evaluation_units = [
        unit for unit in units if assignments[unit.document_id] == "evaluation" and _unit_supports_view(unit, view)
    ]
    candidates = [
        item
        for item in accepted
        if _candidate_split(item, assignments) == "evaluation" and _candidate_supports_view(item, view)
    ]
    queries_path = root / "queries.jsonl"
    _write_jsonl(
        queries_path,
        (
            {
                "_id": item.query_id,
                "text": item.candidate.question,
                "metadata": {
                    "corpus_id": dataset_id,
                    "source_document_ids": sorted({unit.document_id for unit in item.positive_units}),
                    "positive_unit_ids": [unit.unit_id for unit in item.positive_units],
                    "language": item.record.language,
                    "query_surface": item.candidate.query_surface,
                    "evidence_modality": item.evidence_modality,
                },
            }
            for item in candidates
        ),
    )
    corpus_path = root / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        (_evaluation_document(unit, view, assets, dataset_id) for unit in evaluation_units),
    )
    qrels_path = qrels_dir / "test.tsv"
    _write_qrels(
        qrels_path,
        ((item.query_id, unit.unit_id, 1) for item in candidates for unit in item.positive_units),
    )
    return [queries_path, corpus_path, qrels_path]


def _evaluation_document(
    unit: RetrievalUnit,
    view: ViewName,
    assets: dict[str, str],
    dataset_id: str,
) -> dict[str, Any]:
    """Serialize one view-specific evaluation document."""
    title = unit.document_id
    if unit.page_number is not None:
        title += f" page {unit.page_number}"
    document: dict[str, Any] = {
        "_id": unit.unit_id,
        "title": title,
        "metadata": {
            "corpus_id": dataset_id,
            "source_document_id": unit.document_id,
            "page_number": unit.page_number,
            "source_uri": unit.source_uri,
        },
    }
    if view in {"text", "image_and_text"}:
        document["text"] = unit.text
    if view in {"image", "image_and_text"}:
        document["image_path"] = assets[unit.unit_id]
    return document


def _write_qrels(path: Path, rows: Iterable[tuple[str, str, int]]) -> None:
    """Write BEIR qrels with the expected header."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write deterministic UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _artifact_record(path: Path, output_dir: Path) -> dict[str, Any]:
    """Return a compact checksummed artifact record."""
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
