# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer stages with deterministic, versioned dataset boundaries."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import data_designer.config as dd
import pyarrow as pa
import pyarrow.parquet as pq
from data_designer.config.config_builder import BuilderConfig
from data_designer.config.seed import SeedConfig
from data_designer.config.utils.media_helpers import detect_image_format, is_image_path
from data_designer.config.version import get_library_version
from data_designer.engine.secret_resolver import EnvironmentResolver
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface import DataDesigner
from data_designer.interface.composite_workflow import CompositeWorkflow
from data_designer.interface.workflow_metadata import WorkflowMetadata
from pydantic import BaseModel

from data_designer_retrieval_sdg.multimodal.bounds import check_request
from data_designer_retrieval_sdg.multimodal.config import RetrievalStructuredColumnConfig
from data_designer_retrieval_sdg.multimodal.inference import Request, model_configuration
from data_designer_retrieval_sdg.multimodal.models import (
    Candidate,
    FailedGeneration,
    GenerationContext,
    JudgeStage,
    ModelRole,
    MultimodalSDGConfig,
)
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource
from data_designer_retrieval_sdg.multimodal.storage import artifact, digest, fingerprint, write_bytes, write_json
from data_designer_retrieval_sdg.multimodal.structured import FAILURE_KEY


@dataclass(frozen=True)
class ModelColumn:
    """One Data Designer structured column and its optional upstream quality gate.

    Attributes:
        name: Response column name within the stage.
        schema: Structured response schema.
        role: Generator or fallback judge role.
        images: Whether the column receives image attachments.
        skip: Optional expression depending on upstream quality judgments.
        judge_stage: Optional judge override shared with request-budget validation.
    """

    name: str
    schema: type[BaseModel]
    role: ModelRole
    images: bool = False
    skip: str | None = None
    judge_stage: JudgeStage | None = None


@dataclass(frozen=True)
class Stage:
    """Model stage with deterministic input preparation and output consumption.

    Attributes:
        name: Stable name in Data Designer workflow metadata.
        columns: Data Designer structured columns, including their skip dependencies.
        prepare: Convert boundary state and sources into named requests. May update
            state to record exact source/slot memberships for consumption.
        consume: Apply validated responses to boundary state; never call an LLM.
    """

    name: str
    columns: tuple[ModelColumn, ...]
    prepare: Callable
    consume: Callable


def model_configurations(config: MultimodalSDGConfig) -> tuple[list[dd.ModelConfig], list[dd.ModelProvider]]:
    """Use the checked deployment settings for each native workflow role.

    Args:
        config: Generator and active judge settings with request concurrency.

    Returns:
        Native models and providers sharing stage aliases, with repeated readiness
        probes disabled after the workflow-level check.
    """
    models, providers = [], []
    for role, settings in {"generator": config.generator, **config.active_judges()}.items():
        model, provider = model_configuration(settings, config.concurrency, skip_health_check=True)
        model.alias = model.provider = provider.name = role
        models.append(model)
        providers.append(provider)
    return models, providers


def stage_builder(stage: Stage, config: MultimodalSDGConfig) -> dd.DataDesignerConfigBuilder:
    """Expose all model columns, dependencies, and skips to Data Designer."""
    builder = dd.DataDesignerConfigBuilder(model_configs=model_configurations(config)[0])
    for column in stage.columns:
        skip = column.skip or "{{ _skip }}"
        if config.failed_record_policy == "skip" and column.skip:
            guards = [
                f"{parent.name}.{FAILURE_KEY} is defined"
                for parent in stage.columns
                if parent.name + "." in column.skip
            ]
            skip = "{{ " + " or ".join([*guards, "(" + skip.removeprefix("{{").removesuffix("}}") + ")"]) + " }}"
        builder.add_column(
            RetrievalStructuredColumnConfig(
                name=column.name,
                model_alias=column.judge_stage or column.role,
                prompt="{{ " + column.name + "_prompt }}",
                output_format=column.schema,
                multi_modal_context=[dd.ImageContext(column_name=column.name + "_images")] if column.images else None,
                skip=dd.SkipConfig(when=skip),
                propagate_skip=False,
                failed_record_policy=config.failed_record_policy,
            )
        )
    return builder


def write_seed(path: Path, requests: list[dict[str, Request]], stage: Stage, config: MultimodalSDGConfig) -> None:
    """Write bounded prompt batches and image references; empty stages perform no calls.

    A skipped control row carries an empty boundary through the linear workflow.
    It never becomes a candidate or exported source.
    """
    write_parquet(path, seed_tables(path, requests, stage, config))


def seed_tables(
    path: Path, requests: list[dict[str, Request]], stage: Stage, config: MultimodalSDGConfig
) -> Iterator[pa.Table]:
    """Yield bounded seed batches containing Data Designer image references, never encoded pixels.

    Args:
        path: Seed destination, also owning any normalized image aliases.
        requests: Ordered stage requests, or an empty list for the skipped control row.
        stage: Data Designer response columns.
        config: Request validation and batch-size limits.

    Yields:
        Tables with a stable schema, including empty image lists and nullable metadata.
    """
    fields = [
        ("_row_id", pa.int64()),
        ("_skip", pa.bool_()),
        ("context_id", pa.string()),
        ("source_unit_ids", pa.list_(pa.string())),
        ("image_unit_ids", pa.list_(pa.string())),
        ("query", pa.string()),
        ("requested_instructions", pa.string()),
    ]
    for column in stage.columns:
        fields.extend([(column.name + "_prompt", pa.string()), (column.name + "_images", pa.list_(pa.string()))])
    schema = pa.schema(fields)
    rows, image_paths = [], {}
    for index, group in enumerate(requests or [{}]):
        source_request = next((request for request in group.values() if request.context is not None), None)
        row = {
            "_row_id": index,
            "_skip": not bool(group),
            "context_id": source_request.context.context_id if source_request else None,
            "source_unit_ids": source_request.context.unit_ids if source_request else [],
            "image_unit_ids": list(source_request.image_unit_ids) if source_request else [],
            "query": source_request.query if source_request else None,
            "requested_instructions": json.dumps(source_request.instructions) if source_request else "[]",
        }
        for column in stage.columns:
            request = group.get(column.name)
            if request is not None:
                check_request(request, config)
            row[column.name + "_prompt"] = request.text if request else ""
            images = []
            for image in request.images if request else ():
                if image not in image_paths:
                    image_paths[image] = image_reference_path(image, path.parent / "images")
                images.append(image_paths[image])
            row[column.name + "_images"] = images
        rows.append(row)
        if len(rows) == config.batch_size:
            yield pa.Table.from_pylist(rows, schema=schema)
            rows = []
    if rows:
        yield pa.Table.from_pylist(rows, schema=schema)


def image_reference_path(image: Path, aliases: Path) -> str:
    """Resolve a Data Designer image path, preserving arbitrary input suffixes and exact bytes.

    Args:
        image: Original frozen image attachment.
        aliases: Run-owned directory for byte-identical aliases when Data Designer detection needs a suffix.

    Returns:
        An absolute path whose extension Data Designer recognizes as an image reference.
    """
    if is_image_path(str(image)):
        return str(image.resolve())
    content = image.read_bytes()
    alias = aliases / f"{digest(image)}.{detect_image_format(content).value}"
    write_bytes(alias, content)
    return str(alias.resolve())


def response_files(path: Path) -> list[Path]:
    """List original and separately retained recovery responses in stable order.

    Args:
        path: Original Data Designer stage directory containing batches and recovery attempts.

    Returns:
        Original response paths followed by recovery response paths, with each
        group sorted by path. Returns an empty list when no responses exist.
    """
    return [
        *sorted((path / "parquet-files").glob("*.parquet")),
        *sorted((path / "recovery").glob("attempt-*/responses/parquet-files/*.parquet")),
    ]


def stage_responses(
    path: Path, seed: Path, stage: Stage, *, allow_failures: bool = False
) -> list[dict[str, BaseModel | None] | None]:
    """Reject missing/duplicate Data Designer rows before a stage can be marked complete."""
    expected_ids = [
        identifier
        for batch in pq.ParquetFile(seed).iter_batches(columns=["_row_id"], batch_size=1024)
        for identifier in batch.column(0).to_pylist()
    ]
    columns = ["_row_id", "_skip", *(column.name for column in stage.columns)]
    rows = [
        row
        for file in response_files(path)
        for batch in pq.ParquetFile(file).iter_batches(columns=columns, batch_size=32)
        for row in batch.to_pylist()
    ]
    if sorted(row["_row_id"] for row in rows) != expected_ids:
        raise RuntimeError("Data Designer stage has missing or duplicate responses; inspect its errors before resuming")
    results = []
    for row in sorted(rows, key=lambda item: item["_row_id"]):
        if row["_skip"]:
            continue
        results.append(parse_stage_response(row, stage, allow_failures=allow_failures))
    return results


def parse_stage_response(
    row: dict, stage: Stage, *, allow_failures: bool = False
) -> dict[str, BaseModel | None] | None:
    """Validate saved model values without requiring a complete stage.

    Args:
        row: Saved Data Designer row containing a value for every declared response column.
        stage: Response column names and schemas used to validate the saved values.
        allow_failures: Interpret tagged exhausted errors only under explicit opt-in.

    Returns:
        Validated models keyed by column name, preserving null values as None.

    Raises:
        KeyError: A declared response column is absent from the row.
        pydantic.ValidationError: A non-null value does not satisfy its response schema.
    """
    responses, failed = {}, False
    for column in stage.columns:
        value = row[column.name]
        decoded = json.loads(value) if isinstance(value, str) else value
        if isinstance(decoded, dict) and FAILURE_KEY in decoded:
            if not allow_failures or set(decoded) != {FAILURE_KEY}:
                raise ValueError("Saved failed response requires the explicit skip policy and a valid diagnostic")
            FailedGeneration.model_validate(decoded[FAILURE_KEY])
            failed = True
            continue
        responses[column.name] = (
            None
            if value is None
            else column.schema.model_validate_json(value)
            if isinstance(value, str)
            else column.schema.model_validate(value)
        )
    return None if failed else responses


def stage_failures(path: Path, seed: Path, stage: Stage) -> list[dict]:
    """Collect exhausted errors with original row and request provenance.

    Args:
        path: Data Designer stage with original and recovery batches.
        seed: Immutable ordered requests and source/query identities.
        stage: Declared Data Designer columns.

    Returns:
        Diagnostics referring to preserved raw response artifacts, never gate labels.
    """
    failures = []
    for file in response_files(path):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                columns = {}
                for column in stage.columns:
                    value = row[column.name]
                    value = json.loads(value) if isinstance(value, str) else value
                    if isinstance(value, dict) and FAILURE_KEY in value:
                        columns[column.name] = FailedGeneration.model_validate(value[FAILURE_KEY]).model_dump()
                if columns:
                    request = pq.read_table(seed, filters=[("_row_id", "=", row["_row_id"])]).to_pylist()[0]
                    failures.append(
                        {
                            "stage": stage.name,
                            "row_id": row["_row_id"],
                            "reason": "model_validation_exhausted",
                            "context_id": request["context_id"],
                            "source_unit_ids": request["source_unit_ids"],
                            "query": request["query"],
                            "requested_instructions": json.loads(request["requested_instructions"]),
                            "columns": columns,
                            "observed_responses": {
                                column.name: (
                                    json.loads(row[column.name])
                                    if isinstance(row[column.name], str)
                                    else row[column.name]
                                )
                                for column in stage.columns
                            },
                            "response_artifact": str(file.relative_to(path)),
                            "response_sha256": digest(file),
                        }
                    )
    return sorted(failures, key=lambda row: row["row_id"])


def recovery_state(
    path: Path,
    seed: Path,
    stage: Stage,
    config: MultimodalSDGConfig,
) -> tuple[list[int], list[Path]]:
    """Validate retained evidence and identify holes in finalized original groups.

    Args:
        path: Original Data Designer stage directory containing batches and recovery attempts.
        seed: Immutable original seed with ordered row identities and control flags.
        stage: Response columns and schemas used to validate retained model outputs.
        config: Frozen workflow settings supplying the resolved native row-group size.

    Returns:
        A pair containing sorted missing row identities from finalized original
        batches and all prior recovery attempt directories sorted by path,
        including interrupted attempts. Wholly absent original batches are
        excluded from the missing identities and remain owned by Data Designer resume.

    Raises:
        ValueError: Saved identities, recovery seeds, control flags or response
            values are inconsistent or invalid.
    """
    original = pq.read_table(seed, columns=["_row_id", "_skip"])
    expected = original.column("_row_id").to_pylist()
    if expected != list(range(len(expected))):
        raise ValueError("Data Designer recovery requires the original ordered row identities")
    seen: set[int] = set()
    finalized: set[int] = set()
    buffer_size = config.resolved_run_config().buffer_size
    for batch in sorted((path / "parquet-files").glob("*.parquet")):
        group = int(batch.stem.removeprefix("batch_"))
        allowed = set(range(group * buffer_size, min((group + 1) * buffer_size, len(expected))))
        ids = pq.read_table(batch, columns=["_row_id"]).column("_row_id").to_pylist()
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= allowed or seen.intersection(ids):
            raise ValueError("Data Designer recovery found invalid or duplicate original row identities")
        seen.update(ids)
        finalized.update(allowed)
    attempts = sorted((path / "recovery").glob("attempt-*"))
    for attempt in attempts:
        if not (attempt / "seed.parquet").is_file():
            if any((attempt / "responses/parquet-files").glob("*.parquet")):
                raise ValueError("Data Designer recovery responses have no original attempt seed")
            # An interruption before atomic seed publication made no durable
            # responses. Preserve the abandoned directory and use a new attempt.
            continue
        attempt_seed = pq.read_table(attempt / "seed.parquet")
        ids = attempt_seed.column("_row_id").to_pylist()
        selected = pq.read_table(seed, filters=[("_row_id", "in", ids)])
        if not ids or not set(ids) <= finalized or not selected.equals(attempt_seed):
            raise ValueError("Data Designer recovery seed differs from its immutable original rows")
        for batch in sorted((attempt / "responses/parquet-files").glob("*.parquet")):
            recovered = pq.read_table(batch, columns=["_row_id"]).column("_row_id").to_pylist()
            if len(recovered) != len(set(recovered)) or not set(recovered) <= set(ids) or seen.intersection(recovered):
                raise ValueError("Data Designer recovery found invalid or duplicate recovered row identities")
            seen.update(recovered)
    columns = ["_row_id", "_skip", *(column.name for column in stage.columns)]
    expected_skip = original.column("_skip").to_pylist()
    for file in response_files(path):
        for batch in pq.ParquetFile(file).iter_batches(columns=columns, batch_size=32):
            for row in batch.to_pylist():
                if row["_skip"] != expected_skip[row["_row_id"]]:
                    raise ValueError("Data Designer recovery control row differs from its original seed")
                parse_stage_response(row, stage, allow_failures=config.failed_record_policy == "skip")
    return sorted(finalized - seen), attempts


def recover_missing_rows(
    path: Path,
    seed: Path,
    stage: Stage,
    config: MultimodalSDGConfig,
    designer: DataDesigner,
) -> None:
    """Retry holes in finalized groups once during an explicitly resumed stage.

    Whole absent groups remain owned by Data Designer resume. Recovery runs use the
    same Data Designer columns and retry limits, with only missing original row
    identities as seeds. Every attempt has its own directory; original batches
    and earlier attempts are never rewritten or resumed.

    Args:
        path: Original Data Designer stage directory.
        seed: Immutable original stage seed.
        stage: Unchanged model columns and their dependencies.
        config: Frozen workflow configuration with explicit resume enabled.
        designer: Data Designer runtime owning inference, traces and bounded retries.

    Raises:
        ValueError: Saved identities, schemas or recovery seeds are inconsistent.
    """
    missing, attempts = recovery_state(path, seed, stage, config)
    if not missing:
        return
    number = max((int(attempt.name.removeprefix("attempt-")) for attempt in attempts), default=-1) + 1
    attempt = path / "recovery" / f"attempt-{number:05d}"
    selection = pq.read_table(seed, filters=[("_row_id", "in", missing)])
    write_parquet(attempt / "seed.parquet", selection)
    builder = stage_builder(stage, config).with_seed_dataset(dd.LocalFileSeedSource(path=str(attempt / "seed.parquet")))
    designer.create(builder, num_records=len(missing), dataset_name="responses", artifact_path=attempt)


def finish_stage(
    path: Path,
    *,
    stage: Stage,
    next_stage: Stage | None,
    input_state: Path,
    input_seed: Path,
    config: MultimodalSDGConfig,
    sources: list[RetrievalSource],
    designer: DataDesigner,
) -> Path:
    """Finish a Data Designer stage, repairing finalized-group holes on explicit resume.

    State is a versioned boundary artifact, not an independent response cache.
    Every callback reads its predecessor's artifact so Data Designer resume need not
    replay completed callbacks to rebuild Python object state.
    """
    state = json.loads(input_state.read_text())
    saved_rows = sum(pq.read_metadata(file).num_rows for file in response_files(path))
    if config.resume and saved_rows < pq.read_metadata(input_seed).num_rows:
        recover_missing_rows(path, input_seed, stage, config, designer)
    responses = stage_responses(path, input_seed, stage, allow_failures=config.failed_record_policy == "skip")
    failures = stage_failures(path, input_seed, stage)
    for failure in failures:
        if stage.name == "toc-confirmation":
            failure["source_unit_ids"] = [state["toc_ids"][failure["row_id"]]]
        elif stage.name == "summary-combinations":
            members = [state["outcomes"][i]["context"] for i in state["combinations"][failure["row_id"]]]
            failure["source_unit_ids"] = list(dict.fromkeys(key for member in members for key in member["unit_ids"]))
            failure["summary_context_ids"] = [member["context_id"] for member in members]
        elif stage.name == "query-judgments":
            record = state["candidate_slots"][failure["row_id"]]
            outcome, slot = state["outcomes"][record["outcome_index"]], record["slot"]
            context = outcome["context"]
            failure.update(
                query_id="q_" + fingerprint([context["context_id"], slot["slot"], slot["query"]])[:32],
                slot=slot["slot"],
                requested_instruction=outcome["instructions"][slot["slot"]],
                language=context["language"],
            )
    if failures:
        state.setdefault("skipped_records", []).extend(failures)
        write_json(path / "boundary/skipped_records.json", failures)
    stage.consume(state, responses, sources, config)
    requests = next_stage.prepare(state, sources, config) if next_stage else []
    write_json(path / "boundary/state.json", state)
    output = path / "boundary/seed.parquet"
    if next_stage:
        write_seed(output, requests, next_stage, config)
    else:
        write_parquet(output, pa.table({"completed": [True]}))
    write_json(
        path / "boundary/integrity.json",
        [artifact(file, path) for file in boundary_artifacts(path)],
    )
    return output


def build_workflow(
    config: MultimodalSDGConfig,
    sources: list[RetrievalSource],
    contexts: list[GenerationContext],
    *,
    artifact_path: Path,
) -> CompositeWorkflow:
    """Build the complete Data Designer graph before execution, including optional stages.

    Args:
        config: Model roles, request bounds, and workflow options.
        sources: Immutable original sources and frozen image paths.
        contexts: Validated initial context memberships.
        artifact_path: Run-owned Data Designer artifact directory.

    Returns:
        A Data Designer CompositeWorkflow whose final boundary contains candidates and
        context outcomes in its state artifact.
    """
    from data_designer_retrieval_sdg.multimodal.stages import workflow_stages
    from data_designer_retrieval_sdg.multimodal.workflow import run_identity

    artifact_path = artifact_path.resolve()
    identity_data = run_identity(config, sources, contexts)
    write_json(artifact_path / "identity.json", identity_data)
    verify_boundaries(artifact_path)
    designer = DataDesigner(
        artifact_path=artifact_path,
        secret_resolver=EnvironmentResolver(),
        auto_configure_logging=False,
        model_providers=model_configurations(config)[1],
    )
    run_config = config.resolved_run_config()
    designer.set_run_config(run_config)
    stages = workflow_stages(config)
    state = {"contexts": [context.model_dump() for context in contexts], "outcomes": []}
    requests = stages[0].prepare(state, sources, config)
    identity = fingerprint(identity_data)
    input_state = artifact_path / "inputs" / identity / "state.json"
    input_seed = input_state.with_name("seed.parquet")
    write_json(input_state, state)
    write_seed(input_seed, requests, stages[0], config)
    write_json(
        input_state.with_name("integrity.json"),
        [artifact(file, input_state.parent) for file in seed_artifacts(input_state.parent)],
    )
    workflow = designer.compose_workflow(name="multimodal")
    metadata = verify_stage_metadata(artifact_path)
    if metadata is not None and len(metadata.stages) > len(stages):
        raise ValueError("Data Designer resume metadata has unexpected stages; existing output is preserved")
    upstream_fingerprint = None
    for index, stage in enumerate(stages):
        stage_path = artifact_path / "multimodal" / f"stage-{index}-{stage.name}"
        if config.resume and (stage_path / "recovery").exists():
            recovery_state(stage_path, input_seed, stage, config)
        builder = stage_builder(stage, config)
        if metadata is not None and index < len(metadata.stages):
            upstream_fingerprint = verify_resume_stage(
                metadata,
                index,
                stage,
                builder,
                input_seed,
                identity,
                upstream_fingerprint,
                run_config.buffer_size,
                artifact_path / "multimodal",
            )
        if index == 0:
            builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(input_seed)))
        workflow.add_stage(
            stage.name,
            builder,
            num_records=max(1, len(requests)) if index == 0 else None,
            on_success=partial(
                finish_stage,
                stage=stage,
                next_stage=stages[index + 1] if index + 1 < len(stages) else None,
                input_state=input_state,
                input_seed=input_seed,
                config=config,
                sources=sources,
                designer=designer,
            ),
            on_success_version=identity,
        )
        input_state, input_seed = stage_path / "boundary/state.json", stage_path / "boundary/seed.parquet"
    return workflow


def generate_candidates(
    config: MultimodalSDGConfig, sources: list[RetrievalSource], contexts: list[GenerationContext]
) -> tuple[list[Candidate], list[dict]]:
    """Run the workflow and return candidates and context outcomes.

    Args:
        config: Validated workflow settings and explicit resume policy.
        sources: Complete original corpus with frozen image paths.
        contexts: Validated initial context memberships.

    Returns:
        Candidates, including rejections, and complete context/slot outcomes.
    """

    workflow = build_workflow(config, sources, contexts, artifact_path=config.output_dir / "data_designer")
    result = workflow.run(resume=ResumeMode.IF_POSSIBLE if config.resume else ResumeMode.NEVER)
    state = json.loads(result.get_stage_output_path(result.final_stage_name).with_name("state.json").read_text())
    if config.failed_record_policy == "skip":
        write_json(
            config.output_dir / "generation_failures.json",
            {
                "policy": "skip",
                "records": state.get("skipped_records", []),
            },
        )
    return [Candidate.model_validate(value) for value in state["candidates"]], state["outcomes"]


def write_parquet(path: Path, table: pa.Table | Iterable[pa.Table]) -> None:
    """Stream boundary tables to an immutable, atomically published local file.

    Args:
        path: Destination, which may already contain only identical serialized bytes.
        table: A table or a nonempty stream of tables with the same schema.

    Raises:
        ValueError: Existing bytes differ or the table stream is empty.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = iter([table] if isinstance(table, pa.Table) else table)
    first = next(tables, None)
    if first is None:
        raise ValueError("Parquet publication requires at least one table")
    with tempfile.TemporaryDirectory(prefix=".sdg-", dir=path.parent) as directory:
        temporary = Path(directory) / "content"
        with pq.ParquetWriter(temporary, first.schema) as writer:
            writer.write_table(first)
            for batch in tables:
                writer.write_table(batch)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if digest(path) != digest(temporary):
                raise ValueError(f"Artifact changed; use a new run directory: {path}") from None


def seed_artifacts(path: Path) -> list[Path]:
    """List the immutable state, seed and image aliases passed into a stage.

    Args:
        path: Initial input or stage boundary directory.

    Returns:
        Required state and seed paths, plus any generated image aliases.
    """
    return [
        path / "state.json",
        path / "seed.parquet",
        *sorted(file for file in (path / "images").glob("*") if file.is_file()),
    ]


def boundary_artifacts(path: Path) -> list[Path]:
    """List the complete durable evidence covered by a stage boundary checksum.

    Args:
        path: Data Designer stage directory.

    Returns:
        State, successor seed, diagnostics, image aliases, responses and recovery seeds.
    """
    return [
        *seed_artifacts(path / "boundary"),
        *sorted((path / "boundary").glob("skipped_records.json")),
        *response_files(path),
        *sorted((path / "recovery").glob("attempt-*/seed.parquet")),
    ]


def verify_artifact_manifest(manifest: Path, root: Path, files: list[Path]) -> None:
    """Verify an immutable inventory and every file it covers.

    Args:
        manifest: Saved inventory of paths, sizes and SHA-256 checksums.
        root: Base directory for inventory paths.
        files: Complete expected artifact set at this boundary.

    Raises:
        ValueError: Inventory or evidence is missing, malformed or changed.
    """
    try:
        records = json.loads(manifest.read_text())
        expected = {file.relative_to(root).as_posix() for file in files}
        if not isinstance(records, list) or len(records) != len(expected):
            raise ValueError("Incomplete integrity manifest")
        if {record["path"] for record in records} != expected:
            raise ValueError("Integrity manifest does not cover the stage evidence")
        for record in records:
            path = root / record["path"]
            if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("Stage evidence changed")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Data Designer boundary integrity failure: {root}") from exc


def verify_boundaries(artifact_path: Path, *, require_complete: bool = False) -> None:
    """Verify initial inputs and stage evidence before checkpoint or completed-bundle reuse.

    Args:
        artifact_path: Run-owned Data Designer artifact directory.
        require_complete: Require the final query-judgment boundary for a completed run.

    Raises:
        ValueError: Required workflow evidence or a boundary checksum is missing or changed.
    """
    try:
        identity = fingerprint(json.loads((artifact_path / "identity.json").read_text()))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("Data Designer input integrity failure: missing or invalid identity") from exc
    inputs = artifact_path / "inputs" / identity
    input_manifest = inputs / "integrity.json"
    if require_complete or input_manifest.exists() or (artifact_path / "multimodal/workflow-metadata.json").exists():
        verify_artifact_manifest(input_manifest, inputs, seed_artifacts(inputs))
    metadata = verify_stage_metadata(artifact_path)
    if require_complete and (
        metadata is None
        or not metadata.stages
        or metadata.stages[-1].name != "query-judgments"
        or any(stage.status != "completed" for stage in metadata.stages)
    ):
        raise ValueError("Completed run integrity failure: native workflow is missing or incomplete")
    if metadata is not None:
        for stage in metadata.stages:
            manifest = artifact_path / "multimodal" / stage.stage_dir / "boundary/integrity.json"
            if stage.status == "completed" and not manifest.is_file():
                raise ValueError(f"Data Designer stage integrity failure: {stage.stage_dir}")
    for manifest in sorted(artifact_path.glob("multimodal/stage-*/boundary/integrity.json")):
        stage_path = manifest.parent.parent
        verify_artifact_manifest(manifest, stage_path, boundary_artifacts(stage_path))


def workflow_stage_fingerprint(
    stage_config: dd.DataDesignerConfig,
    num_records: int,
    identity: str,
    upstream_fingerprint: str | None,
) -> str:
    """Match Data Designer's stage identity for this workflow's fixed stage settings.

    Data Designer 0.9.4/0.9.5 expose config fingerprints but no workflow preflight
    API. Keep this small compatibility payload aligned with CompositeWorkflow,
    so its IF_POSSIBLE fallback cannot erase incompatible saved evidence.

    Args:
        stage_config: Complete stage configuration, including its ordered seed.
        num_records: Original seed row count, including an optional control row.
        identity: Frozen callback version for this run.
        upstream_fingerprint: Previous stage identity, or None for the first stage.

    Returns:
        The native workflow fingerprint for the expected stage.
    """
    return fingerprint(
        {
            "config_fingerprint": stage_config.fingerprint(),
            "num_records": num_records,
            "sampling_strategy": "ordered",
            "selection_strategy": None,
            "allow_empty": False,
            "on_success_version": identity,
            "output_processors": [],
            "output": "final",
            "library_version": get_library_version(),
            "upstream_fingerprint": upstream_fingerprint,
        }
    )


def verify_stage_metadata(artifact_path: Path) -> WorkflowMetadata | None:
    """Validate native checkpoint structure and identities without changing artifacts.

    Args:
        artifact_path: Run-owned Data Designer artifact directory.

    Returns:
        Validated native metadata, or None before any workflow evidence exists.

    Raises:
        ValueError: Saved evidence lacks compatible metadata required for safe reuse.
    """
    workflow = artifact_path / "multimodal"
    metadata_path = workflow / "workflow-metadata.json"
    saved_stages = [stage for stage in sorted(workflow.glob("stage-*")) if any(stage.iterdir())]
    if not saved_stages and not metadata_path.exists():
        return None
    location = saved_stages[0].name if saved_stages else workflow.name
    try:
        metadata = WorkflowMetadata.model_validate_json(metadata_path.read_text())
        identity = fingerprint(json.loads((artifact_path / "identity.json").read_text()))
        if metadata.name != "multimodal" or metadata.library_version != get_library_version():
            raise ValueError("Workflow identity differs")
        records = {record.stage_dir: record for record in metadata.stages}
        for stage in saved_stages:
            location = stage.name
            if stage.name not in records:
                raise ValueError("Saved stage has no workflow record")
        upstream = None
        for index, record in enumerate(metadata.stages):
            location = record.stage_dir
            previous = metadata.stages[index - 1].name if index else None
            if (
                record.index != index
                or record.stage_dir != f"stage-{index}-{record.name}"
                or record.depends_on != ([previous] if previous else [])
                or record.status not in {"running", "failed", "completed"}
                or (index < len(metadata.stages) - 1 and record.status != "completed")
                or record.allow_empty
                or record.output_processors
                or record.output != "final"
                or record.sampling_strategy != "ordered"
                or record.selection_strategy is not None
                or record.on_success_version != identity
                or record.seeded_from_stage != previous
            ):
                raise ValueError("Stage identity differs")
            stage_config = dd.DataDesignerConfig.model_validate(record.config)
            expected = workflow_stage_fingerprint(stage_config, record.num_records_requested, identity, upstream)
            if record.fingerprint != expected:
                raise ValueError("Stage fingerprint differs")
            upstream = expected
            if record.status == "completed":
                expected_output = workflow / record.stage_dir / "boundary/seed.parquet"
                for value in (record.output_seed_path, record.callback_output_path):
                    selected = Path(value) if value else Path()
                    selected = selected if selected.is_absolute() else workflow / selected
                    if selected.resolve() != expected_output.resolve():
                        raise ValueError("Completed stage output differs")
                if record.stage_output_override_path is not None or record.output_processor_output_path is not None:
                    raise ValueError("Unexpected stage output override")
            elif any((workflow / record.stage_dir / "parquet-files").glob("*.parquet")):
                if not (workflow / record.stage_dir / "metadata.json").is_file():
                    raise ValueError("Unfinished stage has no dataset metadata")
        return metadata
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Data Designer resume metadata is invalid or missing for saved stage {location}; "
            f"existing output is preserved. Inspect {metadata_path} or use a new run directory."
        ) from exc


def verify_resume_stage(
    metadata: WorkflowMetadata,
    index: int,
    stage: Stage,
    builder: dd.DataDesignerConfigBuilder,
    seed: Path,
    identity: str,
    upstream_fingerprint: str | None,
    buffer_size: int,
    workflow_path: Path,
) -> str:
    """Compare a saved stage against the current graph before native resume can mutate it.

    Args:
        metadata: Structurally validated native workflow checkpoint.
        index: Stage position in the current graph.
        stage: Current stage declaration.
        builder: Current response-column configuration before upstream seeding.
        seed: Expected immutable input seed.
        identity: Current frozen callback version.
        upstream_fingerprint: Verified predecessor identity.
        buffer_size: Resolved native row-group size required for dataset resume.
        workflow_path: Directory containing the saved stage.

    Returns:
        The verified native stage fingerprint.

    Raises:
        ValueError: Metadata differs from the current graph or dataset checkpoint.
    """
    record = metadata.stages[index]
    try:
        expected_config = builder.build().model_copy(
            update={"seed_config": SeedConfig(source=dd.LocalFileSeedSource(path=str(seed)))}
        )
        count = pq.read_metadata(seed).num_rows
        expected = workflow_stage_fingerprint(expected_config, count, identity, upstream_fingerprint)
        if (
            record.name != stage.name
            or record.fingerprint != expected
            or record.config != expected_config.model_dump(mode="json")
        ):
            raise ValueError("Saved stage differs from the current graph")
        checkpoint = workflow_path / record.stage_dir / "metadata.json"
        if record.status in {"running", "failed"} and checkpoint.exists():
            dataset = json.loads(checkpoint.read_text())
            compiled = BuilderConfig.model_validate_json(
                checkpoint.with_name("builder_config.json").read_text()
            ).data_designer
            declared = compiled.model_copy(
                update={"columns": [column for column in compiled.columns if column.column_type != "seed-dataset"]}
            )
            if declared.model_dump(mode="json") != expected_config.model_dump(mode="json"):
                raise ValueError("Dataset builder differs from its stage")
            required = {
                **compiled.fingerprint(),
                "buffer_size": buffer_size,
                "target_num_records": count,
                "original_target_num_records": count,
                "preserve_dropped_columns": True,
            }
            if not isinstance(dataset, dict) or any(dataset.get(key) != value for key, value in required.items()):
                raise ValueError("Dataset checkpoint differs from its stage")
        return expected
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Data Designer resume metadata differs for saved stage {record.stage_dir}; existing output is preserved. "
            f"Inspect {workflow_path / record.stage_dir} or use a new run directory."
        ) from exc
