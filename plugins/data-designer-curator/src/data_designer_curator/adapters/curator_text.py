# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_designer_curator.config import CuratorExecutionConfig
from data_designer_curator.errors import CuratorDependencyError, CuratorExecutionError

if TYPE_CHECKING:
    import pandas as pd


CURATOR_ID_COLUMN = "_dd_curator_id"
CURATOR_TEXT_COLUMN = "_dd_curator_text"
ORIGINAL_INDEX_COLUMN = "_dd_original_index"


class CuratorExecutionSession:
    """Own optional Curator/Ray setup for one processor invocation."""

    def __init__(self, config: CuratorExecutionConfig, *, working_dir: Path) -> None:
        self.config = config
        self.working_dir = working_dir
        self._client: Any | None = None
        self._old_ray_address: str | None = None
        self._ray_address_changed = False

    def __enter__(self) -> "CuratorExecutionSession":
        if self.config.mode == "none":
            return self
        if self.config.mode == "local_ray":
            self._start_local_ray()
        elif self.config.mode == "existing_ray":
            self._connect_existing_ray()
        else:
            raise CuratorExecutionError(f"Unsupported Curator execution mode: {self.config.mode!r}")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._client is not None and hasattr(self._client, "stop"):
            self._client.stop()
        self._restore_ray_address()

    def _start_local_ray(self) -> None:
        try:
            from nemo_curator.core.client import RayClient
        except Exception as error:
            raise CuratorDependencyError(
                "NeMo Curator Ray support is not installed. Install NeMo Curator with the text_cuda12 extra "
                "in the same environment."
            ) from error

        kwargs = {
            "num_cpus": self.config.num_cpus,
            "num_gpus": self.config.num_gpus,
            "object_store_memory": self.config.object_store_memory,
            "enable_object_spilling": self.config.enable_object_spilling,
            "include_dashboard": self.config.include_dashboard,
            "metrics_dir": self.config.metrics_dir,
            **self.config.client_kwargs,
        }
        if self.config.ray_temp_dir is not None:
            kwargs["ray_temp_dir"] = self.config.ray_temp_dir
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        self._client = RayClient(**kwargs)
        self._client.start()

    def _connect_existing_ray(self) -> None:
        ray_address = self.config.ray_address or os.environ.get("RAY_ADDRESS")
        if ray_address is None:
            raise CuratorExecutionError("existing_ray mode requires ray_address or RAY_ADDRESS.")

        self._old_ray_address = os.environ.get("RAY_ADDRESS")
        self._ray_address_changed = True
        os.environ["RAY_ADDRESS"] = ray_address

        try:
            import ray
        except Exception as error:
            raise CuratorDependencyError("ray is not installed.") from error

        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)

    def _restore_ray_address(self) -> None:
        if not self._ray_address_changed:
            return
        if self._old_ray_address is None:
            os.environ.pop("RAY_ADDRESS", None)
        else:
            os.environ["RAY_ADDRESS"] = self._old_ray_address


class CuratorTextAdapter:
    """Thin adapter over NeMo Curator text curation atoms."""

    def exact_dedup(
        self,
        *,
        data: pd.DataFrame,
        text_columns: list[str],
        id_column: str | None,
        hash_method: str,
        cache_dir: Path,
        execution: CuratorExecutionConfig,
    ) -> pd.DataFrame:

        prepared, text_field, id_field = self._prepare_exact_input(data, text_columns, id_column)
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._run_legacy_exact_dedup(
                data=prepared,
                text_field=text_field,
                id_field=id_field,
                hash_method=hash_method,
                cache_dir=cache_dir,
                execution=execution,
            )
        except CuratorDependencyError:
            return self._run_workflow_exact_dedup(
                data=prepared,
                text_field=text_field,
                id_field=id_field,
                cache_dir=cache_dir,
                execution=execution,
            )
        except Exception as error:
            if isinstance(error, CuratorExecutionError):
                raise
            raise CuratorExecutionError(f"Curator exact dedup failed: {error}") from error

    def score_filter(
        self,
        *,
        data: pd.DataFrame,
        score_column: str,
        min_score: float | None,
        max_score: float | None,
        keep_null_scores: bool,
    ) -> tuple[pd.DataFrame, pd.Series]:
        import pandas as pd

        Filter = _import_curator_filter()

        def keep_score(score: object) -> bool:
            if pd.isna(score):
                return keep_null_scores
            if min_score is not None and score < min_score:
                return False
            return not (max_score is not None and score > max_score)

        try:
            stage = Filter(filter_fn=keep_score, filter_field=score_column)
            mask = stage.compute_filter_mask(data, keep_score, score_column, False)
        except Exception as error:
            raise CuratorExecutionError(f"Curator score filter failed: {error}") from error

        return data.loc[mask].reset_index(drop=True), mask

    def _prepare_exact_input(
        self,
        data: pd.DataFrame,
        text_columns: list[str],
        id_column: str | None,
    ) -> tuple[pd.DataFrame, str, str]:
        prepared = data.copy()
        if len(text_columns) == 1:
            text_field = text_columns[0]
        else:
            text_field = CURATOR_TEXT_COLUMN
            prepared[text_field] = prepared[text_columns].fillna("").astype(str).agg("\x1f".join, axis=1)

        if id_column is None:
            id_field = CURATOR_ID_COLUMN
            prepared[id_field] = prepared[ORIGINAL_INDEX_COLUMN].astype(str)
        else:
            id_field = id_column

        return prepared, text_field, id_field

    def _run_legacy_exact_dedup(
        self,
        *,
        data: pd.DataFrame,
        text_field: str,
        id_field: str,
        hash_method: str,
        cache_dir: Path,
        execution: CuratorExecutionConfig,
    ) -> pd.DataFrame:
        ExactDuplicates, DocumentDataset = _import_legacy_exact_dedup()
        with CuratorExecutionSession(execution, working_dir=cache_dir):
            dataset = _document_dataset_from_pandas(DocumentDataset, data)
            deduper = ExactDuplicates(
                id_field=id_field,
                text_field=text_field,
                hash_method=hash_method,
                perform_removal=True,
                cache_dir=str(cache_dir),
            )

            if hasattr(deduper, "identify_duplicates") and hasattr(deduper, "remove"):
                duplicates = deduper.identify_duplicates(dataset)
                output = deduper.remove(dataset, duplicates)
            else:
                output = deduper(dataset)

        return _document_dataset_to_pandas(output)

    def _run_workflow_exact_dedup(
        self,
        *,
        data: pd.DataFrame,
        text_field: str,
        id_field: str,
        cache_dir: Path,
        execution: CuratorExecutionConfig,
    ) -> pd.DataFrame:
        ExactDeduplicationWorkflow, TextDuplicatesRemovalWorkflow = _import_exact_dedup_workflows()
        input_dir = cache_dir / "input"
        ids_dir = cache_dir / "exact"
        output_dir = cache_dir / "deduplicated"
        input_dir.mkdir(parents=True, exist_ok=True)
        data.to_parquet(input_dir / "part.0.parquet", index=False)

        with CuratorExecutionSession(execution, working_dir=cache_dir):
            exact_workflow = ExactDeduplicationWorkflow(
                input_path=str(input_dir),
                output_path=str(ids_dir),
                text_field=text_field,
                perform_removal=False,
                assign_id=False,
                id_field=id_field,
                input_filetype="parquet",
            )
            exact_workflow.run()

            removal_workflow = TextDuplicatesRemovalWorkflow(
                input_path=str(input_dir),
                ids_to_remove_path=str(ids_dir / "ExactDuplicateIds"),
                output_path=str(output_dir),
                input_filetype="parquet",
                input_id_field=id_field,
                ids_to_remove_duplicate_id_field=id_field,
            )
            removal_workflow.run()
        return _read_parquet_dir(output_dir)


def _import_legacy_exact_dedup() -> tuple[type, type]:
    try:
        from nemo_curator import ExactDuplicates
        from nemo_curator.datasets import DocumentDataset
    except Exception as error:
        raise CuratorDependencyError(
            "NeMo Curator legacy exact dedup is not installed. Install a compatible NeMo Curator version "
            "or use the current text_cuda12 workflow dependencies."
        ) from error
    return ExactDuplicates, DocumentDataset


def _import_exact_dedup_workflows() -> tuple[type, type]:
    try:
        from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
        from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
    except Exception as error:
        raise CuratorDependencyError(
            "NeMo Curator exact dedup workflows are not installed. Install NeMo Curator with the text_cuda12 "
            "extra in the same environment."
        ) from error
    return ExactDeduplicationWorkflow, TextDuplicatesRemovalWorkflow


def _import_curator_filter() -> type:
    import_errors: list[Exception] = []
    for import_fn in (_import_current_filter, _import_legacy_filter):
        try:
            return import_fn()
        except Exception as error:
            import_errors.append(error)
    raise CuratorDependencyError(
        "NeMo Curator text filters are not installed. Install data-designer-curator[curator-text-cpu] "
        "or install NeMo Curator in the same environment."
    ) from import_errors[-1]


def _import_current_filter() -> type:
    from nemo_curator.stages.text.filters.score_filter import Filter

    return Filter


def _import_legacy_filter() -> type:
    from nemo_curator.modules import Filter

    return Filter


def _document_dataset_from_pandas(DocumentDataset: type, data: pd.DataFrame) -> object:
    if hasattr(DocumentDataset, "from_pandas"):
        return DocumentDataset.from_pandas(data, npartitions=1)
    raise CuratorExecutionError("Installed NeMo Curator DocumentDataset does not support from_pandas().")


def _document_dataset_to_pandas(dataset: object) -> pd.DataFrame:
    import pandas as pd

    if hasattr(dataset, "to_pandas"):
        value = dataset.to_pandas()
        return value.compute() if hasattr(value, "compute") else value

    for attr in ("df", "dataset_df"):
        value = getattr(dataset, attr, None)
        if value is None:
            continue
        value = value.compute() if hasattr(value, "compute") else value
        if isinstance(value, pd.DataFrame):
            return value

    raise CuratorExecutionError("Unable to convert NeMo Curator DocumentDataset output to pandas.")


def _read_parquet_dir(output_dir: Path) -> pd.DataFrame:
    import pandas as pd

    files = sorted(output_dir.rglob("*.parquet"))
    if not files:
        raise CuratorExecutionError(f"Curator did not write parquet output under {str(output_dir)!r}.")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
