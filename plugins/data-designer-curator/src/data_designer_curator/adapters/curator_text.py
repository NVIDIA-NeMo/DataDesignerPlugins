# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_designer_curator.config import CuratorExecutionConfig
from data_designer_curator.errors import CuratorDependencyError, CuratorExecutionError

if TYPE_CHECKING:
    import pandas as pd


CURATOR_ID_COLUMN = "_dd_curator_id"
CURATOR_TEXT_COLUMN = "_dd_curator_text"
ORIGINAL_INDEX_COLUMN = "_dd_original_index"
FUZZY_ID_COLUMN = "_curator_dedup_id"

MODIFIER_PRIMITIVES = {
    "boilerplate_string": "nemo_curator.stages.text.modifiers.string.c4:BoilerPlateStringModifier",
    "line_remover": "nemo_curator.stages.text.modifiers.string.line_remover:LineRemover",
    "markdown_remover": "nemo_curator.stages.text.modifiers.string.markdown_remover:MarkdownRemover",
    "newline_normalizer": "nemo_curator.stages.text.modifiers.string.newline_normalizer:NewlineNormalizer",
    "quotation_remover": "nemo_curator.stages.text.modifiers.string.quotation_remover:QuotationRemover",
    "slicer": "nemo_curator.stages.text.modifiers.string.slicer:Slicer",
    "unicode_reformatter": "nemo_curator.stages.text.modifiers.unicode.unicode_reformatter:UnicodeReformatter",
    "url_remover": "nemo_curator.stages.text.modifiers.string.url_remover:UrlRemover",
}

TEXT_FILTER_PRIMITIVES = {
    "alpha": "nemo_curator.stages.text.filters.heuristic.code.code:AlphaFilter",
    "boilerplate_string": "nemo_curator.stages.text.filters.heuristic.string:BoilerPlateStringFilter",
    "bullets": "nemo_curator.stages.text.filters.heuristic.string:BulletsFilter",
    "common_english_words": "nemo_curator.stages.text.filters.heuristic.string:CommonEnglishWordsFilter",
    "ellipsis": "nemo_curator.stages.text.filters.heuristic.string:EllipsisFilter",
    "general_comment_to_code": "nemo_curator.stages.text.filters.heuristic.code.code:GeneralCommentToCodeFilter",
    "histogram": "nemo_curator.stages.text.filters.histogram.histogram:HistogramFilter",
    "html_boilerplate": "nemo_curator.stages.text.filters.heuristic.code.code:HTMLBoilerplateFilter",
    "long_word": "nemo_curator.stages.text.filters.heuristic.string:LongWordFilter",
    "mean_word_length": "nemo_curator.stages.text.filters.heuristic.string:MeanWordLengthFilter",
    "non_alpha_numeric": "nemo_curator.stages.text.filters.heuristic.string:NonAlphaNumericFilter",
    "number_of_lines_of_code": "nemo_curator.stages.text.filters.heuristic.code.code:NumberOfLinesOfCodeFilter",
    "numbers": "nemo_curator.stages.text.filters.heuristic.string:NumbersFilter",
    "parentheses": "nemo_curator.stages.text.filters.heuristic.string:ParenthesesFilter",
    "per_extension": "nemo_curator.stages.text.filters.heuristic.code.code:PerExtensionFilter",
    "pornographic_urls": "nemo_curator.stages.text.filters.heuristic.string:PornographicUrlsFilter",
    "punctuation": "nemo_curator.stages.text.filters.heuristic.string:PunctuationFilter",
    "python_comment_to_code": "nemo_curator.stages.text.filters.heuristic.code.code:PythonCommentToCodeFilter",
    "repeated_lines": "nemo_curator.stages.text.filters.heuristic.repetition.repetition:RepeatedLinesFilter",
    "repeated_lines_by_char": (
        "nemo_curator.stages.text.filters.heuristic.repetition.repetition:RepeatedLinesByCharFilter"
    ),
    "repeated_paragraphs": "nemo_curator.stages.text.filters.heuristic.repetition.repetition:RepeatedParagraphsFilter",
    "repeated_paragraphs_by_char": (
        "nemo_curator.stages.text.filters.heuristic.repetition.repetition:RepeatedParagraphsByCharFilter"
    ),
    "repeating_duplicate_ngrams": (
        "nemo_curator.stages.text.filters.heuristic.repetition.repetition:RepeatingDuplicateNGramsFilter"
    ),
    "repeating_top_ngrams": (
        "nemo_curator.stages.text.filters.heuristic.repetition.repetition:RepeatingTopNGramsFilter"
    ),
    "substring": "nemo_curator.stages.text.filters.heuristic.string:SubstringFilter",
    "symbols_to_words": "nemo_curator.stages.text.filters.heuristic.string:SymbolsToWordsFilter",
    "token_count": "nemo_curator.stages.text.filters.token.token_count:TokenCountFilter",
    "tokenizer_fertility": "nemo_curator.stages.text.filters.heuristic.code.code:TokenizerFertilityFilter",
    "urls": "nemo_curator.stages.text.filters.heuristic.string:UrlsFilter",
    "whitespace": "nemo_curator.stages.text.filters.heuristic.string:WhiteSpaceFilter",
    "word_count": "nemo_curator.stages.text.filters.heuristic.string:WordCountFilter",
    "words_without_alphabets": "nemo_curator.stages.text.filters.heuristic.string:WordsWithoutAlphabetsFilter",
    "xml_header": "nemo_curator.stages.text.filters.heuristic.code.code:XMLHeaderFilter",
}


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
    """Thin adapter over NeMo Curator text curation primitives."""

    def dedup(
        self,
        *,
        data: pd.DataFrame,
        dedup_type: str,
        text_columns: list[str],
        id_column: str | None,
        params: dict[str, Any],
        cache_dir: Path,
        execution: CuratorExecutionConfig,
    ) -> pd.DataFrame:

        prepared, text_field, id_field = self._prepare_dedup_input(data, text_columns, id_column)
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            if dedup_type == "exact":
                return self._run_exact_dedup(
                    data=prepared,
                    text_field=text_field,
                    id_field=id_field,
                    cache_dir=cache_dir,
                    execution=execution,
                    params=params,
                )
            if dedup_type == "fuzzy":
                return self._run_fuzzy_dedup(
                    data=prepared,
                    text_field=text_field,
                    cache_dir=cache_dir,
                    execution=execution,
                    params=params,
                )
            if dedup_type == "semantic":
                return self._run_semantic_dedup(
                    data=prepared,
                    text_field=text_field,
                    id_field=id_field,
                    cache_dir=cache_dir,
                    execution=execution,
                    params=params,
                )
        except Exception as error:
            if isinstance(error, CuratorExecutionError | CuratorDependencyError):
                raise
            raise CuratorExecutionError(f"Curator {dedup_type} dedup failed: {error}") from error

        raise CuratorExecutionError(f"Unsupported Curator dedup type: {dedup_type!r}")

    def _run_exact_dedup(
        self,
        *,
        data: pd.DataFrame,
        text_field: str,
        id_field: str,
        cache_dir: Path,
        execution: CuratorExecutionConfig,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        return self._run_current_workflow_dedup(
            data=data,
            text_field=text_field,
            id_field=id_field,
            cache_dir=cache_dir,
            execution=execution,
            workflow_type="exact",
            params=params,
        )

    def _run_fuzzy_dedup(
        self,
        *,
        data: pd.DataFrame,
        text_field: str,
        cache_dir: Path,
        execution: CuratorExecutionConfig,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        return self._run_current_workflow_dedup(
            data=data,
            text_field=text_field,
            id_field=FUZZY_ID_COLUMN,
            cache_dir=cache_dir,
            execution=execution,
            workflow_type="fuzzy",
            params=params,
        )

    def _run_semantic_dedup(
        self,
        *,
        data: pd.DataFrame,
        text_field: str,
        id_field: str,
        cache_dir: Path,
        execution: CuratorExecutionConfig,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        TextSemanticDeduplicationWorkflow = _import_semantic_dedup_workflow()
        input_dir = cache_dir / "input"
        output_dir = cache_dir / "semantic"
        input_dir.mkdir(parents=True, exist_ok=True)
        data.to_parquet(input_dir / "part.0.parquet", index=False)

        workflow_params = dict(params)
        workflow_params.pop("perform_removal", None)
        workflow_params.pop("input_path", None)
        workflow_params.pop("output_path", None)
        workflow_params.pop("cache_path", None)
        workflow_params.pop("text_field", None)
        workflow_params.pop("id_field", None)
        workflow_params.pop("input_filetype", None)

        with CuratorExecutionSession(execution, working_dir=cache_dir):
            workflow = TextSemanticDeduplicationWorkflow(
                input_path=str(input_dir),
                output_path=str(output_dir),
                cache_path=str(output_dir / "cache"),
                text_field=text_field,
                id_field=id_field,
                perform_removal=True,
                input_filetype="parquet",
                **workflow_params,
            )
            workflow.run()
        return _read_parquet_dir(output_dir / "deduplicated")

    def modify(
        self,
        *,
        data: pd.DataFrame,
        input_field: str,
        output_field: str | None,
        modifiers: list[dict[str, Any]],
    ) -> pd.DataFrame:
        Modify, DocumentBatch = _import_curator_modify()
        target_field = output_field or input_field
        working = data.copy()
        if output_field is not None:
            working[output_field] = working[input_field]

        try:
            stage = Modify(
                modifier_fn=[_build_modifier(spec) for spec in modifiers],
                input_fields=target_field,
            )
            batch = DocumentBatch(task_id="data-designer", dataset_name="data-designer", data=working)
            output = stage.process(batch)
        except Exception as error:
            raise CuratorExecutionError(f"Curator modify failed: {error}") from error

        return _document_batch_to_pandas(output).reset_index(drop=True)

    def text_filter(
        self,
        *,
        data: pd.DataFrame,
        default_text_field: str,
        filters: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        import pandas as pd

        ScoreFilter = _import_curator_score_filter()
        filter_objs = [_build_text_filter(spec) for spec in filters]
        text_fields = [spec.get("text_field") or default_text_field for spec in filters]
        score_fields = [spec.get("score_field") for spec in filters]
        inverts = [spec.get("invert", False) for spec in filters]

        try:
            single_filter = len(filter_objs) == 1
            stage = ScoreFilter(
                filter_obj=filter_objs[0] if single_filter else filter_objs,
                text_field=text_fields[0] if single_filter else text_fields,
                score_field=score_fields[0] if single_filter else score_fields,
                invert=inverts[0] if single_filter else inverts,
            )
            scored = data.copy()
            keep_mask = pd.Series(True, index=scored.index)
            for filter_obj, text_field, score_field, invert in zip(
                stage.filter_obj,
                stage.text_field,
                stage.score_field,
                stage.invert,
                strict=True,
            ):
                active = scored.loc[keep_mask].copy()
                active_mask = stage.compute_filter_mask(active, filter_obj, text_field, score_field, invert)
                if score_field is not None and score_field in active:
                    scored.loc[active.index, score_field] = active[score_field]
                keep_mask.loc[active.index] = active_mask.to_numpy()
        except Exception as error:
            raise CuratorExecutionError(f"Curator text filter failed: {error}") from error

        return scored.loc[keep_mask].reset_index(drop=True), keep_mask, scored

    def _prepare_dedup_input(
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

    def _run_current_workflow_dedup(
        self,
        *,
        data: pd.DataFrame,
        text_field: str,
        id_field: str,
        cache_dir: Path,
        execution: CuratorExecutionConfig,
        workflow_type: str,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        TextDuplicatesRemovalWorkflow = _import_duplicates_removal_workflow()
        input_dir = cache_dir / "input"
        ids_dir = cache_dir / workflow_type
        output_dir = cache_dir / "deduplicated"
        input_dir.mkdir(parents=True, exist_ok=True)
        data.to_parquet(input_dir / "part.0.parquet", index=False)

        workflow_params = dict(params)
        workflow_params.pop("input_path", None)
        workflow_params.pop("output_path", None)
        workflow_params.pop("cache_path", None)
        workflow_params.pop("text_field", None)
        workflow_params.pop("perform_removal", None)
        workflow_params.pop("input_filetype", None)
        workflow_params.pop("assign_id", None)
        workflow_params.pop("id_field", None)

        with CuratorExecutionSession(execution, working_dir=cache_dir):
            if workflow_type == "exact":
                DeduplicationWorkflow = _import_exact_dedup_workflow()
                workflow = DeduplicationWorkflow(
                    input_path=str(input_dir),
                    output_path=str(ids_dir),
                    text_field=text_field,
                    perform_removal=False,
                    assign_id=False,
                    id_field=id_field,
                    input_filetype="parquet",
                    **workflow_params,
                )
                workflow.run()
                ids_to_remove_path = ids_dir / "ExactDuplicateIds"
                id_generator_path = None
                duplicate_id_field = id_field
            elif workflow_type == "fuzzy":
                DeduplicationWorkflow = _import_fuzzy_dedup_workflow()
                workflow = DeduplicationWorkflow(
                    input_path=str(input_dir),
                    cache_path=str(ids_dir / "cache"),
                    output_path=str(ids_dir),
                    text_field=text_field,
                    perform_removal=False,
                    input_filetype="parquet",
                    **workflow_params,
                )
                result = workflow.run()
                ids_to_remove_path = ids_dir / "FuzzyDuplicateIds"
                id_generator_path = _get_workflow_metadata(result, "id_generator_path") or str(
                    ids_dir / "fuzzy_id_generator.json"
                )
                duplicate_id_field = FUZZY_ID_COLUMN
            else:
                raise CuratorExecutionError(f"Unsupported current Curator dedup workflow: {workflow_type!r}")

            removal_workflow = TextDuplicatesRemovalWorkflow(
                input_path=str(input_dir),
                ids_to_remove_path=str(ids_to_remove_path),
                output_path=str(output_dir),
                input_filetype="parquet",
                id_field=id_field,
                duplicate_id_field=duplicate_id_field,
                id_generator_path=id_generator_path,
                output_mode="overwrite",
            )
            removal_workflow.run()
        return _read_parquet_dir(output_dir)


def _import_exact_dedup_workflow() -> type:
    try:
        from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
    except Exception as error:
        raise CuratorDependencyError(
            "NeMo Curator exact dedup workflows are not installed. Install NeMo Curator with the text_cuda12 "
            "extra in the same environment."
        ) from error
    return ExactDeduplicationWorkflow


def _import_fuzzy_dedup_workflow() -> type:
    try:
        from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
    except Exception as error:
        raise CuratorDependencyError(
            "NeMo Curator fuzzy dedup workflow is not installed. Install NeMo Curator with the text_cuda12 "
            "extra in the same environment."
        ) from error
    return FuzzyDeduplicationWorkflow


def _import_semantic_dedup_workflow() -> type:
    try:
        from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow
    except Exception as error:
        raise CuratorDependencyError(
            "NeMo Curator semantic dedup workflow is not installed. Install NeMo Curator with the text_cuda12 "
            "extra in the same environment."
        ) from error
    return TextSemanticDeduplicationWorkflow


def _import_duplicates_removal_workflow() -> type:
    try:
        from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
    except Exception as error:
        raise CuratorDependencyError(
            "NeMo Curator duplicate removal workflow is not installed. Install NeMo Curator with the text_cuda12 "
            "extra in the same environment."
        ) from error
    return TextDuplicatesRemovalWorkflow


def _import_curator_modify() -> tuple[type, type]:
    try:
        from nemo_curator.stages.text.modifiers.modifier import Modify
        from nemo_curator.tasks import DocumentBatch
    except Exception as error:
        raise CuratorDependencyError(
            "NeMo Curator text modifiers are not installed. Install data-designer-curator[curator-text-cpu] "
            "or install NeMo Curator in the same environment."
        ) from error
    return Modify, DocumentBatch


def _import_curator_score_filter() -> type:
    try:
        from nemo_curator.stages.text.filters.score_filter import ScoreFilter
    except Exception as error:
        raise CuratorDependencyError(
            "NeMo Curator text filters are not installed. Install data-designer-curator[curator-text-cpu] "
            "or install NeMo Curator in the same environment."
        ) from error
    return ScoreFilter


def _build_modifier(spec: dict[str, Any]) -> object:
    return _build_primitive(spec, MODIFIER_PRIMITIVES)


def _build_text_filter(spec: dict[str, Any]) -> object:
    return _build_primitive(spec, TEXT_FILTER_PRIMITIVES)


def _build_primitive(spec: dict[str, Any], registry: dict[str, str]) -> object:
    primitive = spec["primitive"]
    path = registry[primitive]
    params = spec.get("params", {})
    try:
        return _load_class(path)(**params)
    except Exception as error:
        raise CuratorExecutionError(f"Unable to build Curator primitive {primitive!r}: {error}") from error


def _load_class(path: str) -> type:
    module_name, _, class_name = path.partition(":")
    module = import_module(module_name)
    return getattr(module, class_name)


def _get_workflow_metadata(result: object, key: str) -> object | None:
    if hasattr(result, "get_metadata"):
        return result.get_metadata(key)
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _document_batch_to_pandas(batch: object) -> pd.DataFrame:
    if batch is None:
        raise CuratorExecutionError("Curator stage returned no output batch.")
    if hasattr(batch, "to_pandas"):
        return batch.to_pandas()
    raise CuratorExecutionError("Unable to convert NeMo Curator DocumentBatch output to pandas.")


def _read_parquet_dir(output_dir: Path) -> pd.DataFrame:
    import pandas as pd

    files = sorted(output_dir.rglob("*.parquet"))
    if not files:
        raise CuratorExecutionError(f"Curator did not write parquet output under {str(output_dir)!r}.")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
