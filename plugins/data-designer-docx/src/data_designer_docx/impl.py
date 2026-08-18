# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.utils import deserialize_json_values
from pydantic import ValidationError

from data_designer_docx.config import DocxProcessorConfig
from data_designer_docx.render import render_document, safe_filename
from data_designer_docx.schema import WordDocument

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

DOCX_SUFFIX = ".docx"


def to_text(value: Any) -> str:
    """Convert an interpolated Jinja2 value to text.

    Used as the ``record_str_fn`` finalize hook so ``None`` renders as an empty
    string rather than the literal ``"None"`` in file names and metadata.

    Args:
        value: The value being interpolated into a template.

    Returns:
        The value as a string, or an empty string when the value is ``None``.
    """
    return "" if value is None else str(value)


class DocxProcessor(WithJinja2UserTemplateRendering, Processor[DocxProcessorConfig]):
    """Writes one ``.docx`` file per row as each batch completes.

    Runs at the post-batch stage rather than after generation so that documents
    stream out while the run is still in progress, the row count stays fixed as
    the async engine requires, and the dataset remains resumable.
    """

    def _initialize(self) -> None:
        """Reset the per-run record of file names already written."""
        self._used_filenames: set[str] = set()

    @property
    def output_dir(self) -> Path:
        """Directory that documents for this processor are written to."""
        return self.base_dataset_path / self.config.output_subdir / self.config.name

    def relative_path(self, filename: str) -> str:
        """Build the dataset-relative path stored in the output column.

        Args:
            filename: The file name written to disk.

        Returns:
            A path relative to the dataset directory, keeping the dataset portable.
        """
        return f"{self.config.output_subdir}/{self.config.name}/{filename}"

    def render_for_all_records(self, template: str, columns: list[str], records: list[dict]) -> list[str]:
        """Render one Jinja2 template across every record in a batch.

        The renderer is prepared once per template rather than once per row,
        because building the sandboxed environment is the expensive part.

        Args:
            template: A user-supplied Jinja2 template.
            columns: Column names allowed as template references.
            records: The batch records to render against.

        Returns:
            One rendered string per record, in batch order.
        """
        self.prepare_jinja2_template_renderer(template, columns, record_str_fn=to_text)
        return [self.render_template(record) for record in records]

    def unique_filename(self, rendered: str) -> str:
        """Sanitize a rendered file name and make it unique within the run.

        Args:
            rendered: The raw rendered file name.

        Returns:
            A filesystem-safe name, suffixed with ``-1``, ``-2``, and so on if needed.
        """
        filename = safe_filename(rendered)
        if filename not in self._used_filenames:
            self._used_filenames.add(filename)
            return filename
        stem = filename[: -len(DOCX_SUFFIX)]
        counter = 1
        while f"{stem}-{counter}{DOCX_SUFFIX}" in self._used_filenames:
            counter += 1
        deduped = f"{stem}-{counter}{DOCX_SUFFIX}"
        self._used_filenames.add(deduped)
        return deduped

    def parse_document(self, value: Any) -> WordDocument | None:
        """Parse a structured column value into a :class:`WordDocument`.

        Args:
            value: A JSON string or already-decoded mapping from the dataset.

        Returns:
            The parsed document, or ``None`` when the value is missing or invalid.
        """
        if value is None:
            return None
        try:
            if isinstance(value, str):
                return WordDocument.model_validate_json(value)
            return WordDocument.model_validate(value)
        except (ValidationError, ValueError) as exc:
            logger.warning(f"⚠️ Skipping row: {self.config.document_column!r} is not a valid document ({exc}).")
            return None

    def render_options(self, records: list[dict], columns: list[str]) -> dict[str, Any]:
        """Pre-render every configured template across the batch.

        Args:
            records: The batch records.
            columns: Column names allowed as template references.

        Returns:
            A mapping with rendered ``filenames``, ``metadata``, ``core_properties``,
            and ``footers`` entries.
        """
        return {
            "filenames": self.render_for_all_records(self.config.filename_template, columns, records),
            "metadata": {
                label: self.render_for_all_records(template, columns, records)
                for label, template in self.config.metadata_columns.items()
            },
            "core_properties": {
                prop: self.render_for_all_records(template, columns, records)
                for prop, template in self.config.core_property_columns.items()
            },
            "footers": (
                self.render_for_all_records(self.config.footer_template, columns, records)
                if self.config.footer_template
                else None
            ),
        }

    def process_after_batch(self, data: pd.DataFrame, *, current_batch_number: int | None) -> pd.DataFrame:
        """Render each row of a completed batch to a ``.docx`` file.

        Args:
            data: The generated batch data.
            current_batch_number: The batch index, or ``None`` in preview mode.

        Returns:
            The batch with the output path column added.

        Raises:
            ValueError: If the configured document column is not in the dataset.
        """
        if data.empty and self.config.document_column not in data.columns:
            logger.warning("⚠️ Empty batch reached the docx processor; no documents written.")
            return data

        if self.config.document_column not in data.columns:
            raise ValueError(
                f"Column {self.config.document_column!r} not found in the dataset. "
                f"Available columns: {sorted(data.columns)}"
            )

        columns = data.columns.to_list()
        records = [deserialize_json_values(record) for record in data.to_dict(orient="records")]
        options = self.render_options(records, columns)

        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        written: list[str | None] = []
        for row, record in enumerate(records):
            document = self.parse_document(record.get(self.config.document_column))
            if document is None:
                written.append(None)
                continue
            filename = self.unique_filename(options["filenames"][row])
            render_document(
                document,
                output_dir / filename,
                metadata={label: values[row] for label, values in options["metadata"].items()},
                template_path=self.config.template_path,
                footer_text=options["footers"][row] if options["footers"] else None,
                table_style=self.config.table_style,
                number_sections=self.config.number_sections,
                core_properties={prop: values[row] for prop, values in options["core_properties"].items()},
            )
            written.append(self.relative_path(filename))

        data[self.config.output_path_column] = written
        logger.info(f"📄 Wrote {sum(path is not None for path in written)} .docx file(s) to {output_dir}")
        return data
