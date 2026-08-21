# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.base import ProcessorConfig
from pydantic import Field, field_validator

# Directories Data Designer creates, reads, or deletes inside a dataset folder.
# Hardcoded rather than imported so this module stays free of engine imports; the
# processor additionally asserts containment against the live artifact storage at
# write time. Keep in sync with data_designer.engine.storage.artifact_storage.
RESERVED_DIRECTORY_NAMES = frozenset(
    {
        "parquet-files",
        "processors-files",
        "dropped-columns-parquet-files",
        "tmp-partial-parquet-files",
        "images",
    }
)

DEFAULT_FILENAME_TEMPLATE = "document.docx"


def validate_path_component(value: str, *, field: str, allow_nested: bool) -> str:
    """Reject path values that escape the dataset directory or shadow managed folders.

    Args:
        value: The configured path fragment.
        field: Field name, used in error messages.
        allow_nested: Whether ``/`` separated segments are permitted.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the value is absolute, contains traversal segments, or names
            a Data Designer-managed directory.
    """
    if not value or not value.strip():
        raise ValueError(f"{field} must not be empty.")
    if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
        raise ValueError(f"{field}={value!r} must be relative to the dataset directory, not absolute.")

    segments = [segment for segment in value.split("/") if segment]
    if not allow_nested and len(segments) > 1:
        raise ValueError(f"{field}={value!r} must be a single directory name, not a nested path.")

    for segment in segments:
        if segment in {".", ".."}:
            raise ValueError(f"{field}={value!r} must not contain '.' or '..' path segments.")
        if segment in RESERVED_DIRECTORY_NAMES:
            raise ValueError(
                f"{field}={value!r} uses the Data Designer-managed directory {segment!r}. "
                "Data Designer reads or deletes these folders, so documents written there "
                "can be misread as parquet or removed on resume."
            )
    return "/".join(segments)


class DocxProcessorConfig(ProcessorConfig):
    """Renders one Microsoft Word document per row.

    Files are written to ``<artifact_path>/<dataset>/<output_subdir>/<name>/`` and
    the relative path of each file is stored in ``output_path_column`` so rows and
    documents stay joined.

    Documents are *not* written under ``processors-files/``. Data Designer reads
    every directory there back as a parquet dataset, so binary artifacts need
    their own folder, in the same way generated images live under ``images/``.
    Both ``output_subdir`` and ``name`` are validated as contained, non-reserved
    path components.

    Attributes:
        document_column: Column holding a ``WordDocument``-shaped value, typically
            produced by an ``LLMStructuredColumnConfig`` using ``WordDocument`` as
            its ``output_format``.
        output_subdir: Folder under the dataset directory that documents are
            written to. Must be relative and must not name a managed directory.
        filename_template: Jinja2 template for the file name, rendered per row and
            sanitized. The default references no dataset columns; collisions are
            resolved by appending ``-1``, ``-2``, and so on.
        output_path_column: Name of the column that receives the written path.
        metadata_columns: Label to Jinja2 template pairs rendered as a front-matter
            table beneath the document title.
        core_property_columns: Word core property name (``author``, ``category``,
            ``comments``, ``subject``, ``keywords``) to Jinja2 template pairs.
        template_path: Optional ``.docx`` supplying styles, header, and footer. The
            template should contain styles only; python-docx appends generated
            content after any body content already present.
        footer_template: Optional Jinja2 template for the page footer, applied to
            every section of the rendered document.
        table_style: Table style name, which must exist in the template document.
        number_sections: Whether to prefix section headings with ``1.``, ``2.``, and so on.
    """

    processor_type: Literal["docx"] = "docx"

    document_column: str = Field(description="Column containing the structured document.")
    output_subdir: str = Field(
        default="documents",
        description="Folder under the dataset directory that .docx files are written to.",
    )
    filename_template: str = Field(
        default=DEFAULT_FILENAME_TEMPLATE,
        description=(
            "Jinja2 template for the output file name. The default references no dataset columns; "
            "duplicates are de-duplicated with a numeric suffix."
        ),
    )
    output_path_column: str = Field(
        default="docx_path",
        description="Column that receives the relative path of the written file.",
    )
    metadata_columns: dict[str, str] = Field(
        default_factory=dict,
        description="Label to Jinja2 template pairs rendered as a front-matter table.",
    )
    core_property_columns: dict[str, str] = Field(
        default_factory=dict,
        description="Word core property name to Jinja2 template pairs.",
    )
    template_path: str | None = Field(
        default=None,
        description="Optional .docx template supplying styles, header, and footer.",
    )
    footer_template: str | None = Field(
        default=None,
        description="Optional Jinja2 template for the page footer.",
    )
    table_style: str = Field(default="Table Grid", description="Table style name.")
    number_sections: bool = Field(default=True, description="Number section headings.")

    @field_validator("output_subdir")
    @classmethod
    def validate_output_subdir(cls, value: str) -> str:
        """Ensure the output folder stays inside the dataset and avoids managed names."""
        return validate_path_component(value, field="output_subdir", allow_nested=True)

    @field_validator("name")
    @classmethod
    def validate_name_is_safe_directory(cls, value: str) -> str:
        """The processor name becomes a directory, so it must be a single safe component."""
        return validate_path_component(value, field="name", allow_nested=False)
