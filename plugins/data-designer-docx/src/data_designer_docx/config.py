# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.base import ProcessorConfig
from pydantic import Field, field_validator

RESERVED_SUBDIRS = frozenset({"processors-files", "parquet-files", "dropped-columns-parquet-files"})


class DocxProcessorConfig(ProcessorConfig):
    """Renders one Microsoft Word document per row.

    Files are written to ``<artifact_path>/<dataset>/<output_subdir>/<name>/`` and
    the relative path of each file is stored in ``output_path_column`` so rows and
    documents stay joined.

    Documents are *not* written under ``processors-files/``. Data Designer reads
    every directory there back as a parquet dataset, so binary artifacts need
    their own folder, in the same way generated images live under ``images/``.

    Attributes:
        document_column: Column holding a ``WordDocument``-shaped value, typically
            produced by an ``LLMStructuredColumnConfig`` using ``WordDocument`` as
            its ``output_format``.
        output_subdir: Folder under the dataset directory that documents are
            written to. Reserved Data Designer folder names are rejected.
        filename_template: Jinja2 template for the file name, rendered per row and
            sanitized. Collisions are resolved by appending ``-1``, ``-2``, and so on.
        output_path_column: Name of the column that receives the written path.
        metadata_columns: Label to Jinja2 template pairs rendered as a front-matter
            table beneath the document title.
        core_property_columns: Word core property name (``author``, ``category``,
            ``comments``, ``subject``, ``keywords``) to Jinja2 template pairs.
        template_path: Optional ``.docx`` supplying styles, header, and footer. The
            template should contain styles only; python-docx appends generated
            content after any body content already present.
        footer_template: Optional Jinja2 template for the page footer.
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
        default="{{ doc_id }}.docx",
        description="Jinja2 template for the output file name.",
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
        """Reject folder names that Data Designer manages itself.

        Args:
            value: The requested output subdirectory.

        Returns:
            The validated subdirectory name.

        Raises:
            ValueError: If the name collides with a Data Designer-managed folder.
        """
        if value.strip("/") in RESERVED_SUBDIRS:
            raise ValueError(
                f"output_subdir={value!r} collides with a Data Designer-managed folder. "
                "Directories under 'processors-files' are read back as parquet datasets."
            )
        return value
