# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The document model.

This module holds the single most important idea in this example: one Pydantic
model is used twice.

1. As the ``output_format`` of an ``LLMStructuredColumnConfig``, so the LLM is
   constrained to emit a valid document *outline* rather than a wall of prose.
2. As the input contract of the ``.docx`` renderer.

Because both ends share the model, "the LLM produced something the renderer
can't handle" stops being a class of bug you have to defend against in the
renderer with regexes and heuristics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocTable(BaseModel):
    """A simple rectangular table.

    Rows are ragged in practice — LLMs drop or add a cell now and then — so the
    renderer normalizes each row against ``columns`` rather than trusting it.
    """

    caption: str = Field(description="Short caption describing what the table contains.")
    columns: list[str] = Field(description="Column headers, 2 to 4 of them.")
    rows: list[list[str]] = Field(description="Table rows. Each row has one cell per column header.")


class DocSection(BaseModel):
    """One numbered section of the document."""

    heading: str = Field(description="Section heading, title case, no numbering prefix.")
    paragraphs: list[str] = Field(
        description="One to three body paragraphs of prose. No markdown, no bullet characters."
    )
    bullets: list[str] = Field(
        default_factory=list,
        description=(
            "Optional bulleted requirements or steps for this section. Use an empty list when the "
            "section reads better as prose only."
        ),
    )


class WordDocument(BaseModel):
    """A complete business document, structured for rendering."""

    title: str = Field(description="Document title.")
    subtitle: str = Field(description="One-line subtitle, e.g. the scope or the owning function.")
    summary: str = Field(description="A single paragraph executive summary, 40-80 words.")
    sections: list[DocSection] = Field(description="Four to six sections that make up the body.")
    key_data: DocTable = Field(
        description=(
            "A table carrying the document's structured facts — thresholds, review cadences, "
            "roles and responsibilities, retention windows, or similar."
        )
    )
