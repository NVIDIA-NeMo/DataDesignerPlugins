# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd
import pytest
from data_designer.config import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner
from docx import Document
from pydantic import ValidationError

from data_designer_docx.config import DocxProcessorConfig
from data_designer_docx.plugin import plugin
from data_designer_docx.render import normalize_rows, render_document, safe_filename
from data_designer_docx.schema import DocSection, DocTable, WordDocument


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)


def make_document(title: str = "Access Control Standard") -> WordDocument:
    """Build a small but complete document for rendering tests."""
    return WordDocument(
        title=title,
        subtitle="Information Security",
        summary="This standard defines how access is granted, reviewed, and revoked.",
        sections=[
            DocSection(heading="Scope", paragraphs=["Applies to all systems."], bullets=["Production"]),
            DocSection(heading="Roles", paragraphs=["The owner attests quarterly."]),
        ],
        key_data=DocTable(
            caption="Review Cadence",
            columns=["Role", "Action", "Frequency"],
            rows=[["Owner", "Attest", "Quarterly"]],
        ),
    )


class TestDocxProcessorConfig:
    def test_defaults(self) -> None:
        config = DocxProcessorConfig(name="docs", document_column="document")
        assert config.processor_type == "docx"
        assert config.output_subdir == "documents"
        assert config.output_path_column == "docx_path"

    @pytest.mark.parametrize("reserved", ["processors-files", "parquet-files"])
    def test_reserved_output_subdir_is_rejected(self, reserved: str) -> None:
        """Data Designer reads these folders back as parquet, so documents must not go there."""
        with pytest.raises(ValidationError, match="Data Designer-managed folder"):
            DocxProcessorConfig(name="docs", document_column="document", output_subdir=reserved)


class TestSafeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("POL-1 Access Control", "POL-1-Access-Control.docx"),
            ("already.docx", "already.docx"),
            ("../../etc/passwd", "etc-passwd.docx"),
            ("///", "document.docx"),
        ],
    )
    def test_sanitizes(self, raw: str, expected: str) -> None:
        assert safe_filename(raw) == expected


class TestNormalizeRows:
    def test_pads_short_rows(self) -> None:
        """Structured outputs constrain JSON shape, not cell counts, so rows arrive ragged."""
        table = DocTable(caption="c", columns=["a", "b", "c"], rows=[["1", "2"]])
        assert normalize_rows(table) == [["1", "2", ""]]

    def test_truncates_long_rows(self) -> None:
        table = DocTable(caption="c", columns=["a", "b"], rows=[["1", "2", "3"]])
        assert normalize_rows(table) == [["1", "2"]]


class TestRenderDocument:
    def test_writes_readable_docx(self, tmp_path: Path) -> None:
        path = render_document(
            make_document(),
            tmp_path / "out.docx",
            metadata={"Document ID": "POL-1"},
            footer_text="Internal",
            core_properties={"author": "Jane Doe", "category": "Standard"},
        )

        rendered = Document(str(path))
        headings = [p.text for p in rendered.paragraphs if p.style.name.startswith(("Title", "Heading"))]
        assert headings[0] == "Access Control Standard"
        assert "1. Scope" in headings
        assert rendered.core_properties.author == "Jane Doe"
        assert rendered.sections[0].footer.paragraphs[0].text == "Internal"

    def test_metadata_and_key_data_tables(self, tmp_path: Path) -> None:
        path = render_document(make_document(), tmp_path / "out.docx", metadata={"Owner": "Jane"})

        rendered = Document(str(path))
        assert len(rendered.tables) == 2
        assert [cell.text for cell in rendered.tables[1].rows[0].cells] == ["Role", "Action", "Frequency"]

    def test_number_sections_disabled(self, tmp_path: Path) -> None:
        path = render_document(make_document(), tmp_path / "out.docx", number_sections=False)

        rendered = Document(str(path))
        headings = [p.text for p in rendered.paragraphs if p.style.name.startswith("Heading")]
        assert "Scope" in headings
        assert "1. Scope" not in headings

    def test_template_styles_are_inherited(self, tmp_path: Path) -> None:
        template_path = tmp_path / "template.docx"
        template = Document()
        template.styles["Normal"].font.name = "Georgia"
        template.save(str(template_path))

        path = render_document(make_document(), tmp_path / "out.docx", template_path=template_path)

        assert Document(str(path)).styles["Normal"].font.name == "Georgia"


class TestDocxProcessorPreviewIntegration:
    """Run the processor through Data Designer using seed data, so no model is needed."""

    @pytest.fixture()
    def seed_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "doc_id": ["POL-1", "POL-2"],
                "document": [make_document("First").model_dump_json(), make_document("Second").model_dump_json()],
            }
        )

    def build(self, seed_df: pd.DataFrame, **overrides: object) -> DataDesignerConfigBuilder:
        builder = DataDesignerConfigBuilder()
        builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
        # Data Designer's profiler requires at least one generated column, and a
        # seed-plus-processor config has none.
        builder.add_column(ExpressionColumnConfig(name="doc_label", expr="{{ doc_id }}"))
        builder.add_processor(
            DocxProcessorConfig(
                name="word-documents",
                document_column="document",
                filename_template="{{ doc_id }}.docx",
                **overrides,
            )
        )
        return builder

    def test_preview_writes_documents(self, seed_df: pd.DataFrame, tmp_path: Path) -> None:
        artifact_path = tmp_path / "artifacts"
        artifact_path.mkdir()

        result = DataDesigner(artifact_path=artifact_path).preview(self.build(seed_df), num_records=2)

        assert "docx_path" in result.dataset.columns
        paths = sorted(artifact_path.rglob("*.docx"))
        assert [path.name for path in paths] == ["POL-1.docx", "POL-2.docx"]
        assert Document(str(paths[0])).paragraphs[0].text == "First"

    def test_output_path_column_resolves(self, seed_df: pd.DataFrame, tmp_path: Path) -> None:
        """The stored path is dataset-relative, keeping the dataset portable.

        Uses create() rather than preview() because only DatasetCreationResults
        exposes artifact_storage, which is how a caller resolves the path.
        """
        artifact_path = tmp_path / "artifacts"
        artifact_path.mkdir()

        results = DataDesigner(artifact_path=artifact_path).create(
            self.build(seed_df), num_records=2, dataset_name="documents-test"
        )

        relative = results.load_dataset()["docx_path"].iloc[0]
        assert relative.startswith("documents/word-documents/")
        assert (results.artifact_storage.base_dataset_path / relative).is_file()

    def test_invalid_document_is_skipped(self, tmp_path: Path) -> None:
        """A malformed row yields a null path instead of failing the whole batch."""
        artifact_path = tmp_path / "artifacts"
        artifact_path.mkdir()
        seed_df = pd.DataFrame(
            {
                "doc_id": ["POL-1", "POL-2"],
                "document": [make_document("Good").model_dump_json(), json.dumps({"nope": True})],
            }
        )

        result = DataDesigner(artifact_path=artifact_path).preview(self.build(seed_df), num_records=2)

        assert result.dataset["docx_path"].isna().sum() == 1
        assert len(list(artifact_path.rglob("*.docx"))) == 1
