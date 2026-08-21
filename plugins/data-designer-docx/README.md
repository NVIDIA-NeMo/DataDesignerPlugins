# data-designer-docx

Render Data Designer rows as Microsoft Word documents.

Data Designer generates rows; document pipelines consume `.docx`. This plugin
closes that gap with a processor that writes one Word document per row — with
headings, tables, a front-matter metadata table, footers, and Word core
properties — while keeping the file path joined to its row in the dataset.

## Installation

```bash
uv add data-designer data-designer-docx
```

## Usage

```python
import data_designer.config as dd
from data_designer_docx.config import DocxProcessorConfig
from data_designer_docx.schema import WordDocument

config_builder.add_column(
    dd.LLMStructuredColumnConfig(
        name="document",
        model_alias="doc-writer",
        output_format=WordDocument,
        prompt="Write an internal {{ doc_type }} for {{ company }}.",
    )
)

config_builder.add_processor(
    DocxProcessorConfig(
        name="word-documents",
        document_column="document",
        filename_template="{{ doc_id }}-{{ doc_type }}.docx",
        metadata_columns={"Document ID": "{{ doc_id }}", "Company": "{{ company }}"},
        footer_template="{{ company }} · {{ doc_id }}",
    )
)
```

Files land in `<artifact_path>/<dataset>/documents/word-documents/`, and the
relative path of each one is written back into the dataset as `docx_path`.

See [`docs/`](docs/) for the full field reference and design notes.
