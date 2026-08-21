# Usage

A complete pipeline: samplers describe the corpus, one structured column writes
each document, and the processor renders them.

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

from data_designer_docx.config import DocxProcessorConfig
from data_designer_docx.schema import WordDocument

MODEL_ALIAS = "doc-writer"

config_builder = dd.DataDesignerConfigBuilder(
    model_configs=[
        dd.ModelConfig(
            alias=MODEL_ALIAS,
            model="nvidia/nemotron-3-super-120b-a12b",
            provider="nvidia",
            # A whole document in one call is a long structured generation.
            inference_parameters=dd.ChatCompletionInferenceParams(max_tokens=8192),
        )
    ]
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="doc_id",
        sampler_type=dd.SamplerType.UUID,
        params=dd.UUIDSamplerParams(prefix="POL-", short_form=True, uppercase=True),
    )
)
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="doc_type",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["Remote Work Policy", "Incident Response Runbook"]),
    )
)

config_builder.add_column(
    dd.LLMStructuredColumnConfig(
        name="document",
        model_alias=MODEL_ALIAS,
        output_format=WordDocument,
        prompt=(
            "Write an internal {{ doc_type }} (document ID {{ doc_id }}). "
            "Write like a real corporate policy: flat, procedural, no marketing language."
        ),
    )
)

config_builder.add_processor(
    DocxProcessorConfig(
        name="word-documents",
        document_column="document",
        filename_template="{{ doc_id }}-{{ doc_type }}.docx",
        metadata_columns={"Document ID": "{{ doc_id }}"},
        footer_template="{{ doc_id }}",
    )
)

results = DataDesigner().create(config_builder, num_records=10, dataset_name="policies")
dataset = results.load_dataset()
```

## Reading the output

`docx_path` is relative to the dataset directory, which keeps the dataset
portable — move the folder and the paths still resolve.

```python
from docx import Document

path = results.artifact_storage.base_dataset_path / dataset["docx_path"].iloc[0]
rendered = Document(str(path))

print(rendered.core_properties.author)
print(rendered.sections[0].footer.paragraphs[0].text)
```

## Customizing the document shape

`WordDocument` describes a title, subtitle, summary, sections, and one key-data
table. To change what gets generated, subclass or replace it and pass your model
as the column's `output_format`; the renderer only requires the fields it reads.

The `Field(description=...)` strings on that model are not documentation. Data
Designer serializes the JSON Schema into the prompt inside `<response_schema>`
tags, so those descriptions are prompt text the model reads — the fastest lever
for changing output quality.

## Rows without a valid document

A row whose document column fails validation is skipped: its `docx_path` is
null, a warning is logged, and the rest of the batch still renders.
