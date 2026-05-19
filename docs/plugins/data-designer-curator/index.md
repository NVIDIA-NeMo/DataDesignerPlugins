# data-designer-curator

NeMo Curator-backed curation plugins for Data Designer.

## Installation

```bash
uv add data-designer data-designer-curator
```

The text modifier and filter processors call NeMo Curator CPU text primitives:

```bash
uv add "data-designer-curator[curator-text-cpu]"
```

The dedup processor calls NeMo Curator's GPU dedup workflows. Install NeMo
Curator's `text_cuda12` extra in the same environment for that processor.

## Plugins

| Entry point | Type | Purpose |
| --- | --- | --- |
| `curator-dedup` | Processor | Call NeMo Curator deduplication workflows for generated rows. |
| `curator-modify` | Processor | Apply a chain of Curator text modifier primitives. |
| `curator-text-filter` | Processor | Apply a chain of Curator document filter primitives. |

## Workflow Chaining

Curator processors can run as a seed-preparation stage for a later SDG workflow.
The first workflow reads raw seeds, adds curation metadata, and runs Curator
postprocessors. The next workflow reads the curated final dataset as its seed
source.

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer_curator.config import (
    CuratorModifierConfig,
    CuratorModifyProcessorConfig,
    CuratorDedupProcessorConfig,
    CuratorTextFilterConfig,
    CuratorTextFilterProcessorConfig,
)

data_designer = DataDesigner()

curation = dd.DataDesignerConfigBuilder()
curation.with_seed_dataset(dd.LocalFileSeedSource(path="raw_seed_rows.parquet"))
curation.add_processor(
    CuratorModifyProcessorConfig(
        name="clean_seed_context",
        input_field="context",
        output_field="clean_context",
        modifiers=[
            CuratorModifierConfig(primitive="unicode_reformatter"),
            CuratorModifierConfig(primitive="markdown_remover"),
            CuratorModifierConfig(primitive="url_remover"),
        ],
    )
)
curation.add_processor(
    CuratorTextFilterProcessorConfig(
        name="keep_curated_seeds",
        text_field="clean_context",
        filters=[
            CuratorTextFilterConfig(
                primitive="word_count",
                params={"min_words": 20, "max_words": 500},
                score_field="curator_word_count",
            )
        ],
    )
)
curation.add_processor(
    CuratorDedupProcessorConfig(
        name="dedup_curated_seeds",
        dedup_type="fuzzy",
        text_columns=["clean_context"],
        params={"char_ngrams": 24},
    )
)

curated = data_designer.create(
    curation,
    num_records=10_000,
    dataset_name="curated_seed_stage",
)

curated_seed_path = curated.artifact_storage.final_dataset_path / "*.parquet"

generation = dd.DataDesignerConfigBuilder()
generation.with_seed_dataset(dd.LocalFileSeedSource(path=str(curated_seed_path)))

# Add SDG columns that use the curated seed columns.
```

After-generation processors update the stage's final `parquet-files` output.
Processor audit artifacts remain under `processors-files/<processor-name>/`.

## Curator Modify

```python
from data_designer_curator.config import CuratorModifierConfig, CuratorModifyProcessorConfig

builder.add_processor(
    CuratorModifyProcessorConfig(
        name="clean_answers",
        input_field="answer",
        modifiers=[
            CuratorModifierConfig(primitive="unicode_reformatter"),
            CuratorModifierConfig(primitive="markdown_remover"),
        ],
    )
)
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Processor name used for artifacts. |
| `input_field` | Yes | Text column passed to Curator modifiers. |
| `modifiers` | Yes | Ordered Curator modifier primitive specs. |
| `output_field` | No | Optional output column. Defaults to modifying `input_field` in place. |

Supported modifier primitives include `unicode_reformatter`, `markdown_remover`,
`url_remover`, `newline_normalizer`, `line_remover`, `quotation_remover`,
`slicer`, and `boilerplate_string`.

## Curator Text Filter

```python
from data_designer_curator.config import CuratorTextFilterConfig, CuratorTextFilterProcessorConfig

builder.add_processor(
    CuratorTextFilterProcessorConfig(
        name="keep_prompt_sized_answers",
        text_field="answer",
        filters=[
            CuratorTextFilterConfig(
                primitive="word_count",
                params={"min_words": 20, "max_words": 500},
                score_field="curator_word_count",
            )
        ],
    )
)
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Processor name used for artifacts. |
| `text_field` | Yes | Default text column passed to Curator filters. |
| `filters` | Yes | Ordered Curator document filter primitive specs. |
| `audit` | No | Write an audit parquet file under processor artifacts. |

Each filter spec has `primitive`, optional constructor `params`, optional
per-filter `text_field`, optional `score_field`, and `invert`.

Supported filter primitives include `word_count`, `mean_word_length`,
`long_word`, `urls`, `numbers`, `non_alpha_numeric`, `whitespace`,
`common_english_words`, repetition filters, token filters, and code filters.

## Curator Dedup

```python
from data_designer_curator.config import CuratorDedupProcessorConfig

builder.add_processor(
    CuratorDedupProcessorConfig(
        name="dedup_answers",
        dedup_type="exact",
        text_columns=["answer"],
    )
)
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Processor name used for artifacts. |
| `dedup_type` | No | Curator dedup workflow: `exact`, `fuzzy`, or `semantic`. Defaults to `exact`. |
| `text_columns` | Yes | Columns used to identify duplicates. |
| `id_column` | No | Existing stable ID column to pass to Curator. |
| `params` | No | Extra keyword arguments passed to the selected Curator workflow. |
| `cache_dir` | No | Curator cache directory. Defaults under processor artifacts. |
| `execution` | No | Optional Curator/Ray execution settings. Defaults to a local Ray client. |
| `audit` | No | Write an audit parquet file under processor artifacts. |

## Audit Artifacts

Row-dropping processors write `audit.parquet` when `audit=True`:

```text
artifacts/<dataset-name>/processors-files/<processor-name>/audit.parquet
```

The audit includes original row index, processor name, action, reason, group ID,
and representative index.
