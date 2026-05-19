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

The exact dedup processor calls NeMo Curator's GPU dedup workflow. Install
NeMo Curator's `text_cuda12` extra in the same environment for that processor.

## Usage

This package provides three curation plugins:

- `exact-dedup`: call NeMo Curator exact deduplication for generated rows.
- `curator-modify`: apply a chain of Curator text modifier primitives.
- `curator-text-filter`: apply a chain of Curator document filter primitives.

```python
import data_designer.config as dd
from data_designer_curator.config import (
    CuratorModifierConfig,
    CuratorModifyProcessorConfig,
    CuratorTextFilterConfig,
    CuratorTextFilterProcessorConfig,
    ExactDedupProcessorConfig,
)

builder = dd.DataDesignerConfigBuilder()

# Add generation columns first.

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
builder.add_processor(
    ExactDedupProcessorConfig(
        name="dedup_answers",
        text_columns=["answer"],
    )
)
```

### Workflow chaining example

Curator processors can also be used as a seed-preparation stage for a later SDG
workflow. The first workflow reads raw seeds, adds curation metadata, and runs
Curator postprocessors. The next workflow reads the curated final dataset as its
seed source.

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer_curator.config import (
    CuratorModifierConfig,
    CuratorModifyProcessorConfig,
    CuratorTextFilterConfig,
    CuratorTextFilterProcessorConfig,
    ExactDedupProcessorConfig,
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
    ExactDedupProcessorConfig(
        name="dedup_curated_seeds",
        text_columns=["clean_context"],
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

For the full plugin authoring guide, see the
[main repository docs](https://nvidia-nemo.github.io/DataDesignerPlugins/authoring/).

Plugin documentation for the repository site lives in this package's `docs/`
directory.
