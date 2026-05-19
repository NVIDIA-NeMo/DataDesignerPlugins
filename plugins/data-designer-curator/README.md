# data-designer-curator

NeMo Curator-backed curation plugins for Data Designer.

## Installation

```bash
uv add data-designer data-designer-curator
```

The metadata filter processor calls NeMo Curator's CPU text filtering primitive:

```bash
uv add "data-designer-curator[curator-text-cpu]"
```

The exact dedup processor calls NeMo Curator's GPU dedup workflow. Install
NeMo Curator's `text_cuda12` extra in the same environment for that processor.

For remote HTTP scoring support:

```bash
uv add "data-designer-curator[remote]"
```

## Usage

This package provides three curation plugins:

- `exact-dedup`: call NeMo Curator exact deduplication for generated rows.
- `score-filter`: call NeMo Curator metadata filtering for score thresholds.
- `remote-score`: call an external HTTP scoring endpoint for each row.

```python
import data_designer.config as dd
from data_designer_curator.config import ExactDedupProcessorConfig, ScoreFilterProcessorConfig

builder = dd.DataDesignerConfigBuilder()

# Add generation columns first.

builder.add_processor(
    ExactDedupProcessorConfig(
        name="dedup_answers",
        text_columns=["answer"],
    )
)
builder.add_processor(
    ScoreFilterProcessorConfig(
        name="keep_high_quality",
        score_column="quality_score",
        min_score=0.8,
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
    ExactDedupProcessorConfig,
    RemoteScoreColumnConfig,
    ScoreFilterProcessorConfig,
)

data_designer = DataDesigner()

curation = dd.DataDesignerConfigBuilder()
curation.with_seed_dataset(dd.LocalFileSeedSource(path="raw_seed_rows.parquet"))
curation.add_column(
    RemoteScoreColumnConfig(
        name="quality_score",
        endpoint_url="https://quality.example/score",
        target_columns=["context"],
        score_path="score",
    )
)
curation.add_processor(
    ScoreFilterProcessorConfig(
        name="keep_curated_seeds",
        score_column="quality_score",
        min_score=0.8,
    )
)
curation.add_processor(
    ExactDedupProcessorConfig(
        name="dedup_curated_seeds",
        text_columns=["context"],
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
