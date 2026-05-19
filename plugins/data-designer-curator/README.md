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

For the full plugin authoring guide, see the
[main repository docs](https://nvidia-nemo.github.io/DataDesignerPlugins/authoring/).

Plugin documentation for the repository site lives in this package's `docs/`
directory.
