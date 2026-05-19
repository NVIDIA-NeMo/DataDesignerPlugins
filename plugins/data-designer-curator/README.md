# data-designer-curator

Curator-style curation plugins for Data Designer.

## Installation

```bash
uv add data-designer data-designer-curator
```

For remote HTTP scoring support:

```bash
uv add "data-designer-curator[remote]"
```

## Usage

This package provides three lightweight curation plugins:

- `exact-dedup`: drop exact duplicate rows by one or more columns.
- `score-filter`: keep rows that pass score thresholds.
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
        keep="first",
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
