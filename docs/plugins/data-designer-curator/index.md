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

## Plugins

| Entry point | Type | Purpose |
| --- | --- | --- |
| `exact-dedup` | Processor | Drop exact duplicate rows by one or more columns. |
| `score-filter` | Processor | Keep rows that pass score thresholds. |
| `remote-score` | Column generator | Call an external HTTP scoring endpoint for each row. |

## Exact Dedup

```python
from data_designer_curator.config import ExactDedupProcessorConfig

builder.add_processor(
    ExactDedupProcessorConfig(
        name="dedup_answers",
        text_columns=["answer"],
        keep="first",
    )
)
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Processor name used for artifacts. |
| `text_columns` | Yes | Columns used to identify exact duplicates. |
| `keep` | No | Representative policy: `first`, `last`, `highest_score`, or `lowest_score`. |
| `score_column` | No | Score column required for score-based `keep` policies. |
| `audit` | No | Write an audit parquet file under processor artifacts. |

## Score Filter

```python
from data_designer_curator.config import ScoreFilterProcessorConfig

builder.add_processor(
    ScoreFilterProcessorConfig(
        name="keep_high_quality",
        score_column="quality_score",
        min_score=0.8,
    )
)
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Processor name used for artifacts. |
| `score_column` | Yes | Existing column containing numeric scores. |
| `min_score` | No | Inclusive lower bound. |
| `max_score` | No | Inclusive upper bound. |
| `keep_null_scores` | No | Keep rows whose score is null. |
| `audit` | No | Write an audit parquet file under processor artifacts. |

## Remote Score

```python
from data_designer_curator.config import RemoteScoreColumnConfig

builder.add_column(
    RemoteScoreColumnConfig(
        name="quality_score",
        endpoint_url="https://quality.internal/score",
        target_columns=["question", "answer"],
        score_path="score",
        side_effect_output_column="quality_metadata",
    )
)
```

Endpoint requests use this shape:

```json
{
  "data": [
    {"question": "...", "answer": "..."}
  ]
}
```

Endpoint responses must include a non-empty `data` list:

```json
{
  "data": [
    {"score": 0.93, "label": "high_quality"}
  ]
}
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Output score column. |
| `endpoint_url` | Yes | HTTP endpoint called with one row at a time. |
| `target_columns` | Yes | Input columns sent in the request. |
| `timeout_seconds` | No | Request timeout. |
| `headers` | No | Static request headers. |
| `score_path` | No | Dot path to read from the first response item. |
| `side_effect_output_column` | No | Optional column containing the full response item. |

## Audit Artifacts

Row-dropping processors write `audit.parquet` when `audit=True`:

```text
artifacts/<dataset-name>/processors-files/<processor-name>/audit.parquet
```

The audit includes original row index, processor name, action, reason, group ID,
and representative index.
