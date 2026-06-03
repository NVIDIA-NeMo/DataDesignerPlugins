# data-designer-retrieval-sdg

Data Designer toolkit for **retriever synthetic data generation**. The
package registers two `data_designer.plugins` entry points, ships a
ready-made multi-step QA generation pipeline, and exposes a CLI that
generates QA pairs and converts them into training formats compatible
with [Automodel](https://github.com/NVIDIA-NeMo/Automodel) retriever
finetuning.

## Plugins

A single package contributes two plugins to DataDesigner's registries
via `[project.entry-points."data_designer.plugins"]`:

| Slug | Type | Purpose |
|------|------|---------|
| `embedding-dedup` | column generator | Generic cosine-similarity dedup of any list-valued column. Implements native `agenerate()` for the async engine. |
| `document-chunker` | seed reader | Sentence-chunks a directory of text files and emits structured sections, with optional multi-document bundling. |

Both are registered automatically through Python entry points when the
package is installed (see [Installation](#installation)).

## Native async and resumable generation

`embedding-dedup` implements `agenerate()` directly on top of
`model.agenerate_text_embeddings`, so the column participates in
DataDesigner's async cell-level scheduler.

The `generate` command uses DataDesigner's native resumable generation.
Use a stable `--artifact-path`, `--dataset-name`, and `--buffer-size`, then
resume an interrupted run with `--resume always`:

```bash
data-designer-retrieval-sdg generate \
    --input-dir ./my_documents \
    --output-dir ./generated_output \
    --dataset-name my_retrieval_run \
    --buffer-size 200 \
    --resume always
```

Use `--resume if_possible` to resume only when the saved config matches and
otherwise start a fresh run.

`--buffer-size` controls DataDesigner's checkpoint/write granularity and must
match across resumed runs. In DataDesigner 0.6.1, `create()` still profiles the
completed dataset before returning, so `--buffer-size` is not a hard cap on
final peak memory for very large runs.

## Installation

The package is distributed from the NVIDIA-NeMo plugin index (hosted on
GitHub Pages); it is not on PyPI. Install it by adding the plugin index
alongside PyPI:

```bash
uv pip install \
  --default-index https://pypi.org/simple/ \
  --index https://nvidia-nemo.github.io/DataDesignerPlugins/simple/ \
  data-designer-retrieval-sdg
```

For projects managed with `uv`, add it as a dependency:

```bash
uv add \
  --default-index https://pypi.org/simple/ \
  --index https://nvidia-nemo.github.io/DataDesignerPlugins/simple/ \
  data-designer-retrieval-sdg
```

`pip` users can pass the equivalent flags:

```bash
pip install \
  --index-url https://pypi.org/simple/ \
  --extra-index-url https://nvidia-nemo.github.io/DataDesignerPlugins/simple/ \
  data-designer-retrieval-sdg
```

Standard version constraints work (`>=0.1`, `==0.1.0`, ...). The
NVIDIA-NeMo index only serves `data-designer-*` plugin packages; the
default PyPI index supplies transitive dependencies (`data-designer`,
`nltk`, `pyarrow`, `pyyaml`).

For development inside the monorepo:

```bash
make sync                     # install all packages into .venv
source .venv/bin/activate     # activate the virtual environment
```

Or prefix any command with `uv run`:

```bash
uv run data-designer-retrieval-sdg generate --help
```

## Quick start

### Generate QA pairs

```bash
data-designer-retrieval-sdg generate \
    --input-dir ./my_documents \
    --output-dir ./generated_output \
    --dataset-name my_retrieval_run \
    --buffer-size 200 \
    --resume if_possible \
    --num-pairs 7
```

Generation writes DataDesigner artifacts under `--artifact-path` and exports a
single JSONL file to `--output-dir`.

### Convert to training format

```bash
data-designer-retrieval-sdg convert ./generated_output/my_retrieval_run.jsonl \
    --corpus-id my_corpus
```

Legacy `generated_batch*.json` directories remain supported by `convert`, but
`generate` no longer writes per-batch JSON files. The old manual restart flags
`--batch-size`, `--start-batch-index`, and `--end-batch-index` were removed
because DataDesigner now owns checkpointing through `--buffer-size` and
`--resume`. For very large corpora, keep input partitions sized for
DataDesigner's final profiling step until DataDesigner exposes a no-materialize
create/export path.

### Use as a library

```python
from data_designer_retrieval_sdg import (
    DocumentChunkerSeedSource,
    build_qa_generation_pipeline,
)

seed_source = DocumentChunkerSeedSource(
    path="./docs",
    file_extensions=[".txt", ".md"],
)
config_builder = build_qa_generation_pipeline(seed_source)
```

## Plugin configuration examples

### `embedding-dedup` column

```python
from data_designer_retrieval_sdg.config import EmbeddingDedupColumnConfig

config_builder.add_column(
    EmbeddingDedupColumnConfig(
        name="deduplicated_qa_pairs",
        source_column="qa_generation",   # upstream column with the items
        items_key="pairs",               # key under the source column ("None" if the column is already a list)
        text_field="question",           # field on each item to embed
        model_alias="embed",             # registered embedding model alias
        similarity_threshold=0.9,
    )
)
```

### `document-chunker` seed reader

```python
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource

seed_source = DocumentChunkerSeedSource(
    path="./docs",
    file_pattern="*",
    recursive=True,
    file_extensions=[".txt", ".md"],
    sentences_per_chunk=5,
    num_sections=1,
    multi_doc=False,                # set True for bundle-per-row mode
)
```

Output schema (one record per row): `file_name`, `text`, `chunks`,
`sections_structured`, `bundle_id`, `bundle_members`, `is_multi_doc`.
