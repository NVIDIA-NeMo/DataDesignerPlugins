# data-designer-retrieval-sdg

Data Designer toolkit for **retriever synthetic data generation**. The
package registers two `data_designer.plugins` entry points, ships a
ready-made multi-step QA generation pipeline, and exposes a CLI that
generates QA pairs and converts them into training formats compatible
with [Automodel](https://github.com/NVIDIA-NeMo/Automodel) retriever
finetuning.

## Plugins

The single PyPI package contributes two plugins to DataDesigner's
registries via `[project.entry-points."data_designer.plugins"]`:

| Slug | Type | Purpose |
|------|------|---------|
| `embedding-dedup` | column generator | Generic cosine-similarity dedup of any list-valued column. Implements native `agenerate()` for the async engine. |
| `document-chunker` | seed reader | Sentence-chunks a directory of text files and emits structured sections, with optional multi-document bundling. |

Both ship with the same `pip install data-designer-retrieval-sdg` and
become discoverable automatically through Python entry points.

## Recipe

When installed with a Data Designer version that supports recipes, this
package also registers a `retrieval-sdg` recipe through
`[project.entry-points."data_designer.recipes"]`. The recipe lets users run
the opinionated end-to-end pipeline from the main Data Designer CLI:

```bash
data-designer run-recipe retrieval-sdg --config retrieval-sdg.yaml --num-records 200
```

Minimal recipe config:

```yaml
input_dir: ./my_documents
generated_output_dir: ./generated_output
corpus_id: my_corpus
```

Inspect the full config schema with:

```bash
data-designer recipes show retrieval-sdg
```

## Native async (`DATA_DESIGNER_ASYNC_ENGINE=1`)

`embedding-dedup` implements `agenerate()` directly on top of
`model.agenerate_text_embeddings`, so the column participates in
DataDesigner's async cell-level scheduler whenever the env var is set:

```bash
export DATA_DESIGNER_ASYNC_ENGINE=1
data-designer-retrieval-sdg generate ...
```

The async engine requires Python 3.11+; without the env var the package
runs on Python 3.10+ via the framework's sync bridge.

## Installation

```bash
pip install data-designer-retrieval-sdg
```

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
    --num-pairs 7
```

### Convert to training format

```bash
data-designer-retrieval-sdg convert ./generated_output \
    --corpus-id my_corpus
```

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
