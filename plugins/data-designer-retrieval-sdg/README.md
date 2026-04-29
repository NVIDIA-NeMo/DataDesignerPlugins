# data-designer-retrieval-sdg

Data Designer plugin for **retriever synthetic data generation**. Generates
multi-hop QA pairs from text documents and converts them into training
formats compatible with [Automodel](https://github.com/NVIDIA-NeMo/Automodel)
retriever finetuning.

## Features

- **Retrieval-sdg-dedup column plugin** — embedding-based QA-pair deduplication
  registered as a `data_designer.plugins` entry point.
- **Four-column SDG pipeline** — artifact extraction → QA generation →
  deduplication → quality evaluation, all orchestrated via DataDesigner.
- **Data conversion** — convert raw SDG output to NeMo Retriever training
  format (`train.json`, `val.json`), BEIR evaluation format, and corpus
  parquet with `merlin_metadata.json`.
- **CLI** — `data-designer-retrieval-sdg generate` and
  `data-designer-retrieval-sdg convert` subcommands.

## Installation

```bash
pip install data-designer-retrieval-sdg
```

Or, for development inside the monorepo:

```bash
make sync   # from the repo root
```

## Development setup

When working inside the monorepo the CLI and library are installed into the
workspace virtual environment. Activate it before running commands:

```bash
make sync                     # install all packages into .venv
source .venv/bin/activate     # activate the virtual environment
```

Alternatively, prefix any command with `uv run` to execute inside the venv
without activating it:

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
    build_qa_generation_pipeline,
    load_text_files_from_directory,
)

seed_df = load_text_files_from_directory(Path("./docs"))
config_builder = build_qa_generation_pipeline(seed_df)
```

## Plugin column type

The package registers the `retrieval-sdg-dedup` column type. Use it in a
DataDesigner pipeline to deduplicate QA pairs by embedding cosine
similarity:

```python
from data_designer_retrieval_sdg.config import RetrievalSdgDedupColumnConfig

config_builder.add_column(
    RetrievalSdgDedupColumnConfig(
        name="deduplicated_qa_pairs",
        qa_pairs_column="qa_generation",
        embedding_alias="embed",
        dedupe_similarity_threshold=0.9,
    )
)
```
