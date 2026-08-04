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

Use `--resume if_possible` to resume when compatible artifacts are available and
start fresh otherwise. DataDesigner owns checkpoint discovery, configuration
compatibility, partial-result cleanup, and the behavior of every resume mode. The
plugin does not maintain a second resume state or inspect corpus bytes.

`--buffer-size` controls DataDesigner's checkpoint/write granularity and remains
part of the resolved config. In DataDesigner 0.6.1, `create()` still profiles the
completed dataset before returning, so `--buffer-size` is not a hard cap on final
peak memory for very large runs.

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

## Run configuration

The package contains one complete generation default and one complete conversion
default. No separate model profile is required. Print either declarative resolved
configuration without scanning inputs or starting a run:

```bash
data-designer-retrieval-sdg generate --print-resolved-config
data-designer-retrieval-sdg convert --print-resolved-config
```

Layer a YAML or JSON file over the packaged default with `--config`. Ordinary
CLI flags override that file, and repeatable `--set key=value` entries have final
precedence:

```bash
data-designer-retrieval-sdg generate \
    --config ./generation.yaml \
    --min-complexity 3 \
    --set pipeline.similarity_threshold=0.92
```

A generation file can be intentionally small because omitted values remain
visible through `--print-resolved-config`:

```yaml
schema_version: 1
seed_source:
  path: ./my_documents
output_dir: ./generated_output
artifact_path: ./artifacts
dataset_name: my_retrieval_run
resume: if_possible
num_records: 1000
pipeline:
  num_pairs: 7
```

Relative paths are interpreted from the process working directory. Unknown
fields, unsupported schema versions, and invalid values fail validation. For an
explicit environment-backed provider endpoint or credential, use an exact
`${VARIABLE_NAME}` reference in the config:

```yaml
model_providers:
  - name: nvidia
    provider_type: openai
    endpoint: ${NVIDIA_API_BASE_URL}
    api_key: ${NVIDIA_API_KEY}
```

The environment variable names are recorded as provenance, while credential
values and authorization headers are redacted.

`num_records` limits the first N seed records processed by Data Designer; `null`
processes all available records. This differs from `seed_source.num_files`, which
limits raw files before optional multi-document bundling. `--print-resolved-config`
shows the configured value without scanning the corpus. The persisted run snapshot
replaces `null` with the discovered record count used by that run.

## Quick start

### Generate QA pairs

```bash
data-designer-retrieval-sdg generate \
    --input-dir ./my_documents \
    --output-dir ./generated_output \
    --dataset-name my_retrieval_run \
    --buffer-size 200 \
    --resume if_possible \
    --num-records 1000 \
    --num-pairs 7
```

Generation writes DataDesigner artifacts under `--artifact-path` and exports a
single JSONL file to `--output-dir`. The packaged default uses
`nvidia/nemotron-3-ultra-550b-a55b` for generation and
`nvidia/nemotron-3-embed-1b` for embedding deduplication.

`--query-counts` and `--reasoning-counts` are exact orthogonal
distributions, and each must sum to `--num-pairs`. Pass entries as
`NAME=COUNT` when changing the default of seven pairs.

After DataDesigner generation and JSONL export both complete, the plugin writes:

- `<artifact-path>/.retrieval_sdg_runs/<dataset-name>/resolved_config.yaml`
- `<artifact-path>/.retrieval_sdg_runs/<dataset-name>/config_provenance.json`

The directory uses DataDesigner's resolved dataset name, including any suffix it
adds for a fresh run. The resolved YAML is complete and redacted. Provenance
includes plugin version, config file names and hashes, explicit override paths,
environment variable names, exact output paths, and requested and generated
record counts. Failed attempts do not create plugin metadata; DataDesigner's own
artifacts remain the authority for resuming them.

### Convert to training format

```bash
data-designer-retrieval-sdg convert ./generated_output/my_retrieval_run.jsonl \
    --corpus-id my_corpus
```

Legacy `generated_batch*.json` directories remain supported by `convert`, but a
directory containing more than one generated-data format class is rejected as
ambiguous. Pass the exact JSONL, JSON, or parquet file in that case. `generate`
no longer writes per-batch JSON files. The old manual restart flags
`--batch-size`, `--start-batch-index`, and `--end-batch-index` were removed
because DataDesigner now owns checkpointing through `--buffer-size` and
`--resume`. For very large corpora, keep input partitions sized for
DataDesigner's final profiling step until DataDesigner exposes a no-materialize
create/export path.

After a typed conversion succeeds, it also writes `resolved_config.yaml` and
`config_provenance.json` under `<output-dir>/.retrieval_sdg_run/`. Provenance
records the exact generated paths and output counts. Failed conversions do not
leave plugin metadata that could be mistaken for a completed run.

### Use as a library

```python
from pathlib import Path

from data_designer_retrieval_sdg import (
    ConversionRunConfig,
    load_generation_config,
    run_conversion_with_config,
    run_generation,
)

loaded = load_generation_config(
    Path("./generation.yaml"),
    set_overrides=["pipeline.min_complexity=3"],
)
generation = run_generation(
    loaded.config,
    config_sources=loaded.sources,
    override_paths=loaded.override_paths,
    environment_variables=loaded.environment_variables,
)
conversion = run_conversion_with_config(
    ConversionRunConfig(
        input_path=generation.output_path,
        corpus_id="my_corpus",
    )
)
assert conversion.train_file is not None
```

The generation result contains the exact exported JSONL path, resolved Data
Designer dataset path and name, record count, producer version, run metadata
paths. Conversion returns the generated train, validation, corpus, evaluation,
and run metadata paths plus example counts.
`GenerationRunConfig`, `GenerationPipelineConfig`, and `ConversionRunConfig`
reject unknown fields so recipe adapters cannot silently pass misspelled
settings. Their redacted serialization replaces provider credentials and
authorization headers.

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

Every emitted row includes a normalized corpus-relative `source_id`. Conversion
uses it for chunk lookup so same-basename documents in different directories do
not overwrite one another. Legacy records without `source_id` retain their full
normalized `file_name` as the lookup key.

Output schema (one record per row): `file_name`, `text`, `chunks`,
`sections_structured`, `bundle_id`, `bundle_members`, `is_multi_doc`.
