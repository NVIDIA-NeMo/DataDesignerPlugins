# data-designer-retrieval-sdg

Data Designer toolkit for **retriever synthetic data generation**. The
package registers three `data_designer.plugins` entry points, ships a
ready-made multi-step QA generation pipeline, and exposes a CLI that
generates QA pairs and converts them into training formats compatible
with [Automodel](https://github.com/NVIDIA-NeMo/Automodel) retriever
finetuning.

## Plugins

A single package contributes three plugins to DataDesigner's registries
via `[project.entry-points."data_designer.plugins"]`:

| Slug | Type | Purpose |
|------|------|---------|
| `embedding-dedup` | column generator | Generic cosine-similarity dedup of any list-valued column. Implements native `agenerate()` for the async engine. |
| `document-chunker` | seed reader | Sentence-chunks a directory of text files and emits structured sections, with optional multi-document bundling. |
| `retrieval-structured` | column generator | Native structured generation accepting one complete bare or JSON-fenced object, with schema validation and bounded corrections. |

All are registered automatically through Python entry points when the
package is installed (see [Installation](#installation)).

## Retrieval data from text and images

For new multimodal fine-tuning workflows, use the
[retrieval-first EA candidate](docs/multimodal-ea.md): direct text/image query
generation, query-only answer-leak checks, source-aware quality judging and graded
positive localization, followed by grouped-query export over a shared corpus.
It accepts generic unit/context JSONL and requires explicit generator/judge
models. It has no dataset-specific adapters or external benchmark dependencies.

The existing QA pipeline and conversion CLI remain available for existing
clients. Their [QA input contract](docs/retrieval-inputs.md) is separate from the
new EA entry point. PDF parsing/OCR remain caller-owned preprocessing.

### VLM-based SDG workflow

Vision-language model (VLM) based synthetic data generation (SDG) turns your
source text and images into retrieval queries with graded supporting sources.
The chart follows the Nemotron EA profile's `context_strategy: sections` path.
A **unit** is one retrievable item, such as a page; a **context** is a group of
units shown together when generating a query.

```text
YOUR PREPROCESSING
  Documents / PDFs
        |
        v
  Extract text / OCR and render images
        |
        v
  sources.jsonl: stable unit IDs + document IDs + text and/or images
        |
        v
DATA DESIGNER RETRIEVAL SDG PLUGIN
  +------------------------------------------------------------------+
  | 1. Plan sections and summarize                                   |
  |    Group whole units in reading order; describe visual content   |
  |    and documents, then summarize each section.                   |
  |    Optional contexts.jsonl supplies your own groups of units.    |
  +------------------------------------------------------------------+
        |
        v
  +------------------------------------------------------------------+
  | 2. Add related section combinations (optional)                   |
  |    Embed summary text, cluster related sections, and summarize   |
  |    pairs/triples, including sections from different documents.   |
  +------------------------------------------------------------------+
        |
        v
  +------------------------------------------------------------------+
  | 3. Select contexts for query generation                          |
  |    Judge each summary: all four scores >= 4/5 by default.        |
  |      - Richness: enough useful detail                            |
  |      - Persona relevance: useful to the configured target reader |
  |      - Query potential: supports varied, specific questions      |
  |      - Clarity: concepts are understandable                      |
  |    Remove duplicates and apply the summary-selection budget.     |
  |    Split oversized contexts at whole-unit boundaries.            |
  +------------------------------------------------------------------+
        |
        v
  +------------------------------------------------------------------+
  | 4. Generate candidate queries with the VLM                       |
  |    Read ORIGINAL text + images for each selected context.        |
  |    Vary source scope: one page, multiple pages, or documents.    |
  |    Sample one text, one figure, and one table task by default.   |
  |    Vary task: lookup, explanation, comparison, yes/no, lists,    |
  |    or calculation; combine evidence when appropriate.            |
  |    Vary format: question, instruction, or keyword search.        |
  |    Vary evidence coverage: sources meet all or part of the need. |
  |    Generate queries, not answers.                                |
  +------------------------------------------------------------------+
        |                                  |
        | Candidate query                  +--> Unsupported task:
        v                                       record abstention
  +------------------------------------------------------------------+
  | 5. Judge queries and identify supporting source units            |
  |    Query alone: check for leaked answers.                        |
  |    Query + original text: judge self-sufficiency.                |
  |    Query + original text/images: judge relevance and support.    |
  |    Label support: 2 = complete, 1 = useful partial evidence.     |
  +------------------------------------------------------------------+
        |
        v
  Relevance and self-sufficiency >= 4/5 (default),
  no leaked answer, and valid supporting source units?
        |
        +--- No ---> Keep rejection and judge details for inspection
        |
        Yes
        v
  +------------------------------------------------------------------+
  | 6. Export a portable retrieval bundle                            |
  |    Accepted queries + graded positives + full eligible corpus.   |
  |    Split by query group; package images, training views,         |
  |    evaluation data, and provenance.                              |
  +------------------------------------------------------------------+
        |
        v
NEMOTRON RECIPE (DOWNSTREAM)
  Mine hard negatives -> Fine-tune embedding model -> Evaluate
```

Summaries and visual descriptions guide planning; they never replace original
corpus text or images. Units that are not selected for generation remain in the
exported corpus. The summary embedding model processes text and is separate
from the VLM generator/judge and the embedding model being fine-tuned.

By default, all four summary grades (information richness, persona relevance,
query-generation potential, and conceptual clarity) must reach 4/5. Semantic
combinations are skipped for language groups with fewer than twelve summaries.
See the [multimodal EA guide](docs/multimodal-ea.md) for configuration, quality
gates, audit artifacts, and resumable execution.

### Query variations

Step 4 varies both the evidence available to the generator and the kind of
query requested. Queries should read like standalone searches by someone who
has not seen the source, without referring to "this page" or "the figure".

| Dimension | Variations and how they arise |
|---|---|
| Source scope | A single unit, multiple units from one document, or units from different documents. When units are pages, these correspond to single-page, multi-page, and cross-document contexts. Sections group nearby units; optional semantic combinations join related sections; `contexts_file` lets you supply explicit memberships. A larger context does not require every query to use every page or document. |
| Evidence | Text passages, figures/charts/diagrams, and tables. The generator sees original text and attached images together. Recorded evidence can be text, image, or both; a requested task category is not proof of which modality actually supports the query. Unsupported tasks return an abstention. |
| Reasoning/task | **Extractive:** retrieve a specific fact. **Open-ended:** explain or synthesize concepts. **Compare-contrast:** compare entities. **Boolean:** answer yes/no, potentially through several reasoning steps. **Enumerative:** list items meeting a condition. **Numerical:** calculate from data, beyond simply reading a value. Text/figure pools include open-ended tasks; the table pool includes numerical tasks. |
| Multi-hop | The prompts encourage combining information across pages when appropriate. Multi-hop is also an observed query label, but is not a separately sampled task in the default module pools. Cross-document context alone does not establish multi-hop reasoning. |
| Query format | A natural-language question, an instruction, or a keyword-style search. The default sampler restricts boolean tasks to questions and excludes keyword format for enumerative tasks. |
| Evidence coverage (answerability) | Generate queries whose information need is fully or partly covered by the supplied sources. A source can be a useful retrieval positive even when additional sources are needed. No answers are generated; supporting units receive independently judged relevance grades. Partial coverage still requires valid source support and passing the query-quality gates; missing evidence is not guaranteed to exist elsewhere in the corpus. |

For example, if the sources contain the relevant evidence, queries could include:

- **Single-page / text / extractive:** "What is the maximum operating temperature of the PX-200 pump?"
- **Multi-page / tables / numerical:** "How much did North Division revenue grow between 2023 and 2024?"
- **Cross-document / text and diagrams / comparison:** "Compare the cooling mechanisms used by the PX-200 and PX-300 pumps."

These are illustrative examples, not generated results or required combinations.
With the default instructions, each context gets three requested slots: one
weighted task from each of the text, figure, and table pools. Sampling is
reproducible for a fixed seed and context ID. Abstentions and quality filtering
can reduce the number of accepted queries, so the final dataset need not be
balanced across these dimensions. Requested type/format and observed labels
are recorded; matching the requested style is diagnostic, not an acceptance
gate. Use custom `instructions` and `instructions_per_context` to control the
requested tasks; see the [multimodal EA guide](docs/multimodal-ea.md).

## Native async and resumable generation

This section describes the existing QA `generate` command. The EA workflow has
its own documented immutable request cache and explicit resume policy.

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
plugin does not maintain a second resume state. For canonical
`RetrievalSourcesFile` inputs, it creates content-addressed snapshots of the
ordered source rows and image bytes under the artifact path before handing the
rows to Data Designer. This makes source changes visible to Data Designer's
native configuration fingerprint. The legacy document-chunker path retains its
existing resume behavior.

`--buffer-size` controls DataDesigner's checkpoint/write granularity and remains
part of the resolved config. DataDesigner still profiles the completed dataset
before returning, so `--buffer-size` is not a hard cap on final peak memory for
very large runs.

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

The Pydantic run models contain one complete generation default and one complete
conversion default. No separate model profile is required. Print either
declarative resolved configuration without scanning inputs or starting a run:

```bash
data-designer-retrieval-sdg generate --print-resolved-config
data-designer-retrieval-sdg convert --print-resolved-config
```

Layer a YAML or JSON file over the model defaults with `--config`. Explicit
CLI flags have final precedence. Pydantic Settings generates typed flags for
every config field, including dotted flags for nested values; familiar flat
aliases remain available for common settings:

```bash
data-designer-retrieval-sdg generate \
    --config ./generation.yaml \
    --min-complexity 3 \
    --pipeline.similarity-threshold 0.92
```

The effective order is Pydantic model defaults, user config file, then explicit CLI
values. There is no separate untyped `--set` override language. Complex values
use JSON when passed through their generated flag, for example
`--pipeline.query-counts '{"multi_hop":3,"structural":2,"contextual":2}'`.

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

The environment variable names are recorded as provenance. All custom provider
header values are redacted. Custom provider body values are also redacted except
for the string-valued plugin settings `input_type` and `truncate`.

Provider definitions follow the same layering rule. A provider from the user
config can be updated by the legacy `--custom-provider-*` shorthand when the
alias matches; fields not supplied on the CLI are preserved.

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
single JSONL file to `--output-dir`. The model defaults use
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

loaded = load_generation_config(Path("./generation.yaml"))
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
settings. Their redacted serialization replaces provider credentials, every
custom header value, and non-allowlisted custom body values.

## Plugin configuration examples

### `embedding-dedup` column

```python
from data_designer_retrieval_sdg.config import EmbeddingDedupColumnConfig

config_builder.add_column(
    EmbeddingDedupColumnConfig(
        name="deduplicated_qa_pairs",
        source_column="qa_generation",  # upstream column with the items
        items_key="pairs",  # key under the source column ("None" if the column is already a list)
        text_field="question",  # field on each item to embed
        model_alias="embed",  # registered embedding model alias
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
    multi_doc=False,  # set True for bundle-per-row mode
)
```

Every emitted row includes a normalized corpus-relative `source_id`. Conversion
uses it for chunk lookup so same-basename documents in different directories do
not overwrite one another. Legacy records without `source_id` retain their full
normalized `file_name` as the lookup key.

Output schema (one record per row): `file_name`, `text`, `chunks`,
`sections_structured`, `bundle_id`, `bundle_members`, `is_multi_doc`.
