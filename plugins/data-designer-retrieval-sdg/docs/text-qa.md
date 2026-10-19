# Generate Text QA Data

Use the text QA pipeline to generate question-answer pairs from text documents
and convert accepted pairs into retrieval training and evaluation files.
This is the existing text pipeline: its functionality, CLI, configuration, and
package-root Python API are unchanged by the addition of multimodal generation.

For queries grounded in text and images, use the separate
[multimodal pipeline](multimodal.md). Its source JSONL and portable bundle are
different from the inputs and outputs described here.

## Before You Start

You need Python 3.10 or later, an active Python environment, a directory of text
files, and credentials for your selected chat and embedding models. Install from
the repository root:

```bash
uv pip install ./plugins/data-designer-retrieval-sdg
data-designer-retrieval-sdg generate --help
```

The default chat model is `nvidia/nemotron-3-ultra-550b-a55b`; embedding
deduplication defaults to `nvidia/nemotron-3-embed-1b`. Set `NVIDIA_API_KEY` through
your credential-management process. Choose other models or providers through the
configuration when those defaults are not available in your deployment.

Prepare text files before generation. The default file extensions are `.txt`,
`.md`, and `.text`, and the default minimum text length is 50 characters. PDF
extraction and optical character recognition (OCR) are preprocessing steps;
this pipeline does not pass images to models.

## 1. Configure a Small Run

Save the following as `generation.yaml` and replace the input directory with your
prepared corpus:

```yaml
schema_version: 1
seed_source:
  path: ./my_documents
output_dir: ./generated_output
artifact_path: ./artifacts
dataset_name: my_retrieval_run
num_records: 10
resume: never
pipeline:
  num_pairs: 7
```

Paths resolve from your working directory. `num_records` limits the seed records
processed; `null` processes all available records. `seed_source.num_files` limits
raw files before optional document bundling. Neither field is a total API-call,
token, or spending limit.

Print the complete resolved configuration before making model calls:

```bash
data-designer-retrieval-sdg generate \
  --config ./generation.yaml \
  --print-resolved-config
```

The command validates configuration without scanning the corpus or starting
inference. Model defaults are overridden by the configuration file, then by
explicit CLI options. Unknown fields and unsupported schema versions are rejected.

To use a custom OpenAI-compatible provider, add a provider definition to the same
file. The placeholders below read environment variables:

```yaml
model_providers:
  - name: nvidia
    provider_type: openai
    endpoint: ${NVIDIA_API_BASE_URL}
    api_key: ${NVIDIA_API_KEY}
```

Set both variables before loading that configuration. Provider credentials and
custom header values are redacted in saved configuration. Environment variable
names are recorded as provenance.

## 2. Generate Question-Answer Pairs

Run the configuration:

```bash
data-designer-retrieval-sdg generate --config ./generation.yaml
```

The pipeline adds four Data Designer columns in order:

| Column | Operation |
| --- | --- |
| `document_artifacts` | Extract structured artifacts from document sections. |
| `qa_generation` | Generate questions and answers from sections and extracted artifacts. |
| `deduplicated_qa_pairs` | Deduplicate questions using embedding similarity. |
| `qa_evaluations` | Score the generated pairs for quality. |

Generation writes Data Designer artifacts under `artifact_path` and a JSONL file
under `output_dir`. Use the output path reported by the command: Data Designer can
add a suffix to the dataset name for a fresh run. Scoring does not establish that
every answer is correct; inspect representative pairs against the source text.

After generation and export complete, the plugin also writes these files under
`<artifact-path>/.retrieval_sdg_runs/<resolved-dataset-name>/`:

- `resolved_config.yaml`: complete, redacted configuration for the run.
- `config_provenance.json`: version, input config hashes, overrides, environment
  variable names, output paths, and record counts.

Failed runs do not produce these completion metadata files. Data Designer's own
artifacts are used for resume.

## 3. Convert the Generated Data

Pass the exact exported JSONL path to `convert`:

```bash
data-designer-retrieval-sdg convert ./generated_output/my_retrieval_run.jsonl \
  --corpus-id my_corpus \
  --output-dir ./retrieval_data
```

Conversion filters pairs by their quality scores and writes retrieval data,
corpus files, and evaluation artifacts. The default quality threshold is `7.0`;
the default split ratios are `train_ratio: 0.8` and `val_ratio: 0.1`.
Review output counts and split contents before downstream mining or training.
Generated evaluation labels are synthetic and do not establish retrieval quality.

A successful typed conversion also writes `resolved_config.yaml` and
`config_provenance.json` under `<output-dir>/.retrieval_sdg_run/`. These record
the output paths and counts. Use the CLI reference for all conversion options:

```bash
data-designer-retrieval-sdg convert --help
data-designer-retrieval-sdg convert --print-resolved-config
```

Conversion also accepts supported JSON and Parquet exports and legacy
`generated_batch*.json` directories. If a directory contains more than one
generated-data format, specify the exact input file to remove the ambiguity.

## Resume a Text QA Run

Use the same inputs, `artifact_path`, `dataset_name`, and pipeline configuration,
then request native Data Designer resume:

```bash
data-designer-retrieval-sdg generate \
  --config ./generation.yaml \
  --resume always
```

`resume: if_possible` resumes when compatible artifacts are available and starts
a fresh run otherwise. Data Designer owns checkpoint discovery, compatibility
checks, and partial-result cleanup. `buffer_size` controls its write granularity;
it is not a hard bound on peak memory during final profiling.

Multimodal's `resume: true` and request cache apply only to the
[multimodal pipeline](multimodal.md#resume-an-interrupted-run).

## Use the Python API

Existing integrations continue importing from the package root. This example
uses the same configuration and the returned JSONL path for conversion:

```python
from pathlib import Path

from data_designer_retrieval_sdg import (
    ConversionRunConfig,
    load_generation_config,
    run_conversion_with_config,
    run_generation,
)

loaded = load_generation_config(Path("generation.yaml"))
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
        output_dir=Path("retrieval_data"),
    )
)
print(generation.output_path)
print(conversion.train_file)
```

`GenerationRunConfig`, `GenerationPipelineConfig`, `DocumentChunkerSeedSource`,
`build_model_providers`, `build_qa_generation_pipeline`, and `preview_generation`
are also available from the package root. The implementation is organized under
`data_designer_retrieval_sdg.text_qa`; recipe callers do not need to change their
package-root imports.

## Adjust Text Generation

Use `generate --help` and `--print-resolved-config` for the full field list.
These options cover common adjustments:

| Goal | Configuration |
| --- | --- |
| Change chunking | Set `seed_source.sentences_per_chunk` and `seed_source.num_sections`. |
| Bundle documents | Set `seed_source.multi_doc`, `bundle_size`, and `bundle_strategy`. |
| Change pair counts | Set `pipeline.num_pairs`, `pipeline.query_counts`, and `pipeline.reasoning_counts`; each distribution must retain its required keys and sum to `num_pairs`. |
| Change deduplication | Set `pipeline.similarity_threshold`. |
| Select model roles | Set the `pipeline` model/provider fields for artifact extraction, QA generation, quality judgment, and embeddings. |

Explicit CLI options override file values, including typed nested options such
as `--pipeline.similarity-threshold 0.92`. Keep a small representative run for
review before scaling the corpus.
