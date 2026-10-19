# data-designer-retrieval-sdg

Generate synthetic retrieval data with Data Designer through two pipelines:

- **Text QA:** generate question-answer pairs from text documents, deduplicate
  questions, score quality, and convert the result for retrieval training.
- **Multimodal:** generate queries and graded supporting sources from text,
  images, or mixed collections, then export a portable retrieval bundle.

Text QA retains its existing functionality, CLI, and package-root Python API.
Its implementation lives in `text_qa/`; the new pipeline lives in `multimodal/`.

## Installation

From a repository checkout containing both pipelines, install into an active
Python 3.10 or later environment:

```bash
uv pip install ./plugins/data-designer-retrieval-sdg
```

Released Text QA functionality is available through the NVIDIA-NeMo plugin index.
Use the checkout installation above for the multimodal workflow until a release
includes it:

```bash
uv pip install \
  --default-index https://pypi.org/simple/ \
  --index https://nvidia-nemo.github.io/DataDesignerPlugins/simple/ \
  data-designer-retrieval-sdg
```

For development in this repository, run `make sync` and prefix commands with
`uv run`.

## Choose a Pipeline

Start with the guide for your task:

| Goal | Documentation |
| --- | --- |
| Compare inputs, outputs, and execution models | [Retrieval Synthetic Data Generation](docs/index.md) |
| Generate question-answer pairs from text files | [Generate Text QA Data](docs/text-qa.md) |
| Try multimodal generation with a prepared NVIDIA hardware page | [Create Your First Retrieval Bundle](docs/first-run.md) |
| Generate queries from your own text and images | [Generate Retrieval Data From Text and Images](docs/multimodal.md) |
| Configure multimodal inputs, model roles, and planning | [Multimodal Configuration Reference](docs/multimodal-reference.md) |
| Understand evidence, views, and evaluation limits | [Understand Multimodal Retrieval Data](docs/multimodal-concepts.md) |

Text QA uses `data-designer-retrieval-sdg generate` and `convert`. Multimodal uses
`python -m data_designer_retrieval_sdg.multimodal --config sdg.yaml`.
Both expose Python APIs. The pipelines have different configuration and output
contracts; use the guide for the pipeline you choose.

## Registered Plugins

Installation registers these Data Designer plugins:

| Name | Type | Purpose |
| --- | --- | --- |
| `document-chunker` | Seed reader | Chunk a directory of text files, optionally bundling documents. |
| `embedding-dedup` | Column generator | Deduplicate questions using embedding cosine similarity. |
| `retrieval-structured` | Column generator | Accept one complete bare or JSON-fenced object with schema validation and bounded corrections. |

Source extraction, hard-negative mining, and fine-tuning are separate steps.
Optional sample review against original sources can help improve generation for
your corpus. Model judgments are not independent ground truth.
