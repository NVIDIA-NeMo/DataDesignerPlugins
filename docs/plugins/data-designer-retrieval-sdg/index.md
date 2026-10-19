# Retrieval Synthetic Data Generation

Generate synthetic data for retrieval with two pipelines built on Data Designer.
**Text QA** generates question-answer (QA) pairs from text documents. **Multimodal**
generates queries with supporting text passages, images, or both.

## Choose a Pipeline

Use the output you need to choose the pipeline:

| Aspect | Text QA | Multimodal |
| --- | --- | --- |
| Input | Directory of text files, with optional document bundling. | Prepared JSON Lines (JSONL) units, or a directory of independent text and image files. |
| Generation | Extract document artifacts, generate question-answer pairs, deduplicate questions, and score quality. | Plan contexts, generate queries, judge relevance, and identify supporting source units. |
| Result | Generated JSONL; convert it to retrieval training and evaluation files. | Portable retrieval bundle with corpus, queries, graded positives, and export views. |
| Command | `data-designer-retrieval-sdg generate`, then `convert`. | `python -m data_designer_retrieval_sdg.multimodal --config sdg.yaml`. |
| Python API | Imports from `data_designer_retrieval_sdg`. | Imports from `data_designer_retrieval_sdg.multimodal`. |

The text QA pipeline retains its existing functionality, configuration, and
package-root API. Its implementation now lives in `text_qa/`. Existing text QA
jobs and recipe integrations can continue using the same commands and imports.

The multimodal pipeline is new functionality. It can also accept a text-only
corpus, but produces query-to-source supervision rather than question-answer
pairs. Its source format, configuration, output, and resume state are separate
from text QA. It does not automatically replace a text QA job.

## Start With Your Task

Choose a guide for the result you want:

| Goal | Guide |
| --- | --- |
| Generate question-answer pairs from text documents | [Generate Text QA Data](text-qa.md) |
| Try multimodal generation with a prepared NVIDIA hardware page | [Create Your First Retrieval Bundle](first-run.md) |
| Use your own text, image, or mixed corpus | [Generate Retrieval Data From Text and Images](multimodal.md) |
| Look up multimodal inputs, model settings, and planning options | [Multimodal Configuration Reference](multimodal-reference.md) |
| Understand evidence, quality checks, views, and evaluation | [Understand Multimodal Retrieval Data](multimodal-concepts.md) |

Neither pipeline performs hard-negative mining or model training. Both pipelines
filter generated data automatically. Optional sample review can help you refine
generation for your corpus. Model-generated scores and labels can contain errors;
use independent evaluation data to measure retrieval quality.

## Data Designer Integration

Text QA builds one four-column Data Designer configuration and uses Data
Designer's generation and resume support. Multimodal coordinates planning,
model-call batches, evidence checks, and bundle export in Python. Each inference
batch uses a registered Data Designer structured-response column.

The package registers three plugins automatically when installed:

| Plugin | Role |
| --- | --- |
| `document-chunker` | Read and chunk text documents for text QA. |
| `embedding-dedup` | Deduplicate generated questions using text embeddings. |
| `retrieval-structured` | Decode and validate multimodal structured responses using Data Designer's generation loop. |

Implementation and tests are organized under `text_qa/` and `multimodal/`.
The multimodal subpackage also owns its source loading, source models, query
grouping, and split assignment.
