# Generate Retrieval Data From Text and Images

Use the multimodal workflow to generate retrieval queries paired with supporting
text passages or images. A successful run produces a portable bundle containing
queries, supporting sources with relevance grades, and records for reviewing the
model judgments. You can use the bundle in downstream retrieval training.

This guide assumes you have extracted your source text and prepared local images.
The pipeline generates queries with supporting source units. For generated
question-answer pairs from text documents, use [Text QA](text-qa.md). The two
pipelines have separate configuration and output formats.

For a first run with supplied data, start with
[Create Your First Retrieval Bundle](first-run.md). This guide covers your own
prepared sources.

## Before You Start

You need the following:

- Python 3.10 or later, an active Python environment, and `uv` for installation.
  The installed plugin must contain `data_designer_retrieval_sdg.multimodal`.
- An OpenAI-compatible chat-completions endpoint and credentials for explicitly
  selected generator and judge models. Both models must accept image inputs when
  your corpus contains images.
- A small, representative source collection and a new output directory.

From a repository checkout containing this workflow, install the plugin in your
active Python environment:

```bash
uv pip install ./plugins/data-designer-retrieval-sdg
python -m data_designer_retrieval_sdg.multimodal --help
```

The help output lists the required `--config` argument. Optional semantic context
combinations require the `multimodal` extra; basic text and image generation does
not require that extra. Refer to the
[configuration reference](multimodal-reference.md#related-section-combinations)
when you need it.

## 1. Prepare Source Records

Create a UTF-8 `sources.jsonl` file with one JSON object per retrievable unit.
Each unit needs a unique `unit_id`, a `document_id`, and text, one image, or both.
The following illustrative records show the three supported input shapes:

```jsonl
{"unit_id":"manual-a-text","document_id":"manual-a","text":"The pump supports three operating modes. Eco mode reduces power use, standard mode maintains normal flow, and boost mode temporarily increases flow.","images":[],"language":"en"}
{"unit_id":"manual-a-chart","document_id":"manual-a","text":"","images":["pages/flow-chart.png"],"language":"en"}
{"unit_id":"manual-b-page","document_id":"manual-b","text":"Extracted text from the specifications page.","images":["pages/specifications.png"],"language":"en","page_number":2}
```

Replace the illustrative text and image paths with your own content. Image paths
resolve relative to `sources.jsonl`; the referenced files must exist. For a
text-only corpus, use records with nonempty `text` and an empty `images` list.

Extract PDF text, perform optical character recognition (OCR), and render page
images before this step. The plugin does not ingest PDFs or infer identities
from filenames. Keep IDs stable and make each unit small enough to fit a model
request without splitting its content. Refer to the
[source-record reference](multimodal-reference.md#source-records) for all fields.

Compare extracted tables with the original page before generation. Preserve row
and column headings, and keep footnote markers separate from numeric values.
Extraction errors can propagate into summaries, support explanations, and labels.

Choose instructions and models for the content you supply:

| Corpus | Configuration Choice | Evidence to Review |
| --- | --- | --- |
| Text only | Use the [text-only instructions](#request-only-text-questions); image-capable models are not required. | Text quotations and source relevance. |
| Images only | Use image-capable generator and judge models; request chart, figure, or table tasks that fit the images. | Visual support against the original image. |
| Text and images | Use image-capable models and instructions for the information needs you want. | Whether each query needs text, images, or both, and whether its positives fit the selected view. |

An instruction label requests a modality; it does not prove that the resulting
query uses it. Refer to [requested tasks and observed evidence](multimodal-concepts.md#requested-tasks-and-observed-evidence).

## 2. Configure the Run

Save the following as `sdg.yaml`. Replace the model and endpoint placeholders
with values from your deployment. Paths in YAML resolve from the directory where
you run the command, not from the YAML file's location.

```yaml
sources_file: ./sources.jsonl
output_dir: ./runs/retrieval-data
dataset_id: service-manuals
context_strategy: unit
generator:
  model: <generator-model>
  endpoint: https://api.example.com/v1
  credential_env: SDG_API_KEY
  max_tokens: 8192
judge:
  model: <judge-model>
  endpoint: https://api.example.com/v1
  credential_env: SDG_API_KEY
  max_tokens: 8192
concurrency: 1
batch_size: 1
missing_response_attempts: 1
seed: 42
```

Set `SDG_API_KEY` in your environment using your credential-management process.
The configuration stores the variable name, not its secret value. There is no
fallback model: verify image support, structured-response behavior, context
limits, and output limits for both roles before scaling up.

The initial configuration uses one source unit per context. A context is the
set of units shown together to generate a query. To combine pages or documents,
refer to [context planning](multimodal-reference.md#context-planning).

### Request Only Text Questions

The default sampler requests one text, one figure, and one table task per
context. Unsupported tasks can abstain. To avoid visual tasks on a text-only
corpus, add explicit instructions to `sdg.yaml`:

```yaml
instructions:
  - name: text-facts
    instruction: >-
      Ask a standalone question about an important fact supported by the source
      text. Include the subject in the question. Return null if no suitable
      question is supported.
    modality: text
    answerability: fully answerable
  - name: text-explanation
    instruction: >-
      Ask a standalone question that explains a process or compares concepts
      using only the source text. Return null if the text cannot support it.
    modality: text
    answerability: fully answerable
```

These profiles request two queries per selected context. Their labels guide the
model; they do not enforce complete answerability or guarantee two accepted
queries. Review the resulting evidence as described below. Image-description
requests already skip sources without images.

## 3. Generate and Inspect the Bundle

Run the configuration with the module command:

```bash
python -m data_designer_retrieval_sdg.multimodal --config ./sdg.yaml
```

A successful run prints the path to `bundle/generation_result.json` inside your
output directory. This handoff references the portable bundle manifest. The run
fails without a completed handoff if no candidates pass its quality and evidence
checks.

Before selecting a training view, save the following as `review_bundle.py` in the
directory where you ran generation. It reads `bundle/report.json`:

```python
import json
from pathlib import Path

bundle = Path("runs/retrieval-data/bundle")
report = json.loads((bundle / "report.json").read_text())
print("Generated:", report["candidate_records"])
print("Accepted:", report["accepted_query_ids"])
print("Rejections:", report["rejection_reasons"])
print("Views:", report["views"])
```

Run the script from the same directory:

```bash
python review_bundle.py
```

To refine generation for your corpus, optionally review a sample of accepted and
rejected records in `bundle/candidate_diagnostics.jsonl` against their original
text and images. Check that queries make sense without seeing the source, quoted
text is accurate, and visual explanations identify the correct figure elements.
Judge scores and valid source IDs do not establish factual correctness.
Refer to [sample review](multimodal-concepts.md#refine-generation-with-sample-review).

Select a view that preserves the required evidence. An image attached to a
text-supported source does not make that query suitable for image-only training.
Refer to [evidence and views](multimodal-concepts.md#evidence-and-export-views).
Use `mixed` to retain text-only, image-only, and paired source units together.
The `image_and_text` view includes only units with images, so it can exclude
queries that also require a text-only unit. Confirm that your downstream consumer
supports the selected view. A small run can produce no training records for a
view even when that view has synthetic evaluation queries.

The bundle contains known positives only. Hard-negative mining, fine-tuning,
and model evaluation are downstream steps. Synthetic evaluation queries are
useful for workflow checks; use independently reviewed data to measure retrieval
quality.

## Continue to Mining and Training

Keep the complete `bundle/` directory when moving the result. Its
`generation_result.json` handoff references a manifest and checksummed artifacts;
copying only the handoff file is insufficient.

In a compatible [Nemotron retrieval recipe checkout](https://github.com/NVIDIA-NeMo/Nemotron),
data preparation consumes this handoff through `sdg_input_path` and requires an
explicit `retrieval_view`. For example, the
[preview multimodal preparation profile](https://github.com/NVIDIA-NeMo/Nemotron/blob/3ff736d1d81a73f687c2d725179fef5aa4be3e55/src/nemotron/recipes/embed/stage1_data_prep/config/mistral3-vl.yaml)
uses these fields:

```yaml
sdg_input_path: /absolute/path/to/bundle/generation_result.json
retrieval_view: image_and_text
```

Apply these settings in the recipe configuration, not the plugin configuration.
Use `image_and_text` only when its evidence rules fit your task and it contains
training records. The linked profile requires a compatible multimodal embedding
checkpoint and its own mining environment. Follow the recipe's setup instructions
before running data preparation or fine-tuning. This pinned profile demonstrates
`image_and_text`; it does not establish `mixed` support in mining or training.
Check the view and corpus-field support of the recipe version you use, including
nullable images for text-only units. Plugin export support alone is insufficient.

The single-page tutorial verifies generation and inspection. It does not supply
enough data to assess training quality. Generate a larger dataset that represents
your corpus before mining negatives. Use independent evaluation queries and
reviewed relevance labels when comparing retrieval models.

## Resume an Interrupted Run

Preserve the run directory and inspect the error and saved artifacts. After
resolving an external issue, set `resume: true` in the same configuration and run
the command again. Keep the inputs, remaining configuration, package code, and
dependencies unchanged.

Resume reuses compatible cached responses. A completed run verifies its artifacts
and returns the existing handoff. Changed inputs or corrupted artifacts fail
integrity checks. Bundle export uses a staging directory and publishes only a
complete bundle. An interrupted export can resume from compatible cached work.
If publication finished before the completion record was saved, resume reuses the
bundle only when every file matches the regenerated export. An existing partial
or different destination is not overwritten; preserve it for investigation and
use a new output directory for a separate run.

## Troubleshoot a Small Run

If generation finishes with no accepted queries, inspect `candidates.json` for
query judgments and localized evidence, and `context_outcomes.json` for summary
selection. Both files are in `output_dir`. The files `bundle/report.json` and
`bundle/candidate_diagnostics.jsonl` are not written in this case. If inference
fails earlier, inspect the command error and any saved artifacts under
`inference/attempts/` instead. Missing or invalid required responses stop the
workflow before export; they are not query-quality rejections.

Use the following checks before increasing the corpus size:

| Symptom | What to Check |
| --- | --- |
| No candidates pass | Check the saved selection decisions and judgments before lowering thresholds. |
| Many answer-leakage rejections | Confirm that the judge distinguishes a question that contains its answer from a question that can be answered. Inspect its reasoning. |
| Missing image or provider image error | Check file paths, image readability, and both models' image-input support and limits. |
| A source unit exceeds a request limit | Split it during preprocessing and assign stable IDs to the new units. Context splitting cannot divide one unit. |
| Incomplete structured responses | Inspect the attempt's errors and saved responses. Check output-token limits, including reasoning-token usage, and the endpoint's [JSON formatting support](multimodal-reference.md#model-settings). |
| A training view is empty | Inspect its accepted-query and split counts. Choose an evidence-compatible view with training records; do not substitute another view silently. |
| An existing run is rejected | Use explicit resume only for a compatible run, or choose a new output directory for changed inputs or configuration. |

Batch size and missing-response attempts are not aggregate request, token, or
spending limits. Monitor provider usage and begin with a bounded source sample.
Use an external job time limit when you need a hard elapsed-time bound.
Keep run directories protected because they contain source content and model
responses.
