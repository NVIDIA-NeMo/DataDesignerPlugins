# Multimodal retrieval SDG — early-access candidate

This is the retrieval-first entry point for new text, image, and mixed-source
workflows. It is a generic producer for the Nemotron multimodal fine-tuning
recipe and AutoModel's native retrieval datasets/miner. It has no benchmark
loaders, dataset-name switches, fixed domains, corpus-specific prompts, section
repairs, or dependency on an external generation repository.

The older `generate` / `run_generation` QA interface remains available for
existing clients; it is **not** the multimodal EA recipe entry point. This path
does not generate answers or apply the older answer-comparison policy.

## Inputs and ownership

Supply canonical source JSONL. Each independently retrievable unit has a unique
opaque `unit_id`, a `document_id`, text and/or one local image. Relative image
paths resolve against the source file. IDs must already be globally unique.
PDF parsing, OCR, page naming, domain mapping and dataset ingestion are caller
preprocessing responsibilities. The plugin never reconstructs sections from
page-delimiter markup.

```json
{"unit_id":"manual-a-p1","document_id":"manual-a","text":"Extracted page text.","images":["pages/a-1.png"],"language":"en"}
{"unit_id":"manual-b-p2","document_id":"manual-b","text":"Related specifications.","images":["pages/b-2.png"],"language":"en"}
```

By default, each unit is one generation context. For multi-page or
cross-document questions, optionally supply contexts JSONL:

```json
{"context_id":"compare-specifications","unit_ids":["manual-a-p1","manual-b-p2"],"language":"en"}
```

Context membership controls which evidence a model sees. It is **not** a query
split group. Unselected units still belong to the full corpus. Oversized or
unknown-unit contexts fail without silently skipping or truncating evidence.
Prepare bounded contexts before running; the default maximum is eight units.
Provider image/token limits may require smaller contexts. There is no hidden
automatic semantic clustering or section-selection pass.

## Run

Install the candidate from its reviewed public commit before using this API;
the existing published package does not yet provide it. No core-package patch
is needed: the registered `retrieval-structured` column uses released Data
Designer and retains schema validation and native bounded corrections.

Save an operator-owned YAML configuration:

```yaml
sources_file: /absolute/path/to/sources.jsonl
contexts_file: /absolute/path/to/contexts.jsonl  # omit for one context per unit
output_dir: /absolute/path/to/new-sdg-run
dataset_id: service-manuals
generator:
  model: your-image-capable-generator
  endpoint: https://your-provider.example/v1
  credential_env: SDG_API_KEY
judge:
  model: your-image-capable-judge
  endpoint: https://your-provider.example/v1
  credential_env: SDG_API_KEY
concurrency: 8
batch_size: 30
max_units_per_context: 8
ratios: {train: 0.8, validation: 0.0, evaluation: 0.2}
seed: 42
resume: false
```

Set the named credential environment variable without placing its value in
YAML. Model identifiers are explicit; there is no hosted-model fallback. Paths
in YAML resolve from the working directory; only source-image paths resolve
from the source file.

```bash
python -m data_designer_retrieval_sdg.multimodal --config /path/to/sdg.yaml
```

Equivalent Python API:

```python
from data_designer_retrieval_sdg.multimodal import MultimodalSDGConfig, run_multimodal_sdg

# settings is the mapping loaded from the YAML above.
handoff = run_multimodal_sdg(MultimodalSDGConfig.model_validate(settings))
```

## Pipeline and gates

1. Summarize each bounded context's text and visual evidence. These summaries
   aid generation but never replace original evidence or determine positives.
2. Generate queries directly from the original text **and attached pixels**.
   The default three slots request text, figure and table information needs.
   Unsupported slots explicitly abstain. Customize `instructions` with a list
   of `{name, instruction}` objects; there is no fixed three-slot schema.
3. Independently judge query self-sufficiency and answer leakage **without**
   sources. Independently judge relevance using the actual source text/images.
4. Localize every useful positive in the context. Grade 2 means complete
   support; grade 1 means useful partial evidence. Multi-part questions can
   have multiple partial positives. Text quotes must match supplied text;
   visual evidence requires an image and a description, not an invented quote.

Acceptance requires relevance and self-sufficiency scores at least 4/5, no
answer leakage, and valid nonempty localized support. Both score thresholds
are explicit configuration. Requested style/type is diagnostic, not a gate.
Observed query labels, rejected candidates, abstentions, summaries, judge
reasoning and all support grades are retained. No failed output is rewritten
into an accepted example. Grades are synthetic relevance labels, **not** HNM
similarity scores; mining remains entirely downstream.

## Handoff, splits and audit

Successful runs publish `bundle/generation_result.json` referencing the
checksummed schema-v2 portable manifest. The bundle contains:

- `source_records.jsonl`, `candidate_diagnostics.jsonl`, `context_outcomes.json`:
  original generated texts, actual judgments and abstentions.
- `retrieval_units.jsonl`, content-addressed `assets/`, binary-image Parquet
  corpus shards, and loader metadata.
- `views/{text,image,image_and_text}/{train,validation}.json`: positive-only
  training records with every localized positive and its grade.
- `synthetic_eval/<view>/`: held-out query JSONL, full eligible corpus and graded
  BEIR qrels; portable images resolve from the **bundle root**.
- `split_manifest.json`, `report.json`, `run_manifest.json`: split assignments,
  distinct IDs/texts/groups, positive counts, view exclusions and file hashes.

Groups resolve before splitting using optional query lineage, normalized
duplicates and opt-in conservative lexical near-duplicates. Independent queries
sharing documents, pages or contexts are not grouped. Unknown semantic
paraphrases cannot be guaranteed detectable. Query variants are retained, not
dropped; all views share the same group assignment. No document-disjoint
protocol is offered by this workflow.

Each view retains **every modality-eligible corpus unit**, including distractors.
A query is excluded from a view if any positive cannot be represented; positives
are never silently dropped. Text-only views additionally exclude visual support.
Missing/empty training views must fail downstream selection, never fall back to
another modality. Labels are source-local, incomplete silver judgments;
unlisted corpus units are unjudged, not known negatives.

## Failure and resume

Runs hold a file lock. Their identity includes configuration, contexts, source
text, image bytes, package implementation hashes and dependency versions.
Requests are cached by exact prompt, schema, model settings and image hashes.
Data Designer owns bounded in-call corrections; the workflow does not add an
unbounded outer retry. Inspect attempt evidence before setting `resume: true`.
Successful cached requests are reused. Changed inputs/configuration/code or
corrupt caches fail instead of silently regenerating or mixing results.

An existing partial bundle is never overwritten. Preserve it and investigate;
do not treat a generation failure as a quality rejection or loosen gates to
make a run finish. The run directory contains potentially sensitive corpus
content and model responses; keep it private even when publishing code.

## EA readiness and limitations

Version `0.3.0a1` is an **unreleased candidate**, not a published package or a quality claim.
CPU tests cover arbitrary collections, actual Data Designer scheduling/parsing
with mocked transport, resume/integrity, evidence checks, and the complete
Nemotron portable consumer when that source is available. Hosted-model quality,
provider-specific image limits, representative end-to-end mining/training and
checkpoint/reload evaluation require a separately recorded release qualification.
Internal experiment outcomes do not certify this newly consolidated workflow.

Before tagging an EA build: review the public PR and companion recipe change,
pin both accepted commits, run a bounded hosted-model smoke test with authorized
generic inputs, inspect quality and rejection records, and qualify native
AutoModel mining, BF16 training/checkpoint/resume and fresh base/final evaluation.
Use independent evaluation data for quality claims. Synthetic held-out results
are workflow diagnostics, not an official benchmark or proof of general uplift.
Do not publish tags or package assets until the release owner approves them.
