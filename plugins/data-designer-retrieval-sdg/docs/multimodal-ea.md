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
split group. Unselected units still belong to the full corpus. Unknown IDs fail.
Oversized memberships are partitioned, retaining every source unit and interleaving
documents to preserve opportunities for cross-document questions. The default
bounds are eight units and 100,000 serialized source-text characters. These are
not token limits; provider image/token limits may require smaller settings. A
single indivisible unit above the bound fails with its ID; preprocess it into
smaller canonical units rather than truncating it or changing its identity here.

The Nemotron EA profile selects `context_strategy: sections`. It describes
images separately from original source text, describes documents, and
summarizes whole-unit sections in document/language input order. Automatic
sections use nonrepeated markdown headings and confirmed table-of-contents (TOC)
boundaries. Likely TOC units receive a structured judge request through the same
content-addressed cache and bounded retry path as other inference. A confirmation
does not authorize trusting the printed page numbers: titles must match normalized
source lines, or two distinct title/page anchors must establish a consistent
offset into explicit, strictly increasing `page_number` metadata. Unit IDs and
filenames are never parsed as page numbers. Missing/ambiguous offsets, invalid
pages and backward boundaries are rejected; Roman labels and unsupported TOC
formats fall back to headings or fixed-size sections.

`section_size` is the target (default 5); structural sections can span up to twice
that many units (default 10). Gaps without usable structure use the target size.
A heading inside a unit maps to the start of that whole unit; text is never split
or reconstructed. Every source unit must appear exactly once in reading order
within its document/language, including the final unit. Coverage is checked again
after enriched-text bounding. `planning/section_boundaries.json` records TOC
confirmations, accepted/rejected mappings, detected boundaries and coverage.
Explicit contexts replace automatic sections. Set `combination_iterations: 0`
to generate only from supplied memberships (subject to selection and bounds).

With `combination_iterations: 20`, summary embeddings feed seeded UMAP/HDBSCAN
clustering and related section pairs/triples, including combinations across
documents. Install the plugin's `multimodal` extra. The EA example uses
[`nvidia/nemotron-3-embed-1b`](https://build.nvidia.com/nvidia/nemotron-3-embed-1b)
through the public NVIDIA API, with `input_type: passage`. Set `NVIDIA_API_KEY`
in the environment; only its variable name is recorded. The endpoint receives
section-summary text, not images. Requests use batches of at most 32 summaries;
responses are checked for complete, ordered, finite embeddings and cached for
resume. `truncate: NONE` makes oversized inputs fail rather than silently truncate.
Model, endpoint, credential-variable name and extra request fields are configurable
independently of the generator and judge.

For a local Sentence Transformers model, set `summary_embedding_endpoint: null`,
`summary_embedding_extra_body: null`, and `summary_embedding_model` to its Hugging
Face ID or local path. Set `summary_embedding_revision` to an immutable revision
and use `summary_embedding_device` to select CPU (default) or an available GPU.
Revision and device settings apply only to local inference. `null` clears inherited
hosted request options; an empty mapping may retain keys under CLI/config merging. Language groups with
fewer than twelve section summaries skip combinations and embedding requests.

Summary grading, deduplication and budget selection happen before the final
eight-unit image/text generation bound. Larger selected memberships are split
without dropping evidence. Identical resulting generation memberships run once.
The original `unit` and `document` strategies remain available, along with opt-in
lexical `related_contexts_per_context`; lexical and semantic combinations cannot
be enabled together. Semantic combinations require the `sections` strategy.

## Run

Install the candidate with its `multimodal` extra from the reviewed public commit before using this API;
the existing published package does not yet provide it. No core-package patch
is needed: the registered `retrieval-structured` column uses released Data
Designer and retains schema validation and native bounded corrections.

Save an operator-owned YAML configuration:

```yaml
sources_file: /absolute/path/to/sources.jsonl
contexts_file: /absolute/path/to/contexts.jsonl  # optional; replaces automatic sections
context_strategy: sections
section_size: 5
combination_iterations: 20
summary_embedding_model: nvidia/nemotron-3-embed-1b
summary_embedding_endpoint: https://integrate.api.nvidia.com/v1
summary_embedding_credential_env: NVIDIA_API_KEY
summary_embedding_extra_body:
  input_type: passage
  truncate: NONE
summary_count: 400
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
max_context_chars: 100000
judge_summaries: true
summary_quality_threshold: 4
require_verbatim_quotes: false
missing_response_attempts: 3
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

1. Enrich and summarize sources, form related summary combinations, then judge
   information richness, persona relevance, query-generation potential and
   conceptual clarity. Every grade must meet the configured floor (default 4).
   Summaries select evidence contexts; they never replace corpus text or positives.
2. Generate queries from original text **and attached pixels**, using packaged
   single/multi-document instructions and standalone-query examples. Each context
   deterministically samples one weighted text, figure and table module, including
   type, format and full/partial answerability. Unsupported slots explicitly abstain.
   Custom `instructions` and `instructions_per_context` remain supported.
3. Judge self-sufficiency with the query **and original source text**. Separately
   classify observed type/format and answer leakage from the query alone. Judge
   relevance with original source text and images. A generated summary cannot
   supply missing evidence to these judges.
4. Localize source support: grade 2 means complete support; grade 1 means useful
   partial evidence. Drop out-of-context IDs and keep the highest grade per
   repeated unit. Text support requires source text; image support requires a
   source image and visual explanation. In non-strict mode, quotes are optional
   diagnostics for every modality: missing or non-matching quotes do not reject
   otherwise valid support. With `require_verbatim_quotes: true`, text and
   text+image supports must contain a nonempty, source-matching quote. Image-only
   supports never require a text quote. All other quality and source-identity
   checks remain unchanged.

The packaged templates and weighted samplers derive from the MIT-licensed
[ViDoRe v3 generation implementation](https://github.com/illuin-tech/vidore-v3-generation/tree/2cfd2f78b8ae4f4bedd87d67f373dc51c12a856e).
`multimodal/THIRD_PARTY_NOTICE.txt` records attribution. This port adapts behavior
to generic source identities, whole-unit sections and the existing output schemas;
it does not import benchmark ingestion or depend on an external checkout.
Its context-aware self-sufficiency score reproduces that judging policy, not an
independent query-only standalone assessment. Retention must be measured on the
actual corpus; it is not guaranteed to match earlier benchmark runs.

Acceptance requires relevance and self-sufficiency scores at least 4/5, no
answer leakage, and valid nonempty localized support. Both score thresholds
are explicit configuration. Requested style/type is diagnostic, not a gate.
Observed query labels, rejected candidates, abstentions, summaries, judge
reasoning and all support grades are retained. No failed output is rewritten
into an accepted example. Grades are synthetic relevance labels, **not** HNM
similarity scores; mining remains entirely downstream.

### Summary selection

`judge_summaries` defaults to true. Each of the four grades must meet
`summary_quality_threshold` (default 4). Full judgments are preserved. A disabled
judge is recorded as absent, never as a fabricated pass. Exact membership/language
duplicates keep the representative with the highest sum, then minimum grade;
ties retain input order.

Optional `summary_near_duplicate_threshold` compares five-character shingles,
requiring at least 90% source-unit overlap, the same document set, summaries at
least 40 characters long, and matching numeric/negation tokens. It compares only
against retained representatives, never transitive clusters. With `unit` or
`document` planning, distinct eight-unit-bounded contexts cannot meet the 90%
overlap guard. With `sections` planning, deduplication precedes the final generation
bound, so larger combined summaries can qualify (for example, nine units contained
in ten). Character bounds can still split those larger planning contexts. The
example omits this optional setting; exact deduplication always remains active.

Set either `summary_count` or `summary_fraction`, never both. Fractions round up
after quality filtering/deduplication. Within each combined-summary priority
category, budgeting uses a hash of `seed`, language and sorted source-unit IDs.
This reproducible ordering prevents the cap from favoring early documents in
lexicographically sorted combinations. It preserves the reference category
priorities, rather than reproducing the reference implementation's incidental
set iteration order. Single-section categories retain input order.

Budgeting prioritizes visual multi-document
combinations, visual single-document combinations, visual sections, then the
corresponding nonvisual groups. The EA
recipe caps selection at 400 summaries. All selection reasons and representative
IDs remain in `context_outcomes.json`; no corpus units are removed.

The `planning/` directory records visual enrichment, document descriptions,
section/combined summaries, membership combinations and all four summary grades.
Descriptions are generated planning aids; original input text and images remain
unchanged. Character bounds include generated visual descriptions, without silent
truncation. Model weights are downloaded to the local Hugging Face cache when
needed. Budget for that model, clustering dependencies and additional hosted calls.

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
  Reports also include source-unit/document coverage, summary selection reasons,
  sequential query gate counts, independent rejection counts, retention rates,
  generated modalities, image-dependent and multi-positive fractions, images per
  context, requested-versus-observed type/format counts, and quote fidelity.
  Generated coverage counts units in contexts with at least one non-abstaining
  query; all-abstaining contexts contribute only to planned coverage. Accepted
  positive coverage separately counts localized positive units.

Groups resolve before splitting using optional query lineage, normalized
duplicates and opt-in conservative lexical near-duplicates. Independent queries
sharing documents, pages or contexts are not grouped. Unknown semantic
paraphrases cannot be guaranteed detectable. Query variants are retained, not
dropped; all views share the same group assignment. No document-disjoint
protocol is offered by this workflow.

Each view retains **every modality-eligible corpus unit**, including distractors.
A query is excluded from a view if any positive cannot be represented; positives
are never silently dropped. Text-only views require text support, image-only views require image support,
and support that needs both modalities is retained only in the combined view.
Attaching an image to a text-grounded source does not make its query image-grounded.
Missing/empty training views must fail downstream selection, never fall back to
another modality. Labels are source-local, incomplete silver judgments;
unlisted corpus units are unjudged, not known negatives.

## Failure and resume

Runs hold a file lock. Their identity includes configuration, contexts, source
text, image bytes, package implementation hashes and dependency versions.
Requests are cached by exact prompt, schema, model settings and image hashes.
Data Designer owns bounded in-call corrections. Normally completed batches with
missing/invalid rows get at most `missing_response_attempts` total attempts (1–3,
default 3), retrying **only uncached rows**. Raised authentication/runtime errors
stop immediately, preserving any completed native final-row shards and cacheable
responses; they are not blindly retried. Attempt records are immutable and
exception messages are not copied into failure metadata. Inspect attempt evidence
before setting `resume: true`.
Query generation uses a request-sized schema requiring exactly one outcome for
each requested slot. Missing, duplicate and extra slots undergo native correction
before any response is cached; response order and explicit abstentions are preserved.
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

Before tagging an EA build: review the public PR and direct recipe PR update,
pin both accepted commits, run a bounded hosted-model smoke test with authorized
generic inputs, inspect quality and rejection records, and qualify native
AutoModel mining, BF16 training/checkpoint/resume and fresh base/final evaluation.
Use independent evaluation data for quality claims. Synthetic held-out results
are workflow diagnostics, not an official benchmark or proof of general uplift.
Do not publish tags or package assets until the release owner approves them.

## Generic feature coverage

The consolidated implementation preserves reusable capabilities, not every
historical algorithm or benchmark policy. The focused public-interface regression
suite is `test_multimodal_planning.py`; native inference/export/resume tests are
in `test_multimodal_sdg.py`.

| Capability | Simple public implementation / regression |
| --- | --- |
| Automatic document/section contexts and summaries | Canonical unit grouping, bounded sections; complete ordered coverage test. |
| Related cross-document contexts | Lexical summary overlap, bounded pairs/triples; disjoint-topic and no-duplicate tests. |
| Summary quality, deduplication, budgets | Recorded source-grounded judgment, best representative, optional near match, post-dedup count/fraction tests. |
| Lossless request bounds | Unit/character bounds and document interleaving; unchanged source bytes/IDs and oversized-unit failure tests. |
| Query diversity | User-defined typed profiles, context-local seeded sampling; deterministic sampling and non-gating mismatch tests. |
| Diagnostics | Coverage, funnel, modalities and requested/observed labels; exact count tests. |
| Quotation policy | Diagnostic default plus strict option; both behaviors tested. |
| Missing-response recovery | Finite pending-only attempts, completed-row harvesting, no retry on runtime exceptions; cache-reuse tests. |
| Recipe configuration | Nemotron forwards producer controls and independent model settings through validated options; rejects unknown/reserved keys. |

Dataset ingestion, domain-specific prompts and corpus repairs belong in caller
preprocessing. Native mining and its checks belong to the recipe/AutoModel. The
generation plugin operates on the canonical source and context contracts above.
