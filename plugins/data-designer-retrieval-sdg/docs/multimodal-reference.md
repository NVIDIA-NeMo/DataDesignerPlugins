# Multimodal Configuration Reference

`MultimodalSDGConfig` defines the configuration for
`run_multimodal_sdg`. The module command loads the same fields from YAML.
Unknown configuration fields are rejected. For a first run, use the
[step-by-step guide](multimodal.md).

## Run Configuration

Paths in the configuration resolve from the current working directory. Only
image paths inside source records resolve relative to their JSONL file.

| Field | Default | Meaning |
| --- | --- | --- |
| `sources_file` | Required | Source JSONL file following the contract below. |
| `output_dir` | Required | New run directory, or an existing compatible directory when resuming. |
| `dataset_id` | Required | Nonempty corpus identifier recorded in exports. |
| `generator`, `judge` | Required | Independent model settings, described below. |
| `contexts_file` | `null` | Explicit context JSONL; replaces automatic context grouping. |
| `concurrency` | `8` | Model-call concurrency, from 1 to 128. |
| `batch_size` | `30` | Request batch size, from 1 to 128. |
| `missing_response_attempts` | `3` | Total attempts for missing or invalid rows in normally completed batches, from 1 to 3. |
| `failed_record_policy` | `error` | Use `skip` to record exhausted model-validation failures and continue with usable records. |
| `seed` | `42` | Seed for instruction sampling, selection, and query-group splitting. Model output can still vary. |
| `resume` | `false` | Reuse a compatible run and cached responses. |
| `ratios` | `{train: 0.8, validation: 0.0, evaluation: 0.2}` | Query-group split fractions, each from 0 to 1 and summing to 1. |
| `group_near_duplicates` | `false` | Add conservative lexical query grouping before splitting. |

The plugin groups requests by model role, response schema, and presence of images.
Each batch runs a Data Designer structured-response column. Matching requests
reuse cached responses keyed by their prompt, schema, model settings, and image
content.

`missing_response_attempts` limits attempts for missing or invalid rows after a
batch completes normally. Valid responses are cached before retrying the missing
rows. Runtime exceptions stop the run after preserving available completed rows;
continuing requires explicit resume. Unexpected or duplicate response identities
stop the run before new responses from that attempt are cached.

This limit is separate from Data Designer's structured-response correction and
provider retry behavior. Batch size and retry settings do not define a total
API-call, cost, or elapsed-time budget. Refer to
[Resume an Interrupted Run](multimodal.md#resume-an-interrupted-run) for recovery.

### Continue After Failed Records

For a new run, add this to the existing YAML configuration:

```yaml
failed_record_policy: skip
```

The default `error` policy stops when required structured responses remain
unavailable. With `skip`, the structured column records a diagnostic after Data
Designer's native validation retries are exhausted. This adds no retry loop and
does not change the native retry budget. `missing_response_attempts` keeps its
separate meaning; a recorded failure is terminal and is not retried by that loop
or by resume. Unexplained missing rows, duplicate identities, provider errors,
and integrity errors are never converted into skipped records.

| Failed stage | Disposition |
| --- | --- |
| TOC confirmation | Preserve the source and omit unverified TOC projections. |
| Visual or document enrichment | Preserve original text/images and omit the failed enrichment. |
| Summary or summary combination | Omit the unavailable summary from generation inputs. |
| Summary judgment | Exclude the summary; a failed judgment cannot become a pass. |
| Query generation | Exclude the context's requested query slots. |
| Query metadata, self-sufficiency, relevance, or localization | Exclude the affected query; successful responses remain cached. |

The original corpus and quality thresholds are preserved. If no queries pass,
export still fails. Inspect `generation_failures.json` in the run directory and
portable bundle for stage, request ID, source/context/query identity where
applicable, validation detail, native attempt limit, and the last invalid
response when available. The portable ledger is included in the manifest's
checksums. `report.json.generation_failures` counts failures by stage, failed
requested slots, and excluded query IDs. Failed query-generation slots contribute
to `requested_slots`, separately from model abstentions and quality rejections.
Earlier planning failures have no known query-slot count.
Queries that fail a later judgment or localization still count in
`funnel.generated` and the `retention_fraction` denominator, once per query.
For example, three generated queries with one localization failure and two
accepted queries report three generated and a retention fraction of `2/3`.

The same policy is supported by `RetrievalStructuredColumnConfig`. In skip mode,
non-null output cells are JSON strings containing either a validated response
or a `_retrieval_sdg_failure` diagnostic. The multimodal workflow decodes these
before its typed consumers. Direct column users must handle the diagnostic
before treating a cell as a model response. Failed records have `None` in any
configured trace and reasoning-content columns, including when every record in
a batch fails, so downstream expressions can still reference those columns.

Resume preserves both cached successes and recorded failures. Changing the
policy or upgrading the implementation changes the frozen run identity; use a
new output directory. Failure rates and exclusions can affect sample composition,
so include them when comparing generation yield.

## Source Records

Each line in `sources_file` is one JSON object describing an independently
retrievable unit. Unknown fields are rejected. Keep these fields in the source
file, separate from run configuration:

| Field | Default | Meaning |
| --- | --- | --- |
| `document_id` | Required | Nonempty identifier for the parent document. Multiple units can share it. |
| `unit_id` | Required | Nonempty identifier unique within the source file. |
| `text` | `""` | Extracted source text. Must be nonempty when no image is supplied. |
| `images` | `[]` | Zero or one local image path. Relative paths resolve from the source file's directory. |
| `language` | `source` | Nonempty language label used in context grouping. |
| `source_uri` | `null` | Optional source provenance string. The loader does not download this URI. |
| `page_number` | `null` | Optional positive page number for structural section planning. |

The loader rejects empty files, duplicate unit IDs, missing image files, and
records containing neither text nor an image. A single unit is not split during
request fitting; prepare smaller units before generation when needed.

For programmatic validation, use the source loader:

```python
from data_designer_retrieval_sdg.multimodal.source_file import load_retrieval_sources

sources = load_retrieval_sources("sources.jsonl")
print(len(sources))
```

`RetrievalSource` and `SplitRatios` are defined in
`data_designer_retrieval_sdg.multimodal.source_models`. Query grouping and split
assignment belong to the same multimodal subpackage. These source records are
not the directory-based input contract used by Text QA.

## Model Settings

Configure each model role explicitly. Model identifiers and endpoint capabilities
come from your serving deployment; the plugin does not choose a replacement model.

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | Required | Nonempty model identifier. |
| `endpoint` | Required | HTTP or HTTPS OpenAI-compatible base URL. |
| `credential_env` | Required | Environment variable containing the credential. |
| `temperature` | `0.6` | Sampling temperature, from 0 to 2. |
| `max_tokens` | `8192` | Positive per-request output-token limit. Account for reasoning tokens where applicable. |
| `timeout` | `600` | Positive per-request timeout in seconds. |
| `extra_body` | `{}` | Provider-specific request options. Cannot override `model`, `messages`, `max_tokens`, or `max_completion_tokens`. |
| `context_window_tokens` | `null` | Enables optional local model-token preflight when set. |
| `tokenizer_file` | `null` | Matching local tokenizer JSON for token preflight. |
| `request_overhead_tokens` | `null` | Nonnegative serving-request overhead allowance for token preflight. |
| `image_tokens_per_image` | `null` | Positive image-token upper bound required for image inputs when token preflight is enabled. |

For endpoints that support JSON-object mode, add the following configuration
under both `generator` and `judge`:

```yaml
extra_body:
  response_format:
    type: json_object
```

This option requests JSON formatting; it does not verify source grounding or
factual correctness. Follow the model's recommended sampling settings. For
reasoning models, `max_tokens` can include both reasoning and the final response.

To enable local token preflight for a role, set `context_window_tokens`,
`tokenizer_file`, and `request_overhead_tokens`. Also set
`image_tokens_per_image` when the corpus contains images. Use values appropriate
to the deployment. The check includes the structured-response instructions,
source input, image allowances, and output reserve.

Without local token preflight, Data Designer's rendered-prompt limit still
applies, but it is not a model context-window guarantee. Provider context and
image limits remain applicable.

## Context Planning

A context specifies the units available for one generation request. These fields
control automatic grouping and the bounds applied to planned contexts:

| Field | Default | Meaning |
| --- | --- | --- |
| `context_strategy` | `unit` | `unit` uses individual sources; `document` groups by document and language; `sections` uses ordered sections. |
| `section_size` | `5` | Positive target units per section. Structural sections can contain up to twice this count. |
| `max_units_per_context` | `8` | Positive maximum units per final generation context. |
| `max_context_chars` | `null` | Optional positive source-text character limit for context membership. |
| `related_contexts_per_context` | `0` | Add up to one or two related contexts using lexical summary overlap. |
| `related_summary_similarity` | `0.2` | Lexical similarity threshold, greater than 0 and at most 1. |

For explicit memberships, supply `contexts_file` with records such as:

```jsonl
{"context_id":"compare-manuals","unit_ids":["manual-a-text","manual-b-page"],"language":"en"}
```

Unknown source IDs and repeated IDs within a context are rejected. Membership
controls available evidence, not query split groups. Disable semantic combinations
with `combination_iterations: 0` and leave `related_contexts_per_context: 0` when
you want only your supplied memberships, subject to selection and request bounds.

Section planning preserves whole units in document/language input order. It uses
nonrepeated Markdown headings and model-confirmed table-of-contents boundaries
when they can be mapped to source units. Page-offset mapping requires explicit,
strictly increasing `page_number` metadata; IDs and filenames are not page numbers.
Unsupported or ambiguous structure falls back to headings or fixed-size sections.
The boundary and coverage decisions are recorded in
`planning/section_boundaries.json`.

With `context_strategy: sections`, visual enrichment describes non-prose content
such as charts and diagrams. It does not retain descriptions of images that the
model classifies as ordinary text. Supply extracted text for scanned prose, or
use `context_strategy: unit` to summarize the original page image directly.

Oversized contexts split on whole-unit boundaries. The workflow also checks
judgment-request sizes after query generation. If a generated query would require
an oversized judgment request, it records the superseded slots and generates new
queries for smaller contexts. It does not judge a multi-unit query against only a
fragment. An indivisible oversized unit fails and needs preprocessing.

### Related Section Combinations

Semantic combinations group related section summaries using embeddings and
clustering. Install the optional dependencies from the repository root:

```bash
uv pip install './plugins/data-designer-retrieval-sdg[multimodal]'
```

The following fields configure this optional stage:

| Field | Default | Meaning |
| --- | --- | --- |
| `combination_iterations` | `0` | Clustering iterations, from 0 to 100. Nonzero values require `context_strategy: sections` and an embedding model. |
| `summary_embedding_model` | `null` | Hosted model identifier, or local Sentence Transformers model ID/path. |
| `summary_embedding_endpoint` | `null` | OpenAI-compatible embedding base URL. When unset, embeddings run locally. |
| `summary_embedding_credential_env` | `NVIDIA_API_KEY` | Credential variable for hosted embeddings. |
| `summary_embedding_extra_body` | `{}` | Hosted request options. Cannot override `model`, `input`, or `encoding_format`. |
| `summary_embedding_revision` | `null` | Optional immutable model revision for local embeddings; incompatible with a hosted endpoint. |
| `summary_embedding_device` | `cpu` | Local embedding device. |

For hosted embeddings, configure options supported by that model, including its
input type and truncation policy. The endpoint receives summary text, not images.
For local embeddings, clear the endpoint and hosted options, select the model and
revision, and allow for local weights and memory requirements.

Semantic combinations require at least 12 section summaries in a language group;
smaller groups skip both embedding requests and combinations. The stage can form
pairs or triples across documents. It cannot be enabled together with lexical
`related_contexts_per_context` expansion.

## Summary Selection and Query Profiles

The following fields control which summaries reach generation and what queries
are requested:

| Field | Default | Meaning |
| --- | --- | --- |
| `persona` | A reader seeking specific evidence and useful information from this corpus. | Intended reader used in summary grading. |
| `judge_summaries` | `true` | Judge information richness, persona relevance, query-generation potential, and conceptual clarity. |
| `summary_quality_threshold` | `4` | Each summary grade must meet this value, from 1 to 5. |
| `summary_count` | `null` | Positive cap after summary filtering and deduplication. |
| `summary_fraction` | `null` | Fraction to retain, greater than 0 and at most 1, rounded up. Mutually exclusive with `summary_count`. |
| `summary_near_duplicate_threshold` | `null` | Optional summary similarity threshold, greater than 0 and at most 1. |
| `instructions` | Text, figure, and table defaults | Nonempty list of query profiles with unique names. A custom list replaces the default profiles. |
| `instructions_per_context` | `null` | Optional positive count sampled without replacement from the configured profile list; cannot exceed its size. When unset, all custom profiles are used. |
| `relevance_threshold` | `4` | Minimum query-source relevance score, from 1 to 5. |
| `self_sufficiency_threshold` | `4` | Minimum source-aware self-sufficiency score, from 1 to 5. |
| `require_verbatim_quotes` | `false` | Require source-matching quotations for text and combined support. |

With both fields omitted, the sampler draws one weighted profile from each text,
figure, and table pool for each selected context. The profiles vary query type,
format, and full or partial answerability using the run seed and context ID.
This requests three queries; unsupported tasks can abstain and quality checks
can reject generated queries.

A custom `instructions` list replaces the default profiles rather than extending
them. Each selected profile requests one query or abstention. Setting
`instructions_per_context` samples entries from the configured list instead of
using the built-in weighted sampler. Omit both fields to restore that sampler;
`instructions: []` is invalid. Refer to
[Customize Query Instructions](multimodal.md#customize-query-instructions) for
reasons to tune the profiles and an example.

Custom profiles use the following fields:

| Field | Required | Values |
| --- | --- | --- |
| `name`, `instruction` | Yes | Nonempty strings; names must be unique within the pool. |
| `query_type` | No | Descriptive string, such as `compare-contrast`. |
| `format` | No | `question`, `instruction`, or `keyword`. |
| `modality` | No | `text`, `image`, or `text_and_image`. |
| `persona`, `answerability` | No | Descriptive strings that guide the request. |

Labels guide generation rather than enforce requested style or answerability.
Refer to [quality checks](multimodal-concepts.md#quality-checks-and-their-limits)
for what acceptance establishes.

Exact summary membership/language deduplication is always active. Optional near
deduplication also requires high source-unit overlap, matching document sets,
and compatible numeric and negation content. Selection caps prioritize visual
multi-document combinations, visual single-document combinations, visual
sections, and then the corresponding nonvisual groups. Selection reasons remain
in `context_outcomes.json`; source units are not removed from the corpus.

## Python API

The public API accepts the same configuration as the module command. This example
loads `sdg.yaml` and returns the portable handoff path:

```python
from pathlib import Path

import yaml

from data_designer_retrieval_sdg.multimodal import MultimodalSDGConfig, run_multimodal_sdg

settings = yaml.safe_load(Path("sdg.yaml").read_text())
config = MultimodalSDGConfig.model_validate(settings)
handoff = run_multimodal_sdg(config)
print(handoff)
```

## Output Files

A successful run writes the following artifacts under its output directory:

| Path | Contents |
| --- | --- |
| `identity.json` | Configuration, source/context identity, image hashes, implementation hashes, and dependency versions. |
| `inputs/assets/` | Frozen copies of source images used during generation. |
| `inference/responses/` | Validated responses keyed by request content, with integrity hashes. |
| `inference/attempts/<attempt-id>/` | Requested IDs, a seed Parquet file with source text and encoded images, Data Designer artifacts, and an outcome or failure record when available. |
| `planning/` | Planning artifacts when produced by the selected strategy. |
| `candidates.json`, `context_outcomes.json` | Accepted/rejected candidates, selection decisions, and generation slots. |
| `complete.json` | Integrity inventory for the completed bundle. |
| `bundle/generation_result.json` | Portable handoff referencing the schema-v2 bundle manifest. |
| `bundle/run_manifest.json` | Policies, models, counts, relative artifact paths, sizes, and checksums. |
| `bundle/retrieval_units.jsonl`, `bundle/assets/` | Canonical corpus units and copied images. |
| `bundle/source_records.jsonl`, `bundle/candidate_diagnostics.jsonl` | Candidate queries and recorded judgments. |
| `bundle/views/<view>/` | Positive-only train/validation records, shared Parquet corpus, and loader metadata. |
| `bundle/synthetic_eval/<view>/` | Query JSONL, eligible corpus JSONL, and graded qrels. Image paths resolve from the bundle root. |
| `bundle/split_manifest.json`, `bundle/report.json` | Split assignments, source coverage, selection/rejection counts, view exclusions, and requested-versus-observed labels. |

Views are created only when they have eligible corpus units, and can have empty
query splits. The pipeline exports `text`, `image`, `image_and_text`, and `mixed` views.
There is no run configuration field for selecting export views; downstream
consumers select from the completed bundle.
Refer to [evidence and export views](multimodal-concepts.md#evidence-and-export-views)
for their eligibility rules.
