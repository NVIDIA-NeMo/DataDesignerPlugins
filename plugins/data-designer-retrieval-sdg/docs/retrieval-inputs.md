# Retrieval SDG from text and images

This page documents the existing QA workflow. New multimodal recipe runs use
the [retrieval-first EA entry point](multimodal-ea.md), with direct query
generation and recorded source-local relevance judgments instead of generated
answer comparison.

The retrieval-SDG package uses one native Data Designer pipeline for text-only,
image-only, and image-plus-text source units. Multimodality is a property of
each seed row, not a separate workflow: every row has an `images` list and a
text-only row simply uses `images=[]`.

## Structured responses without a core patch

The shared pipeline uses the package's `retrieval-structured` column for model
outputs. It accepts either one complete JSON object or one JSON Markdown code
fence. It does not extract objects from prose or repair malformed responses.
Duplicate keys, non-finite numbers, and schema-invalid fields are rejected.
Extra fields are not silently removed, and negative judge decisions remain
negative.

The column extends Data Designer's native structured generator. Image context,
secure prompt rendering, skip propagation, traces, asynchronous scheduling,
and bounded correction retries remain owned by Data Designer. Configure retry
limits through its `RunConfig.max_conversation_correction_steps` and
`RunConfig.max_conversation_restarts`. Exhausted parsing or schema failures
remain generation failures, not accepted retrieval records.

No Data Designer source patch is required. The pipeline builders select this
column automatically. For direct use, supply the same fields as a native
structured column:

```python
from data_designer_retrieval_sdg import RetrievalStructuredColumnConfig

column = RetrievalStructuredColumnConfig(
    name="decision",
    model_alias="judge",
    prompt="Source: {{ text }}\nQuery: {{ query }}\nIs the source relevant? Explain your decision.",
    output_format={
        "type": "object",
        "properties": {"relevant": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["relevant", "reason"],
        "additionalProperties": False,
    },
)
```

Register the `judge` model alias and provide the query and source context when
adding this column to a builder. This example illustrates output handling;
the retrieval pipeline still performs its separate quality and grounding checks.

## Input contract

Use `RetrievalSource` to build a canonical row containing one independently
retrievable unit:

| Field | Required | Description |
|---|---|---|
| `document_id` | Yes | Stable source-document identifier for provenance, not query splitting. |
| `unit_id` | Yes | Stable identifier for the positive retrieval unit. |
| `text` | Conditional | Parsed text. It can be empty when an image is present. |
| `images` | Conditional | Zero or one local PNG, JPEG, or WebP path in the current page-level implementation. |
| `language` | No | Language code or `source` to preserve the source language. |
| `source_uri` | No | Human-readable source locator. |
| `page_number` | No | One-based page number when the unit came from a paginated source. |

One of `text` or `images` must contain source material. The built-in
`DocumentChunkerSeedSource` already emits this contract for ordinary text
files, with one retrieval unit per chunk and `images=[]`.

For recipe integrations, write one `RetrievalSource` JSON object per line and
pass the file through `RetrievalSourcesFile` and `run_generation`:

```python
from pathlib import Path

from data_designer_retrieval_sdg import (
    GenerationRunConfig,
    RetrievalSourcesFile,
    run_generation,
)

result = run_generation(
    GenerationRunConfig(
        seed_source=RetrievalSourcesFile(path=Path("./retrieval-sources.jsonl")),
        artifact_path=Path("./artifacts"),
        output_dir=Path("./generated"),
        dataset_name="retrieval-sdg",
        resume="if_possible",
    )
)
print(result.output_path)
```

This concise configuration uses the package's default model roles and is
directly suitable for text-only source files. If any row contains an image,
configure image-capable artifact-extraction, QA-generation, and quality-judge
models through `GenerationPipelineConfig`. The complete mixed-input example
below shows those roles explicitly.

Relative image paths resolve from the JSONL file's directory. Before generation,
the runner validates the ordered identities, copies images into
content-addressed assets, and writes a content-addressed native seed snapshot
under `artifact_path/.retrieval_sdg_inputs/`. Changed source text or image bytes
therefore produce a different Data Designer configuration fingerprint. A
completed canonical run is accepted only when the exported rows preserve the
selected source IDs, retrieval units, language, order, and count. The raw export
is retained if this final coverage check fails, but success metadata is not
written.

PDF rendering and parsing stay outside the core pipeline. Nemotron Parse,
another parser, or a plain renderer can produce page text and images before
generation. Parsed text is optional; it is useful for text and combined export
views but is not required for image-grounded generation.

## Build and run

The example deliberately mixes a text row with a page-image row. Both pass
through the same pipeline. Set `RETRIEVAL_GENERATOR_MODEL` and
`RETRIEVAL_JUDGE_MODEL` to image-capable chat model IDs available from the
configured provider. The example requires both variables rather than assuming
that a particular preview model is publicly available.

```python
import os
from pathlib import Path

import data_designer.config as dd
import pandas as pd
from data_designer.interface import DataDesigner

from data_designer_retrieval_sdg import (
    RetrievalSource,
    build_retrieval_pipeline,
    export_retrieval_data,
)

sources = [
    RetrievalSource(
        document_id="label-001",
        unit_id="label-001-section-4",
        text="Dosage and administration guidance.",
        language="en",
        source_uri="label-001.txt",
    ),
    RetrievalSource(
        document_id="study-001",
        unit_id="study-001-page-7",
        text="Optional parser output for the page.",
        images=[str(Path("pages/study-001-page-7.png").resolve())],
        language="en",
        source_uri="study-001.pdf",
        page_number=7,
    ),
]
seed = dd.DataFrameSeedSource(df=pd.DataFrame([source.to_seed_record() for source in sources]))
generator_model = os.environ["RETRIEVAL_GENERATOR_MODEL"]
judge_model = os.environ["RETRIEVAL_JUDGE_MODEL"]
models = [
    dd.ModelConfig(
        alias="vision",
        model=generator_model,
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(temperature=0.6),
    ),
    dd.ModelConfig(
        alias="judge",
        model=judge_model,
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(temperature=0.0),
    ),
    dd.ModelConfig(
        alias="embed",
        model="nvidia/nemotron-3-embed-1b",
        provider="nvidia",
        inference_parameters=dd.EmbeddingInferenceParams(extra_body={"input_type": "query", "truncate": "NONE"}),
    ),
]
builder = build_retrieval_pipeline(
    seed,
    model_configs=models,
    artifact_model_alias="vision",
    generator_model_alias="vision",
    judge_model_alias="judge",
    embedding_model_alias="embed",
)

designer = DataDesigner(artifact_path="./artifacts")
result = designer.create(builder, num_records=len(sources), dataset_name="retrieval-sdg")
result.export("./generated.jsonl", format="jsonl")

summary = export_retrieval_data(
    "./generated.jsonl",
    "./recipe-bundle",
    dataset_id="pharmaceuticals/domain-adaptation",
    generator_model=generator_model,
    judge_model=judge_model,
)
print(summary.run_manifest_path)
```

When every row is text-only, the existing `build_qa_generation_pipeline`
entry point remains available and delegates to this same pipeline. Image rows
require vision-capable artifact, generator, and grounding-judge models. Using
a distinct judge model is recommended and recorded for audit, but is not
universally required.

## Quality gates

The generation flow is:

1. Extract structured artifacts and an advisory retrieval-suitability profile
   from text and any attached images. The profile identifies meaningful charts,
   tables, diagrams, layout, and photographs and suggests verifiable operations.
2. Generate standalone retrieval queries without answers. Use the profile to
   guide query types; do not force visual questions on text-heavy pages.
3. Deduplicate query text with the existing `embedding-dedup` plugin.
4. Judge query plausibility, standalone quality, discrimination, surface, and
   language without exposing source text, images, evidence, or answers.
5. Select queries that pass both deterministic wording checks and the independent
   query judge. Missing or duplicate judgements fail closed for that query.
6. Generate answers, evidence, and positive unit IDs only for selected queries.
   The answer schema contains an index instead of query text, so this stage
   cannot rewrite the query. Unsupported queries may be left unanswered.
7. Join answers to the original queries by index and align the surviving query
   judgements. Missing or duplicate answers are quarantined.
8. Read the positive sources independently in `source_assessments`. This call
   sees the queries, positive IDs, native text, images and language, but **not**
   generated answers, claimed evidence, hop summaries or extracted artifacts.
   It records its own answer and support, uncertainty, source relevance,
   evidence modality, native-text quotes and source-language assessment.
9. Compare the unchanged candidate answer and evidence with that independent
   reading in `qa_evaluations`. This call receives no raw source fields or
   images: it cannot reinterpret the source to make the candidate correct.
   It checks material agreement, whether the answer resolves the query, and
   answer leakage. Equivalent wording is allowed; contradictory quantities,
   dates, units or entity/time alignment are not.

The source reader never sees the generated claims it is helping verify.
The final comparison cannot rescue an uncertain, unanswerable, irrelevant or
wrong-language reading. Both calls use `judge_model_alias`; a nonempty source
row adds one structured call compared with a single source-aware answer judge.
All candidates in that row share the call. Text-only rows use the same stages.

When a positive unit has no supplied text, reading its
printed text or table cells still requires image access and is labelled
`image_grounded`. This is distinct from the source profile's assessment of
whether it supports visual reasoning beyond reading prose.

For `text_only` decisions, the source reader supplies `native_text_evidence`: verbatim
quotes with their positive `unit_id`. When any positive contains an image,
the exporter requires matching quotes for every positive unit. A nonempty
title does not establish support for details available only in the image.
Every supplied quote must match that unit's supplied text after whitespace
normalization. Missing coverage or invalid quotes quarantine the candidate;
the exporter never changes its modality label to make it pass.
Exclusively text-only positives can omit these quotes, but still require an
independent assessment for verified export. The final comparison must preserve
the source reading's relevance, language, modality and quotes; disagreement
causes quarantine. The exporter checks the reading's own quotes against native
source text, not against its generated answer or supporting-evidence prose.
Quote matching verifies provenance, not whether the text entails the answer;
the independent source reader and comparison remain responsible for that
assessment. Independent reading can still be wrong, and correlated mistakes can
pass comparison. Inspect accepted examples; these checks do not prove visual
accuracy or retrieval improvement.

Set `answer_model_alias` on `build_retrieval_pipeline` to use a separate model
for answers; it defaults to `generator_model_alias`. Both must support images
when source rows contain them. Selection and assembly use native Data Designer
custom columns; the remaining stages use structured LLM columns and the
existing dedup plugin. Build the configuration with the Python builder so the
custom callables are registered.
When exporting with a separate answer model, pass its identity as `answer_model`
alongside `generator_model` and `judge_model` to retain all three roles in the
bundle manifest.

Suitability is advisory, not an acceptance gate. Text visible in an image may
still support useful queries when the page has no visual reasoning content.
The query generator may return fewer candidates or none when evidence is
insufficient. The framework skips query judging for empty candidate lists and
skips answer generation, source assessment and comparison when no candidates remain.

The exporter also quarantines high-confidence source-relative queries, direct
answer disclosure, and exact queries assigned to different positives. It
never mechanically rewrites failed queries.

An accurate statement that the requested information is absent does not make
the source a positive retrieval example. `answer_resolves_query` is a mandatory
grounding criterion. Outputs from older judge schemas must be rejudged before
export; do not fill this field with an assumed pass.
Records without `source_assessments` remain readable but are quarantined on
export with `missing_source_assessment`, including historical text-only records.
Do not copy candidate answers into the missing assessment. Run fresh independent
reading and comparison; the exporter never repairs or rewrites old records.
If every candidate fails, export raises an error listing rejection-reason codes
and does not create a training bundle.

Every accepted item has a query surface—`question`, `instruction`, or
`keyword`—and a judge-assigned evidence modality—`text_only` or
`image_grounded`. The evidence modality says whether pixels are required; it
does not describe which export representations happen to be available.

## Output

The protocol is grouped query-disjoint retrieval over a fixed, shared corpus.
Query groups are assigned to train, validation, or evaluation; all partitions
retrieve from the full eligible collection, including distractors. Documents
can support queries in several partitions. Sharing a page, document, summary,
or generation context does not automatically group independent questions.

Optional `query_provenance` on each generated record maps final
`deduplicated_qa_pairs` candidate indexes to corpus-independent provenance:

```json
{"query_provenance": {"0": {"query_group_id": "warranty-question",
                           "seed_query_id": "seed-42"}}}
```

`parent_query_ids` can additionally connect known derivations. Populate these
fields from known generation/import provenance, not model guesses. Missing
provenance starts each accepted query in its own group. Normalized duplicates
are joined before splitting; `group_near_duplicates=True` additionally enables
a conservative lexical heuristic. Unknown translations or semantic equivalents
cannot be guaranteed detectable. Dataset-specific mapping belongs in external
experiment or customer preparation scripts, not this plugin.

Group assignments are global across views and recorded in `split_manifest.json`.
The exporter writes independently sampleable views:

- `text`, for units containing text
- `image`, for units containing an image
- `image_and_text`, for units containing an image, with text included when available

Image-grounded candidates never enter the text view. A text-grounded candidate
can enter an image view only when every positive unit actually has an image.
Each view contains one `corpus/shared/` Parquet corpus, positive-only training JSON,
and BEIR-style synthetic evaluation files. Source records, units, judgements,
relative image assets, and checksums make the bundle self-contained.

The version-2 run manifest includes source-suitability counts, early rejection counts,
and rejection-reason counts. Its `quality_contract` is
`independent_source_assessment_v1`; the recorded judge model performs both
independent reading and final comparison. `source_records.jsonl` preserves
`source_assessments`, unchanged candidates, final judgements, source profiles
and `generation_diagnostics`. Early records in `candidate_diagnostics.jsonl`
have `stage="pre_grounding"`, use indexes into the deduplicated query list, and
have no invented answer or positive IDs. Completed-pair diagnostics use indexes
into `deduplicated_qa_pairs`. A suitability profile does not determine the
judge-owned evidence modality.
Early rejections include query selection failures before answer generation
and missing or duplicate answers detected during assembly. The manifest's
`pre_grounding_rejected_candidate_count` includes both; the reason counts
distinguish them.

If image inputs produce no useful image-grounded candidate, the exporter warns
by default. Set `strict_visual=True` to fail instead. This gate does not apply
to a genuinely text-only dataset.

Hard-negative mining remains a later Nemotron recipe stage.

## Current scope

The plugin supports page-level source units and one image per unit. It does not
embed PDF parsing, custom checkpoints, hard-negative mining, region-level
geometry, or a second execution engine. Data Designer owns execution and resume
semantics. A hosted parser adapter can be added later without changing the
canonical retrieval row or the generated-data contract.
