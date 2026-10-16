# Understand Multimodal Retrieval Data

The multimodal workflow creates queries with graded supporting source units.
Its output is retrieval supervision: it identifies relevant evidence without
generating and verifying an answer. This distinction matters when choosing
training data and interpreting quality scores.

For execution steps, use [Generate Retrieval Data From Text and Images](multimodal.md).

## Sources, Contexts, and Queries

A **source unit** is an independently retrievable item, such as a text passage or
page image. A **context** is a set of units presented together during generation.
A **positive** is a unit that the judge identifies as supporting a generated query.

A context can contain multiple pages or documents, but a query does not have to
use every member. Context membership determines the evidence available to the
model. It does not determine the training split or prove multi-step reasoning.

Python coordinates the stages and their saved state. Model-call batches use
Data Designer structured-response columns. This differs from the single
four-column Data Designer graph used by [Text QA](text-qa.md).

The workflow applies these stages:

1. Plan contexts and create summaries. Sources with images can receive separate
   visual descriptions. Optional section combinations expand the contexts.
2. Select summaries using quality grades, duplicate filtering, and optional
   count or fraction limits.
3. Generate queries from original source text and attached images. Unsupported
   instruction slots can return an abstention.
4. Judge query properties and relevance, then identify supporting source units
   and their evidence modality.
5. Export accepted queries and the complete eligible corpus for each view.

Summaries and visual descriptions guide planning. They do not replace original
source evidence in the corpus. Units that are not selected for generation can
still appear as corpus documents.

## Requested Tasks and Observed Evidence

An instruction requests a task, such as a comparison of charts or a factual
question about text. It can also request a question, instruction, or keyword
format. These are generation preferences, not acceptance guarantees.

The default sampler chooses weighted text, figure, and table profiles. The
profiles include fully and partially answerable information needs. Partial
answerability can produce a useful retrieval query whose complete answer needs
additional evidence, which is not guaranteed to exist elsewhere in your corpus.

The generator reports an evidence modality, and the localization judge records
support for each positive. A requested image task does not prove image support.
Inspect requested-versus-observed labels and the original sources when checking
whether the generated dataset has the diversity you need.

The packaged prompts and sampling profiles derive from the MIT-licensed ViDoRe v3
generation implementation. Attribution is recorded in the package's
`multimodal/THIRD_PARTY_NOTICE.txt`. The plugin operates on prepared source units;
it does not include benchmark ingestion.

## Quality Checks and Their Limits

By default, each of the four summary grades must reach 4 out of 5. Candidate
acceptance then requires self-sufficiency and relevance scores of at least 4,
no answer leakage, and nonempty valid localized support.

The checks use different evidence:

| Check | Input | Interpretation |
| --- | --- | --- |
| Query metadata and answer leakage | Query alone | Classifies the written query and checks whether it embeds its answer. |
| Self-sufficiency | Query and original source text | A source-aware score; it does not independently establish that the query is clear without context. |
| Relevance | Query, original text, and images | A model judgment that the source is useful for the information need. |
| Support localization | Query, original text, and images | Identifies source IDs, support grades, quotations, and visual explanations. |

A support grade of 2 means the judge reports complete support; 1 means useful
partial support. These are synthetic relevance labels, not embedding-similarity
scores or prescribed training weights. Both grades are exported as positives.
Grade 1 means incomplete answer coverage, not low confidence that a fact is correct.

The plugin checks source membership and modality requirements. It does not
independently verify every visual statement or prove complete answerability.
Models can reject valid questions, confuse chart elements, or produce inaccurate
support explanations. A source supplying a prerequisite does not necessarily
explain why that prerequisite is needed.

Set `require_verbatim_quotes: true` to require matching quotations for text and
combined text-and-image support. The default is `false`, which records quotation
checks without rejecting missing or nonmatching quotations. Exact quotation
matching does not verify visual reasoning or establish that the quote answers
the whole question.

### Refine Generation With Sample Review

The workflow filters candidates and exports training positives automatically.
It does not require manual approval of individual records. To improve generation
for your corpus, optionally sample accepted and rejected queries across source
types and evidence modalities:

1. Compare each query and diagnostic with the original source. Check product
   names, identifiers, metrics, units, table columns, and conditions.
2. Check that the recorded text or image contains the claimed evidence. A matching
   quotation alone does not establish the relevance of that evidence.
3. Check completeness: grade 2 requires support for the whole query; grade 1
   requires a useful supported part. Topic overlap alone is insufficient.
4. Inspect rejected queries and abstained slots for useful evidence that was
   missed. Retention alone does not measure factual accuracy or corpus coverage.

Keep review findings separately from the checksummed bundle and preserve the
original model outputs. Changed inputs or generation settings require a new run
directory. Sample review can identify improvements, but measuring retrieval
quality requires independent evaluation data.

## Evidence and Export Views

A **view** chooses a corpus representation for downstream retrieval. An
**evidence modality** describes what supports a particular query. They interact
when the exporter decides whether all of a query's positives fit a view.

| View | Corpus Units | Eligible Positive Evidence |
| --- | --- | --- |
| `text` | Units with nonempty text | Every positive must use text support. |
| `image` | Units with an image | Every positive must use image support. |
| `image_and_text` | Units with an image, with their available text | Positives can use text, image, or both, provided every positive unit has an image. |
| `mixed` | All units, including text-only, image-only, and paired units | Retains every accepted query and its complete localized positive set. |

The combined view is not the complete heterogeneous corpus: it excludes units
without images. For example, a query supported by a text-only unit and an
image-only unit is excluded from `text`, `image`, and `image_and_text`. It remains
in `mixed`, which preserves both positive units. Text-only corpus rows in this
view have a null image; image-only rows have empty text.

Views are created during export after query generation and judging. Choosing a
downstream view does not limit which queries the pipeline generated. To focus
generation on particular evidence, configure the instruction pool as described in
[Request Only Text Questions](multimodal.md#request-only-text-questions).
Instruction modalities guide the model; they do not enforce the final support labels.

A view includes every eligible corpus unit, including units with no accepted
queries. It excludes a query when any positive cannot be represented; it does
not drop that positive to make the query fit. An empty or absent view does not
justify falling back to another modality.

### One Source, Different Recorded Support

The [first-run tutorial](first-run.md) supplies a DGX B200 guide page as
extracted text and a full-page image. The text describes software, and the image
also shows the system topology. The following illustrative questions show how
recorded support determines eligible views:

| Question | Example Recorded Support | Compatible Views |
| --- | --- | --- |
| Which DGX B200 software provides active health monitoring and system alerts? | `text`: the NVSM description. | `text`, `image_and_text`, and `mixed`. |
| Which GPUs are grouped under CPU 1 in the DGX B200 topology diagram? | `image`: the diagram shows GPU 4 through GPU 7. | `image`, `image_and_text`, and `mixed`. |
| Which software can administer the GPUs in the DGX B200 topology, and how many GPUs are shown? | `image`: the page image contains the DCGM description and the eight GPUs in the diagram. | `image`, `image_and_text`, and `mixed`. |

These are illustrative support assignments, not captured model judgments. The
full-page image also contains the software prose, so a judge could record image
support for a question about that prose. For the last question, the image alone
contains both requested facts; the presence of text does not require joint
support. The localization judge sees text and images together and can still
misattribute the evidence. The exporter follows recorded support without
inferring alternatives or relabeling a positive to fit another view.
Review the original evidence when a label seems incorrect.

## Splits and Evaluation

Queries are grouped using normalized duplicates, optional lineage, and optional
conservative lexical near-duplicate matching. Groups are assigned to train,
validation, or evaluation before views are written. The same group has the same
split across views. Undetected semantic paraphrases can remain in separate groups.

The protocol is `grouped_query_disjoint`. Corpus documents are shared across
splits; this is not document-disjoint evaluation. Multiple queries about one
source are not automatically a single group.

Synthetic evaluation data contains queries, the eligible corpus, and graded
known positives. Unlisted documents are unjudged, not known negatives. These
labels are incomplete model judgments, not independently established ground
truth. Use independently reviewed evaluation data to assess model quality or
compare training approaches.
