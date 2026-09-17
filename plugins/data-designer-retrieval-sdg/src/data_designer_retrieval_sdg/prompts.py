# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts for source profiling, standalone queries, answers and independent judges."""

_JUDGE_RESPONSE_FORMAT = (
    "Return exactly one JSON object matching the supplied response schema inside a single Markdown "
    "code block opened with ```json and closed with ```. Do not return bare JSON or any text outside that code block. "
    "Keep reasons concise; do not repeat the query or source in every justification. "
    "Report unresolved uncertainty as a rejection rather than repeatedly trying to infer missing facts."
)

_SUBJECT_SCOPE_CONTRACT = """\
For a particular company, study or transaction, the query must identify whose
facts are requested. Topic words and years alone do not identify a reporting
company or research study. Generic roles such as "the Group", "the Issuer",
"respondent mothers" or "the agreement" do not identify that subject.
For example, "management compensation 2012 2013" lacks a company;
"Northstar management compensation 2012 2013" supplies one. Do not copy this
illustrative name. A named counterparty alone may still leave the reporting
company or transaction unknown. Do not require a named entity for a genuinely general conceptual question.
Apply this distinction in every source language and query surface.
An entity plus an unnamed attribute is still ambiguous: "Northstar date" does
not say incorporation, authorization, publication or another event. Do not
silently choose one. "Northstar incorporation date" specifies the requested fact.
Likewise, a policy, regulated price or population-specific statistic must name
its jurisdiction or population when different ones could change the answer.
The query's language does not identify its country. Never supply this scope
from the retrieved page or your background knowledge. These examples are
illustrative, not facts to copy. An already unambiguous named policy, or a
general conceptual need, requires no redundant country or event label.
Values do not identify unnamed fields: a date plus "four yes values" does not
say which event or properties are requested. Counting unlabeled flags or finding
rows containing a word describes an unseen representation, not an independently
meaningful retrieval need. Identify the real-world population and named attributes;
never infer missing column semantics. A scoped lookup of organizations authorized
for a named activity is different from looking for rows with "yes". General advice
about searching tables or counting Boolean fields can be valid when that technique
itself is the requested information, rather than a disguised source lookup.
The requested scope must fit the evidence: "all" items in an entire report or
appendix cannot be established from an unverified fragment or continuation page.
When generating, prefer a supported named item or substantive bounded comparison,
not an exhaustive inventory of an unseen whole. When answering or assessing,
reject an exhaustive request if the supplied positives do not establish completeness;
do not silently substitute "all items visible on this page" for the actual request.
"""

_TEMPORAL_EVENT_SCOPE_CONTRACT = """\
Temporal and event scope is part of the information need, not optional context.
For an answer that varies with time, "recent", "upcoming", "currently", "latest"
or "past 50 years" does not identify a period. Include an as-of date, edition,
bounded interval or explicitly identified event that anchors the requested facts.
A named product, organization or recurring conference alone is not a time anchor.
A publisher plus a topical description, such as "the statistical office's study
of industrial employment", is not a specific study edition or time anchor.
Do not reason that the study will supply its period after retrieval: that uses
unseen source context to repair a query. A specific title and identifiable edition
can supply scope; a generic description of a time-varying analysis cannot.
Illustrative contrasts (do not copy these facts into queries):
- "Find the recent Lumen developer conference panel" is ambiguous;
  "Find the Lumen developer conference 2024 panel on storage" is anchored.
- "Upcoming Atlas acceleration feature" is ambiguous;
  "Atlas acceleration feature announced at its 2024 launch" is anchored.
- "Coastal bird decline over the past 50 years" is ambiguous;
  "Coastal bird decline from 1970 to 2020" is anchored.
Use only source-supported anchors when generating; never assume the current date
or infer a missing event from background knowledge. During query generation, if an
anchor is unavailable, choose a different supported evergreen need or fewer/no
queries, not a fabricated date. During answering/judging, never rewrite a query.
Do not demand dates for timeless definitions, named product capabilities or
procedural conditions: "When does Atlas issue a lane-change notification?" asks
for a trigger condition, not a calendar date. Mentioning time is not itself a flaw.
An explanation about a particular crisis/event requires that event to be identified.
"The crisis", "this event" or "the situation" alone does not identify a cause;
repeating those words in an answer does not resolve the missing information.
Preserve each date's event or state: an original entity date and a later transfer
date do not become two dates of the transfer merely because they share a row.
"Dates associated with" an event still asserts a relationship requiring support.
Do not generate a requested count or date-to-event relationship from co-occurrence.
If missing headers or lost line alignment leave a date's meaning uncertain,
omit that need when generating; reject it when answering or assessing sources.
Apply these distinctions in every source language and query surface.
"""

_SOURCE_IMAGE_CONTEXT = """\
Attached image count: {{ images | length }}
Attachment positions below are one-based and follow the actual image order.
{% for unit in retrieval_units %}
{% if unit.images and unit.images[0] in images %}
Attachment {{ images.index(unit.images[0]) + 1 }} source_unit_ids: {{ unit.unit_id }}
{% endif %}
{% endfor %}
Text presence does not mean images are absent. Inspect the attached pixels for
each declared positive unit as well as its supplied text. Image availability is
not proof of useful visual evidence. Do not claim a photograph or chart is absent
merely because native text does not describe it. If an attachment cannot be read
or bound to a positive unit, report that uncertainty; never invent its contents.

"""

_KEYWORD_INFORMATION_NEED_CONTRAST = (
    "Illustrative query contrast (do not copy into generated queries):\n"
    "Keyword form does not make supplied information unknown. "
    '"Cedar service score definition arithmetic mean of uptime and responsiveness" already gives the definition; '
    '"Cedar service score definition uptime responsiveness" leaves the relationship unknown. '
    "Repeating the supplied relationship adds no requested information. "
    "Named inputs may provide scope; an explicitly requested new explanation, comparison or reference may remain unresolved. "
    "Do not invent that request."
)

ARTIFACT_EXTRACTION_SYSTEM_PROMPT = "Extract source facts and assess their usefulness for retrieval data."

ARTIFACT_EXTRACTION_USER_PROMPT = """\
Read the supplied text and any attached images. Extract up to
{max_artifacts_per_type} items of each relevant semantic artifact type.
An empty text field is valid when an image is attached. Use only visible facts.
Use empty artifact lists for absent information. A bare heading such as
"Agenda" does not establish any meeting, agenda items, dates or topics. Mark
both suitability flags false for a heading-only source with no substantive facts.

CONTENT:
{{{{ text }}}}

Always provide retrieval_profile:
- text_retrieval_useful: readable statements (including text in images) can
  support specific, plausible retrieval queries. A title, logo or isolated
  unlabeled value alone is insufficient. Repeated unnamed dates or Boolean flags
  do not become meaningful attributes through repetition; assess what the fields
  mean, not merely whether their values are readable.
- visual_retrieval_useful: charts, cross-cell table relationships, diagrams,
  spatial/layout relationships or informative photographs support a scoped
  information need beyond merely reading prose. Decoration does not qualify.
- visual_content_types: chart, table, diagram, layout, photograph as applicable;
  empty when no meaningful visual content exists.
- suggested_query_operations: a few concrete, verifiable operations supported
  here, e.g. compare named treatment arms for a named endpoint, find a dated
  company's segment revenue, or retrieve a named product's storage guidance.
  Do not suggest looking up unnamed dates, counting unlabeled flags, or searching
  arbitrary rows for a string. If field meanings are missing, suggest only other
  independently meaningful supported needs; an empty list is valid.
- reason: briefly explain suitability or its absence.

With no attached images, visual_retrieval_useful must be false. If both flags
are false, downstream query generation can return no queries. These are
advisory source-level assessments, not proof that any candidate is grounded.
"""

QUERY_GENERATION_SYSTEM_PROMPT = "Generate standalone retrieval queries for realistic pre-retrieval information needs."

QUERY_GENERATION_USER_PROMPT = (
    _KEYWORD_INFORMATION_NEED_CONTRAST
    + "\n\n"
    + _SUBJECT_SCOPE_CONTRACT
    + _TEMPORAL_EVENT_SCOPE_CONTRACT
    + "Use only source-supported identity and dates. If a continuation page lacks the necessary identity, "
    "choose an independently useful general need or return fewer/no queries. Never invent missing document context.\n\n"
    + """\
Generate up to {num_pairs} distinct retrieval queries, without answers or
evidence. Use the source language declared as "{{{{ language }}}}"; when it is
"source", preserve the material's language.
Return no queries for a bare heading or decorative page with no substantive
facts. Never turn a generic title into invented named topics or user context.

SOURCE FACTS AND SUITABILITY:
{{{{ document_artifacts }}}}

SOURCE SEGMENTS:
{{%- for section in sections_structured %}}
{{{{ section }}}}
{{%- endfor %}}

Images may be attached. The facts and suitability profile are suggestions;
verify them against the supplied source. When visual_retrieval_useful is false,
generate useful text retrieval queries instead of forcing visual questions.
Text visible in an image is valid source content. When there is no useful
information need or insufficient readable evidence, return an empty queries
list. A missing profile means unknown suitability, not an automatic rejection.

Every query must:
- Be a plausible information need a user could formulate BEFORE retrieving
  this source. Include sufficient entity, topic, time, population, metric and
  comparison scope to distinguish it in a large same-domain corpus.
- For recurring rankings, billing amounts or usage histories whose answer varies
  by period, include a source-supported edition, date or anchored window. An
  account name or "a twelve-month history" alone does not identify a period.
  If the source lacks that anchor, choose a supported evergreen information need
  or return fewer/no queries; never invent a date or assume the current edition.
  Evergreen needs, such as how a named tariff defines a fee, need no artificial
  date when their meaning is already clear. Apply this in every source language
  and to question, instruction and keyword forms alike.
- Stand alone. Never use source-relative wording such as "the page", "the
  slide", "the document", "the chart", "the table", "the image", "above",
  "below", "shown" or "according to the supplied material".
- Leave the requested information unknown. Do not include the answer,
  an answer-bearing paraphrase, a translated answer, or the conclusion sought.
  A compact keyword string can reveal a complete answer without forming a sentence:
  naming a metric plus its full formula is not a request for that formula.
  Necessary entity, date, input or comparison anchors are allowed;
  omit the requested result, not its scope. If no supported unresolved need remains,
  return fewer/no queries. Apply this across all source languages and query surfaces.
- Be verifiable with the available evidence. Prefer meaningful comparisons,
  trends, filtering and relationships over intricate multi-step calculations.
  A well-scoped factual lookup is useful; maximum complexity is not a goal.
- Avoid requesting clinical, chemical, causal or operational interpretation
  when the source only reports an observation. Never invent unnamed groups,
  metrics or a comparison period to make a query sound more specific.
- Ask about the quantities or relationships encoded by a chart, not its mark geometry.
  "Which bar extends further?" is a presentation-reading task, even when a
  chart's title, categories and date are named. Ask about the underlying named
  quantities instead, but only when the source supplies the necessary scope.
  This is not a ban on appearance: genuine needs about physical objects,
  diagram structure or visualization design remain useful retrieval queries.
  Apply this distinction in the source language; do not force visual questions.

Use as many distinct surfaces as supported (three for three or more queries,
two for two): question (natural interrogative), instruction (imperative search
request), keyword (compact search terms). Label the actual form. An instruction
or keyword query must not simply be a question relabeled. Truthfulness takes
priority over filling a quota or achieving diversity.
Examples of forms (illustrative only; do not copy these facts into a query):
question: "What storage temperature does Drug X require?"
instruction: "Find storage requirements for Drug X."
keyword: "Drug X storage temperature requirements"
Use different information needs across candidates, not three paraphrases of
one question. When requesting three queries, try one of each form.
Follow this surface schedule for the queries you can support:
{query_surface_schedule}
In particular, keyword candidates must contain compact search terms and
instruction candidates must start with a search action such as Find, Retrieve
or Locate (in the source language). Do not default to all questions.

Requested query-type mix (soft targets, adapt to available evidence):
multi_hop={query_counts_multi_hop}, structural={query_counts_structural},
contextual={query_counts_contextual}.
Requested reasoning-type mix: factual={reasoning_counts_factual},
relational={reasoning_counts_relational}, inferential={reasoning_counts_inferential},
temporal={reasoning_counts_temporal}, procedural={reasoning_counts_procedural},
visual={reasoning_counts_visual}, causal={reasoning_counts_causal}.
These are independent labels. Multi-hop requires {min_hops}-{max_hops} evidence
steps actually present in the supplied segments. On a single-page source do
not invent other pages or segments. Aim for complexity {min_complexity}-5 where
supported, otherwise prefer a simpler useful query or return fewer queries.

Schema reminder: query_type is ONLY "multi_hop", "structural" or "contextual".
"factual" and "visual" are reasoning_type values, never query_type values.
"question", "instruction" and "keyword" are query_surface values only.
"""
)

QUERY_QUALITY_SYSTEM_PROMPT = (
    "You independently judge retrieval queries without seeing their source or answer.\n\n" + _JUDGE_RESPONSE_FORMAT
)

QUERY_QUALITY_USER_PROMPT = (
    _KEYWORD_INFORMATION_NEED_CONTRAST
    + "\n\n"
    + _SUBJECT_SCOPE_CONTRACT
    + _TEMPORAL_EVENT_SCOPE_CONTRACT
    + "Mark retrieval_discriminative=false for unresolved temporal or event scope. "
    "A plausible topic and named product do not override that failure. Judge the query alone; "
    "do not assume a particular announcement or event will become clear after retrieval.\n\n"
    + "Mark retrieval_discriminative=false when the particular subject remains unidentified. "
    "Do not imagine a source-specific company, study or transaction to make the query clear.\n\n"
    + "Before assigning the booleans, identify the requested fact and its material scope using only "
    "words in the query. If you must choose an unstated event, jurisdiction or period to make it "
    "precise, set retrieval_discriminative=false and name that missing anchor in reason. "
    "A plausible interpretation is not the same as an identified information need.\n\n"
    + """\
Judge every candidate below using only its query text and declared surface.
Do not infer or reconstruct the source or answer.

For each candidate_index, assess:
- standalone_query: understandable without source-relative context
- plausible_information_need: something a user could need before retrieval
- retrieval_discriminative: sufficiently scoped for a large same-domain corpus
- query_surface_correct: a question, instruction, or keyword query as declared
- source_language_preserved: written in the declared source language

For a query requesting unnamed field values or identifying a row by its count of
unlabeled flags, mark plausible_information_need=false and
retrieval_discriminative=false. A concrete date or literal word does not supply
the missing real-world meaning or population. Do not reinterpret a source lookup
as a general conceptual question about table-search techniques.

Using only the query, check whether its stated information need is
already completed by facts or relationships it supplies. Mark
plausible_information_need=false when it merely requests those supplied facts again,
including a formula expressed as search keywords. Do not reconstruct an unavailable
answer or reject merely for numbers, named inputs or facts you know from memory.
A genuinely new explanation, comparison or other unknown information can remain
valid; do not invent such a request to rescue it. Apply this in every source
language and across question, instruction and keyword forms.

Declared source language: {{ language }}
When this is "source", the actual source language is unavailable to this
blind stage: do not guess it, mark this criterion true provisionally. The
grounding judge must compare the query against the actual source language.

Reject bare ordinals, generic unnamed metrics, and missing entity, topic, time,
population or comparison anchors when those omissions make the query ambiguous.
Do not require every possible anchor if the information need is already clear.
Recurring rankings, billing amounts and usage histories require
an edition, date or anchored window when the requested answer varies by period.
An account/entity name or unanchored
duration alone does not resolve this ambiguity. Mark retrieval_discriminative=false
for such ambiguous queries; do not infer a missing period from an unavailable source
or assume "current" or "latest". Do not impose dates on genuinely evergreen needs,
such as the meaning of a named billing fee. Apply this in every source language
and across question, instruction and keyword forms.
Distinguish underlying information needs from reading a chart's mark geometry.
A question asking which bar is longer or extends further is a presentation-reading
task, not a plausible pre-retrieval information need, even with a named chart,
categories and date. Mark plausible_information_need false for such tasks.
Do not reject merely for mentioning a chart: comparing named quantities,
visualization design guidance, diagram structure, and genuine appearance-based
lookups about physical objects can be legitimate. Apply this distinction in
every source language without reconstructing the unavailable source.
Return exactly one evaluation for each zero-based candidate index.

{% for candidate in deduplicated_queries %}
Candidate {{ loop.index0 }}
Surface: {{ candidate.query_surface }}
Query: {{ candidate.question }}
{% endfor %}
"""
)

ANSWER_GENERATION_SYSTEM_PROMPT = "Answer immutable retrieval queries using only their positive source evidence."

ANSWER_GENERATION_USER_PROMPT = (
    _SOURCE_IMAGE_CONTEXT
    + _SUBJECT_SCOPE_CONTRACT
    + _TEMPORAL_EVENT_SCOPE_CONTRACT
    + """\
Answer the selected queries below without rewriting them. Return answers keyed
by the zero-based candidate_index shown here; do not return query text.
If the supplied material cannot answer a query, omit that answer. Do not invent
an answer or repair a query to make it answerable.
"Not specified in the provided material" is not a useful answer for positive
retrieval training. Omit the candidate instead. For example, if a source only
says "Agenda", every question about its agenda items must be omitted.

{% for item in query_selection.selected %}
Candidate {{ loop.index0 }}: {{ item.query.question }}
{% endfor %}

<retrieval_units>
{% for unit in retrieval_units %}
unit_id={{ unit.unit_id }}; segment_id={{ unit.segment_id }}; text={{ unit.text }}
{% endfor %}
</retrieval_units>

Images may be attached. Cite only stable unit_id and segment_id values present
above. Identify the independently useful retrieval positives, with concise
evidence supporting every material answer claim. Preserve the source language
(declared: {{ language }}; "source" means use the material's language).

Report observed values, labels, trends and explicitly supported relationships.
Do not add outside domain knowledge or turn a measured assay change into a
chemical/clinical explanation. For arithmetic, state operands and units; use
precision justified by the visible data. Avoid unnecessary interpretation.
Be concise but complete for the requested information, retaining necessary scope,
units and every requested list member. Do not add adjacent facts merely because
they share a list, column or paragraph. Preserve semantic categories:
geographic coverage is not an asset class, and a location label must not become
a financial instrument. Distinguish requested
types from other labels instead of copying a whole column. Apply this in the
source language; brevity must not omit facts needed to resolve the query.
Use hop_count=1 and hop_contexts=[] unless multiple evidence steps are actually
needed; for multi-hop, segment_ids is the union of the hop segment IDs.
"""
)

SOURCE_ASSESSMENT_SYSTEM_PROMPT = (
    "Independently read retrieval sources without seeing generated answers or evidence.\n\n" + _JUDGE_RESPONSE_FORMAT
)

SOURCE_ASSESSMENT_USER_PROMPT = (
    _SOURCE_IMAGE_CONTEXT
    + _SUBJECT_SCOPE_CONTRACT
    + _TEMPORAL_EVENT_SCOPE_CONTRACT
    + """\
Answer each retrieval query independently from its declared positive sources.
Language declaration: {{ language }}. The value "source" means preserve the
actual source language, not a language name. You are not given candidate answers
or generated evidence. Generated interpretations are never authoritative source text.

{% for qa_pair in deduplicated_qa_pairs %}
Candidate {{ loop.index0 }}
QUESTION: {{ qa_pair.question }}
POSITIVE UNIT IDS: {{ qa_pair.positive_unit_ids }}
{% endfor %}

<retrieval_units>
{% for unit in retrieval_units %}
unit_id={{ unit.unit_id }}; segment_id={{ unit.segment_id }}
supplied_text_present={{ (unit.text | trim | length > 0) | lower }}
<supplied_source_text>{{ unit.text }}</supplied_source_text>
{% endfor %}
</retrieval_units>

Return one assessment per candidate with its zero-based candidate_index and
exactly its declared positive_unit_ids. Assess only those positives, not a
different unit that happens to answer the query. Include unsupported candidates:
- independent_answer: the requested information, independently read from sources,
  in the source language; null when the source cannot establish an answer.
- supporting_evidence: explain which positive units support each material fact,
  including entity, time, category and units; null when support is unavailable.
- answerable: the positives actually supply the requested information. Absence
  statements, circular restatements and guesses are not answers. Set false if
  any material fact needed to answer the query is unreadable or uncertain.
- uncertainty_reasons: list every material ambiguity or missing/unreadable fact.
  An empty list means no such uncertainty, not permission to fill missing cells.
- positive_source_relevant: every named positive unit is an independently useful
  retrieval result, not merely where the query happened to originate
- evidence_modality: text_only if positive-unit text fully supports the answer;
  otherwise image_grounded only if attached pixels are necessary. Text read
  from an image with no transcription still requires image access. This label
  describes the evidence required, not the source's classification profile.
  When supplied_text_present=false, text_only is impossible for that positive:
  choose image_grounded if the pixels support the answer, otherwise reject it.
  A table cell or sentence read from pixels is NOT supplied_source_text.
- native_text_evidence: for text_only, list {unit_id, quote} objects copied
  verbatim from each named positive unit's supplied_source_text. The quotes
  must collectively support every material answer claim without looking at the images.
  Check this even when the supplied text is nonempty: a title, caption or topic
  word is not support for details available only in pixels. If any necessary
  claim lacks native-text support, do not label the candidate text_only.
  Test relationships as well as words: a flattened list of labels does not
  establish diagram containment, arrow direction, row membership or which value
  belongs to which category. Mentally hide the image: can the quoted text alone
  establish every requested relationship? If not, use image_grounded when pixels
  resolve it; otherwise reject. Do not assume reading order preserves layout.
  Never quote your independent_answer or supporting_evidence as source text.
  Do not transcribe pixels into this field. For image_grounded, quotes are optional;
  any supplied quote must still occur in that positive unit's supplied text.
  Preserve wording, case and numbers; only whitespace differences are allowed.
- source_language_preserved: compare the query with the actual source language

Question, instruction and keyword query forms are all valid for retrieval;
do not penalize a keyword query merely for not being a full sentence. If the
source cannot supply the requested information, positive_source_relevant must
be false. Do not accept a page just because it shares a topic word with the query.

Do not use the positive source to fill an identity missing from the query.
For a query requesting a particular company's, study's or transaction's facts,
check that the query identifies that subject rather than just a topic or generic
role. If not, set positive_source_relevant=false and record the ambiguity.
A general conceptual question need not name a particular subject.
If a query leaves a material period/event ambiguous, set positive_source_relevant=false
and record it in uncertainty_reasons; a matching topic cannot repair its scope.
If the positive sources themselves never identify the event or anchor needed for
the requested answer, set answerable=false and independent_answer=null. For example,
"this crisis encouraged automation" does not establish which crisis or its context.
Do not fill that gap from nearby unrelated units, generated artifacts or world knowledge.

When the attached image count is zero, no pixels are available. Never infer unseen pixels. Reject invented
metric, category, population, time or comparison semantics. Treat interpretations
and implications as material claims requiring explicit evidence. Accept equivalent
units and appropriate rounding, but do not apply a blanket numeric tolerance:
precise table values must match; estimates from charts must reflect their resolution.
Independently read every needed number, date and table cell from the actual
positive source, without inferring values from nearby entries or a plausible trend.
Keep the independent answer concise but complete for the query, including
necessary context and every requested list member, not adjacent unrelated facts.
Preserve semantic categories: geographic coverage is not an asset class. A shared
list or column does not prove its entries are all the requested type. Independently
identify which labels answer the query; do not copy the entire list indiscriminately.
If the category needed to answer cannot be established, record that uncertainty
and set answerable=false. Apply this in the source language without dropping
requested facts merely to shorten the reading.
Check row/column alignment, labels and units; do not endorse a plausible digit
or date by collecting all values in the same row. Inspect within-cell line
alignment too: a date aligned to the original entity is not automatically a date
of its later transfer. Missing date semantics cannot be supplied by the query
or by a plausible interpretation. If a material cell or value is uncertain
or unreadable, list it in uncertainty_reasons and set answerable=false.
High confidence does not replace this verification.
"""
)

QA_EVALUATION_SYSTEM_PROMPT = (
    "Compare candidate claims with an independent source assessment, without reinterpreting sources.\n\n"
    + _JUDGE_RESPONSE_FORMAT
)

QA_EVALUATION_USER_PROMPT = (
    _KEYWORD_INFORMATION_NEED_CONTRAST
    + "\n\n"
    + """\
Verify each immutable candidate against its independent source assessment.
You receive no raw source text or images. Do not reconstruct or reinterpret the
source to make a candidate correct. The independent reading is a model assessment,
not infallible ground truth and never authoritative source text. If it is missing,
uncertain, unanswerable or inconsistent, reject; do not resolve uncertainty in favor
of the candidate. Never rewrite the candidate or the independent reading.

{% for qa_pair in deduplicated_qa_pairs %}
Candidate {{ loop.index0 }}
QUESTION: {{ qa_pair.question }}
ANSWER: {{ qa_pair.answer }}
CLAIMED EVIDENCE: {{ qa_pair.evidence }}
POSITIVE UNIT IDS: {{ qa_pair.positive_unit_ids }}
{% endfor %}

<independent_assessments>
{{ source_assessments | string }}
</independent_assessments>

Match by candidate_index, not list order. Return one evaluation per candidate.
Score relevance, accuracy, context_support, clarity and overall on 1-10 scales,
and supply improvements. High scores never override a failed mandatory criterion:
- answer_grounded: every material answer claim AND claimed evidence is supported
  by the independent reading. Reject contradictions, unsupported extra claims,
  and mismatched entity, category, population, time or comparison scope.
  A broader category and one of its subcategories are not interchangeable.
  If one reading excludes an entire category and the other only a subtype,
  that is a material disagreement, even when most labels agree. Reject rather
  than choosing which reading is probably right without the source.
- unsupported_claims: list material mismatches and unsupported answer/evidence
  claims, including precise numbers, dates, units and row/column alignment.
  Never favor a candidate value just because its answer and evidence repeat it.
- answer_resolves_query: supplies the requested information without material
  omissions. An absence statement or circular restatement fails, even if true.
  An unidentified "crisis/event" is not a resolved explanation. Set this false
  when an essential temporal/event anchor is missing, even if both readings repeat
  the same ambiguous wording. Do not infer the missing anchor or silently interpret
  "recent", "upcoming" or an unanchored "past N years" as a specific period.
- answer_not_revealed_by_query: no direct, indirect or translated answer leakage.
- positive_source_relevant, evidence_modality, source_language_preserved and
  native_text_evidence: preserve these fields from the independent assessment
  exactly. Do not add new quotes, change the modality or rescue negative source
  findings. Missing/duplicate assessments or mismatched positive IDs must fail.

Compare what the query requests, already supplies, and the answer newly supplies.
Completing grammar, expanding abbreviations, translating or paraphrasing the same
result adds no requested fact. If the answer only restates a result already in
the query, set answer_not_revealed_by_query=false and answer_resolves_query=false,
even when the independent reading agrees and grounding scores are high.
Scope anchors are not leaked answers to a genuinely unresolved request. Do not
invent an unstated request for explanation/evidence or add adjacent facts to rescue
a completed information need. Apply this across languages and all query surfaces.

Compare meaning, not verbatim strings: equivalent phrasing, spelled-out numbers
and correct unit conversions are allowed. Precise table values must agree;
do not use blanket numeric tolerance or treat different dates/categories as
equivalent. Chart estimates must respect the independent reading's uncertainty.
Preserve semantic categories: geographic coverage is not an asset class, even if
both readings repeat the same list. A category substitution fails answer_grounded;
record it in unsupported_claims. Do not rescue it by reinterpreting unseen sources.
Question, instruction and keyword forms are permitted, but form alone never establishes a valid unresolved need.
Apply the same leakage and resolution checks to each.
"""
)
