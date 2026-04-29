# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for the retriever SDG pipeline.

All long-form system and user prompts are centralised here as module-level
constants so that ``pipeline.py`` stays concise and the prompts are easy to
review or override.
"""

# ---------------------------------------------------------------------------
# Artifact extraction
# ---------------------------------------------------------------------------

ARTIFACT_EXTRACTION_SYSTEM_PROMPT = "You are an expert at analyzing documents and extracting semantic artifacts."

ARTIFACT_EXTRACTION_USER_PROMPT = """\
Analyze the following content and extract semantic artifacts that would be \
valuable for generating high-quality question-answer pairs.

Note: The content may contain multiple documents bundled together \
(separated by "=== Document Boundary ==="). \
If multiple documents are present, identify cross-document relationships \
and connections.

CONTENT:
{{{{ text }}}}

ARTIFACT TYPES TO EXTRACT:
- key_concepts: Core ideas and concepts discussed in the document(s)
- relationships: Connections and relationships between different concepts \
(including cross-document relationships)
- themes: Overarching themes and topics
- entities: Specific entities, people, organizations, or items mentioned
- processes: Processes, workflows, or procedures described
- insights: Key insights, conclusions, or findings
- technical_terms: Technical terminology and specialized vocabulary
- contextual_factors: Contextual information that provides background

INSTRUCTIONS:
1. Extract up to {max_artifacts_per_type} artifacts for each relevant type
2. Focus on the most significant and informative elements
3. Provide clear, concise descriptions for each artifact
4. Include context about why each artifact is important
5. Ensure artifacts are specific and actionable for Q&A generation
6. For multi-document bundles, pay special attention to relationships \
and comparisons between documents
"""

# ---------------------------------------------------------------------------
# QA generation
# ---------------------------------------------------------------------------

QA_GENERATION_SYSTEM_PROMPT = (
    "You are an expert at extracting question and answer pairs from provided context/transcript/segments."
)

QA_GENERATION_USER_PROMPT = """\
You are an expert at extracting question and answer pairs from provided \
context/transcript/segments.

<document_facts_block>:
{{%- if document_artifacts.key_concepts %}}
<key_concepts>
{{%- for item in document_artifacts.key_concepts %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</key_concepts>
{{%- endif %}}

{{%- if document_artifacts.relationships %}}
<relationships>
{{%- for item in document_artifacts.relationships %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</relationships>
{{%- endif %}}

{{%- if document_artifacts.themes %}}
<themes>
{{%- for item in document_artifacts.themes %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</themes>
{{%- endif %}}

{{%- if document_artifacts.entities %}}
<entities>
{{%- for item in document_artifacts.entities %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</entities>
{{%- endif %}}

{{%- if document_artifacts.processes %}}
<processes>
{{%- for item in document_artifacts.processes %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</processes>
{{%- endif %}}

{{%- if document_artifacts.insights %}}
<insights>
{{%- for item in document_artifacts.insights %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</insights>
{{%- endif %}}

{{%- if document_artifacts.technical_terms %}}
<technical_terms>
{{%- for item in document_artifacts.technical_terms %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</technical_terms>
{{%- endif %}}

{{%- if document_artifacts.contextual_factors %}}
<contextual_factors>
{{%- for item in document_artifacts.contextual_factors %}}
- {{{{ item.text }}}}: {{{{ item.description }}}}
{{%- endfor %}}
</contextual_factors>
{{%- endif %}}
</document_facts_block>

<context_block>:
{{%- for section in sections_structured %}}
{{{{ section }}}}

{{%- endfor %}}
</context_block>

Guidelines:
1. Generate questions with varying complexity levels between 1 (simple) and \
5 (complex):
   - All questions MUST require understanding connections between different \
parts of the context/transcript/segments
   - Questions should test deep understanding, not simple facts
   - Do not mention the existence of a context/transcript in the generated \
question like "in the transcript", "from the given context", or \
"in Segment 148". Produce a natural, standalone question.
   - Only use facts present in the provided context/transcript; if missing, \
say you cannot generate a question.
   - Example: "How does the speaker's initial explanation of X relate to \
the later implementation of Y?"

2. Question Types to Generate (for the "query_type" field - ONLY these 3 \
values allowed):
   - "multi_hop" ({query_counts_multi_hop} questions): Connect \
{min_hops}-{max_hops} separated segments
   - "structural" ({query_counts_structural} questions): Focus on \
relationships between concepts
   - "contextual" ({query_counts_contextual} questions): Require \
surrounding context to understand
   - Use the cross-part context snippets to connect evidence that lives \
outside the current transcript section

3. Reasoning Types to Include (for the "reasoning_type" field - ONLY these \
7 values allowed):
   - "factual" ({reasoning_counts_factual} questions): Ask for complex \
facts that require synthesizing multiple pieces of information \
(NOT simple lookups)
   - "relational" ({reasoning_counts_relational} questions): Ask how data \
points compare or correlate across different segments
   - "inferential" ({reasoning_counts_inferential} questions): Ask about \
conclusions or implications requiring synthesis
   - "temporal" ({reasoning_counts_temporal} questions): Ask about changes \
or events over time across segments
   - "procedural" ({reasoning_counts_procedural} questions): Ask about \
complex multi-step processes or guidelines
   - "visual" ({reasoning_counts_visual} questions): Ask about visual \
details requiring cross-reference
   - "causal" ({reasoning_counts_causal} questions): Ask about cause-effect \
chains spanning segments

   Example COMPLEX questions by reasoning type:
   - Factual: "What is the total combined budget allocation across all \
departmental initiatives mentioned, and how does it relate to the overall \
fiscal year target?"
   - Relational: "How does the performance metric achieved in Q2 compare to \
both the initial baseline and the revised targets that were set?"
   - Inferential: "Based on the challenges outlined and the proposed \
solutions, what unstated assumptions underlie the strategic pivot?"
   - Temporal: "How did the implementation timeline evolve from the initial \
proposal through the mid-year review to the final execution phase?"
   - Procedural: "What is the complete approval workflow including standard \
requirements, exceptions, and escalation processes?"
   - Visual: "How do the visual elements presented relate to the verbal \
descriptions provided, and what discrepancies exist between them?"
   - Causal: "What chain of events, starting from the initial decision, led \
through various complications to the final outcome?"

4. IMPORTANT - Orthogonal Distributions (query_type and reasoning_type are \
SEPARATE fields):
   - Each question must have BOTH a query_type \
(multi_hop/structural/contextual) AND a reasoning_type \
(factual/relational/inferential/temporal/procedural/visual/causal)
   - These are TWO DIFFERENT fields - do NOT put reasoning types in the \
query_type field!
   - For example: A question can be query_type="multi_hop" with \
reasoning_type="procedural"
   - Ensure the final distribution matches both specified percentages

5. **IMPORTANT - Segment Identification**:
   - The content below contains segments formatted as \
"Segment N (HH:MM:SS - HH:MM:SS): text" or \
"Segment N [Doc: doc_id] (HH:MM:SS - HH:MM:SS): text" where N starts from 1
   - The "[Doc: doc_id]" tag indicates which document the segment belongs to \
(for multi-document bundles)
   - For each question-answer pair you generate, identify ALL segment numbers \
FROM which the question is derived
   - These segments are the source material that should be retrieved when \
someone asks this question
   - Record these segment numbers in the "segment_ids" field as a list of \
integers (e.g., [1, 4, 8])
   - For multi-document bundles, prefer questions that span multiple \
documents to maximize cross-document reasoning
   - For multi-hop questions:
     * The top-level "segment_ids" should be the UNION of all segment IDs \
across all hops
     * Each hop in "hop_contexts" should specify its own "segment_ids" list
     * Example: If hop 1 uses [1, 3] and hop 2 uses [6, 8], then top-level \
segment_ids should be [1, 3, 6, 8]
     * For multi-document bundles, try to have different hops reference \
different documents

6. For Each Question:
   - Must have complexity level {min_complexity} or higher
   - Generate the question FROM the identified segments (these segments are \
the source material)
   - Multi-hop questions must specify hop_count ({min_hops}-{max_hops})
   - Provide hop_contexts: a list where each hop includes "hop_number", \
"segment_ids" (the source segments for this hop), and "summary" \
(a concise summary describing the supporting segments).

7. Generate {num_pairs} distinct question and answer pairs.

The output should be a JSON object with a "pairs" field containing an array \
of {num_pairs} objects, where each object contains:
  - "question": the question, requiring understanding of the \
contexts/transcripts/segments without explicitly referencing the \
context/transcript/segments in the question
  - "answer": comprehensive answer from the contexts/transcripts/segments \
without explicitly referencing the context/transcript/segments in the answer
  - "question_complexity": numeric score {min_complexity}-5
  - "query_type": MUST be exactly one of these three values: "multi_hop", \
"structural", or "contextual" (NO other values allowed - do NOT use \
reasoning types here)
  - "reasoning_type": MUST be exactly one of these seven values: "factual", \
"relational", "inferential", "temporal", "procedural", "visual", or \
"causal" (this is DIFFERENT from query_type)
  - "segment_ids": list of segment numbers (e.g., [1, 4, 8]) that are the \
source material for this question (these should be retrieved when the \
question is asked)
  - "hop_count": number of hops ({min_hops}-{max_hops}) for multi_hop \
questions, or 1 for non-multi-hop questions
  - "hop_contexts": array of hop detail objects with "hop_number", \
"segment_ids", "summary"

CRITICAL: "query_type" and "reasoning_type" are TWO SEPARATE FIELDS with \
different allowed values. Do NOT mix them up:
  - query_type can ONLY be: "multi_hop", "structural", "contextual"
  - reasoning_type can ONLY be: "factual", "relational", "inferential", \
"temporal", "procedural", "visual", "causal"
"""

# ---------------------------------------------------------------------------
# QA evaluation
# ---------------------------------------------------------------------------

QA_EVALUATION_SYSTEM_PROMPT = "You are an expert evaluator of question-answer pairs."

QA_EVALUATION_USER_PROMPT = """\
You are an expert evaluator of question-answer pairs.

You will evaluate multiple question-answer pairs from a document.

{% for qa_pair in deduplicated_qa_pairs %}
=== QA Pair {{ loop.index }} ===

QUESTION: {{ qa_pair.question }}

ANSWER: {{ qa_pair.answer }}

CONTEXT (Relevant Segment IDs): {{ qa_pair.segment_ids }}

{% endfor %}

<segments>
{% for chunk in chunks %}
- Segment {{ chunk.chunk_id }}: {{ chunk.text }}
{% endfor %}
</segments>

Evaluate EACH of the {{ deduplicated_qa_pairs | length }} QA pairs above.
"""
