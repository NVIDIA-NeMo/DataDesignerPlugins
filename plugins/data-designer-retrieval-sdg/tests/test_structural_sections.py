# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-unit structural sectioning: evidence alignment, fallback and cached TOCs."""

import json

import pytest
from test_multimodal_sdg import fixture_config

from data_designer_retrieval_sdg.multimodal.inference import DataDesignerInference
from data_designer_retrieval_sdg.multimodal.models import GenerationContext, TOCConfirmation
from data_designer_retrieval_sdg.multimodal.sections import (
    build_sections,
    section_contexts,
    toc_candidate,
    verify_section_coverage,
)
from data_designer_retrieval_sdg.multimodal.storage import fingerprint, write_json
from data_designer_retrieval_sdg.retrieval.models import RetrievalSource


def sources_with_text(texts, *, pages=True, document="manual", language="en"):
    """Use opaque IDs so accidental identifier/page-number parsing cannot work."""
    return [
        RetrievalSource(
            unit_id=f"{document}/opaque-{i * 17 + 43}",
            document_id=document,
            text=text,
            page_number=i + 1 if pages else None,
            language=language,
        )
        for i, text in enumerate(texts)
    ]


def confirmed(sources):
    """Supply explicit synthetic confirmations; never make a model call in fixtures."""
    return {
        s.unit_id: {"has_table_of_contents": True, "explanation": "Fixture"} for s in sources if toc_candidate(s.text)
    }


@pytest.mark.parametrize("length", [1, 4, 5, 6, 10, 11, 25])
def test_fixed_fallback_flushes_all_units_including_original_tail_bug_cases(tmp_path, length):
    sources = sources_with_text([f"Source content {i}" for i in range(length)])
    before = [s.model_dump() for s in sources]
    sections, audit = build_sections(sources, fixture_config(tmp_path))
    assert [k for c in sections for k in c.unit_ids] == [s.unit_id for s in sources]
    assert all(len(c.unit_ids) <= 5 for c in sections)
    assert audit["coverage"]["exact_ordered_coverage"]
    assert [s.model_dump() for s in sources] == before


def test_heading_boundaries_map_to_whole_units_and_ignore_code_and_repeated_headers(tmp_path):
    texts = ["Text" for _ in range(16)]
    texts[1] = "```markdown\n# Example code heading\n```"
    texts[2] = texts[3] = "# Repeated running header\nText"
    texts[6] = "Preceding paragraph\n## Operating conditions\nDetails"
    texts[13] = "# Appendix\nDetails"
    sources = sources_with_text(texts)
    sections, audit = build_sections(sources, fixture_config(tmp_path))
    assert [len(c.unit_ids) for c in sections] == [6, 7, 3]
    assert [b["index"] for b in audit["documents"][0]["boundaries"]] == [6, 13]
    assert sections[1].unit_ids[0] == sources[6].unit_id
    assert [k for c in sections for k in c.unit_ids] == [s.unit_id for s in sources]


def test_confirmed_toc_infers_offset_and_rejects_bad_entries_without_losing_valid_tail(tmp_path):
    texts = ["Text" for _ in range(14)]
    texts[0] = (
        "Table of contents\n1. Introduction ... 1\nOperating conditions ... 6\nBroken range ... 999\nAppendix ... 12\nBackward entry ... 2\nInvalid ... -1"
    )
    texts[2] = "# Introduction\nDetails"
    texts[7] = "## Operating conditions\nDetails"
    sources = sources_with_text(texts)
    sections, audit = build_sections(sources, fixture_config(tmp_path), confirmed(sources))
    doc = audit["documents"][0]
    assert doc["offset"] == 2
    assert {r["unit_id"] for r in doc["accepted"]} == {sources[i].unit_id for i in [2, 7, 13]}
    assert {r["reason"] for r in doc["rejected"]} == {
        "nonpositive_page",
        "out_of_range_or_unmapped_page",
        "backward_source_boundary",
    }
    assert any(c.unit_ids[0] == sources[13].unit_id for c in sections)
    assert [k for c in sections for k in c.unit_ids] == [s.unit_id for s in sources]


@pytest.mark.parametrize("pages", [True, False])
def test_unanchored_confirmed_toc_does_not_trust_printed_offsets(tmp_path, pages):
    sources = sources_with_text(["Contents\nAlpha ... 1\nBeta ... 3\nGamma ... 8"] + ["Text"] * 9, pages=pages)
    sections, audit = build_sections(sources, fixture_config(tmp_path), confirmed(sources))
    assert [len(c.unit_ids) for c in sections] == [5, 5]
    assert not audit["documents"][0]["accepted"]
    assert len(audit["documents"][0]["rejected"]) == 3


def test_missing_page_metadata_uses_matched_titles_only(tmp_path):
    texts = ["Contents\nAlpha ... 1\nBeta ... 3\nGamma ... 7"] + ["Text"] * 9
    texts[2], texts[5] = "# Alpha", "# Beta"
    sources = sources_with_text(texts, pages=False)
    _, audit = build_sections(sources, fixture_config(tmp_path), confirmed(sources))
    doc = audit["documents"][0]
    assert doc["offset"] is None
    assert {r["unit_id"] for r in doc["accepted"]} == {sources[2].unit_id, sources[5].unit_id}
    assert doc["rejected"][0]["reason"] == "unverified_page_offset"


def test_disagreeing_offsets_and_ambiguous_titles_never_project_numeric_boundaries(tmp_path):
    texts = ["Contents\nAlpha ... 1\nBeta ... 2\nGamma ... 5\nShared ... 6"] + ["Text"] * 11
    texts[2], texts[5], texts[7], texts[8] = "# Alpha", "# Beta", "# Shared", "# Shared"
    sources = sources_with_text(texts)
    _, audit = build_sections(sources, fixture_config(tmp_path), confirmed(sources))
    doc = audit["documents"][0]
    assert doc["offset"] is None
    assert all(r["method"] == "toc_title" for r in doc["accepted"])
    assert {r["title"] for r in doc["rejected"]} == {"Gamma", "Shared"}


def test_backward_page_metadata_disables_projection(tmp_path):
    texts = ["Contents\nAlpha ... 1\nBeta ... 2\nGamma ... 5"] + ["Text"] * 9
    texts[2], texts[5] = "# Alpha", "# Beta"
    sources = sources_with_text(texts)
    sources[4] = sources[4].model_copy(update={"page_number": 1})
    sections, audit = build_sections(sources, fixture_config(tmp_path), confirmed(sources))
    assert audit["documents"][0]["offset"] is None
    assert [k for c in sections for k in c.unit_ids] == [s.unit_id for s in sources]


def test_document_and_language_partitions_and_identity_are_preserved(tmp_path):
    sources = sources_with_text(["# Introduction", "Text"] * 6)
    sources += sources_with_text(
        ["Sommaire\nAlpha ... 1\nBeta ... 3\nGamma ... 6"] + ["Texte"] * 6, document="other", language="fr"
    )
    config = fixture_config(tmp_path)
    sections, audit = build_sections(sources, config, confirmed(sources))
    assert audit["coverage"]["covered_units"] == len(sources)
    assert len(audit["documents"]) == 2
    assert build_sections(sources, config, confirmed(sources))[0] == sections


@pytest.mark.parametrize("bad", ["missing", "duplicate", "reversed", "cross_document"])
def test_coverage_guard_rejects_corrupted_memberships(tmp_path, bad):
    sources = sources_with_text(["One", "Two", "Three"])
    config = fixture_config(tmp_path)
    sections, _ = build_sections(sources, config)
    if bad == "missing":
        sections[0] = sections[0].model_copy(update={"unit_ids": sections[0].unit_ids[:-1]})
    elif bad == "duplicate":
        sections.append(GenerationContext(context_id="extra", unit_ids=[sources[0].unit_id], language="en"))
    elif bad == "reversed":
        sections[0] = sections[0].model_copy(update={"unit_ids": list(reversed(sections[0].unit_ids))})
    else:
        sources[1] = sources[1].model_copy(update={"document_id": "other"})
    with pytest.raises(ValueError, match="coverage|crosses"):
        verify_section_coverage(sections, sources, 10)


class CachedTOCInference(DataDesignerInference):
    """Exercise real cache/schema orchestration with an in-memory model substitute."""

    def __init__(self, root, config):
        super().__init__(root, config)
        self.calls = 0

    def run_batch(self, pending):
        self.calls += 1
        for key, request in pending:
            assert request.schema is TOCConfirmation and request.role == "judge"
            response = TOCConfirmation(explanation="Actual contents list", has_table_of_contents=True).model_dump()
            write_json(
                self.root / "responses" / f"{key}.json",
                {"request_id": key, "response": response, "response_sha256": fingerprint(response)},
            )


def test_toc_confirmation_uses_existing_cache_and_writes_decision_audit(tmp_path):
    config = fixture_config(tmp_path)
    texts = ["Contents\nAlpha ... 1\nBeta ... 3\nGamma ... 5"] + ["Text"] * 7
    texts[2], texts[4] = "# Alpha", "# Beta"
    sources = sources_with_text(texts)
    inference = CachedTOCInference(tmp_path / "inference", config)
    first = section_contexts(sources, config, inference)
    second = section_contexts(sources, config, inference)
    assert second == first and inference.calls == 1
    audit = json.loads((config.output_dir / "planning/section_boundaries.json").read_text())
    assert audit["documents"][0]["offset"] == 2
    assert audit["coverage"]["exact_ordered_coverage"]
    assert len(list((inference.root / "responses").glob("*.json"))) == 1


def test_denied_confirmation_does_not_enable_toc_offsets(tmp_path):
    texts = ["Contents\nAlpha ... 1\nBeta ... 3\nGamma ... 5"] + ["Text"] * 7
    texts[2], texts[4] = "# Alpha", "# Beta"
    sources = sources_with_text(texts)
    _, audit = build_sections(sources, fixture_config(tmp_path), {sources[0].unit_id: {"has_table_of_contents": False}})
    assert audit["documents"][0]["accepted"] == []
    assert all(b["evidence"] == ["heading"] for b in audit["documents"][0]["boundaries"])


@pytest.mark.parametrize("explicit", [False, True])
def test_workflow_confirms_automatic_tocs_but_preserves_explicit_contexts(tmp_path, monkeypatch, explicit):
    from test_multimodal_sections import SectionInference

    from data_designer_retrieval_sdg.multimodal.workflow import run_multimodal_sdg

    config = fixture_config(tmp_path)
    config = config.model_copy(
        update={"context_strategy": "sections", "contexts_file": config.contexts_file if explicit else None}
    )
    rows = [json.loads(line) for line in config.sources_file.read_text().splitlines()]
    for i, row in enumerate(rows[:12]):
        row["page_number"] = i + 1
    rows[0]["text"] = "Contents\nAlpha ... 1\nBeta ... 4\nGamma ... 7"
    rows[2]["text"], rows[5]["text"] = "# Alpha\nPressure 20 bar.", "# Beta\nService interval 6 months."
    config.sources_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inference = SectionInference()
    monkeypatch.setattr(
        "data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", lambda *args: inference
    )
    original = inference.generate
    # Use the existing public workflow with a synthetic model response only for the new schema.
    from unittest.mock import patch

    responses = TOCWorkflowResponses(original)
    with patch.object(inference, "generate", responses.generate):
        handoff = run_multimodal_sdg(config)
    assert handoff.is_file()
    assert responses.confirmations == (0 if explicit else 1)
    path = config.output_dir / "planning/section_boundaries.json"
    assert path.exists() != explicit
    if not explicit:
        audit = json.loads(path.read_text())
        assert audit["coverage"]["covered_units"] == len(rows)
        assert audit["documents"][0]["offset"] == 2


class TOCWorkflowResponses:
    """Delegate all existing response fixtures, recording automatic TOC requests."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.confirmations = 0

    def generate(self, requests):
        if requests and requests[0].schema is TOCConfirmation:
            self.confirmations += len(requests)
            return [TOCConfirmation(explanation="Fixture TOC", has_table_of_contents=True) for _ in requests]
        return self.delegate(requests)
