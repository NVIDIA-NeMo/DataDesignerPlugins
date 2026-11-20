# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical source labels and image attachment alignment in localization requests."""

import json
from pathlib import Path

import pytest

from data_designer_retrieval_sdg.multimodal import models, prompts
from data_designer_retrieval_sdg.multimodal.inference import source_request
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource


@pytest.fixture
def mixed_sources():
    """Supply interleaved documents, repeated page numbers, and shared image bytes."""
    return {
        "intro": RetrievalSource(unit_id="intro", document_id="manual", text="Introduction", page_number=42),
        "bulletin": RetrievalSource(
            unit_id="bulletin", document_id="bulletin", text="Bulletin detail", page_number=7, images=["shared.png"]
        ),
        "diagram": RetrievalSource(
            unit_id="diagram", document_id="manual", text="Diagram caption", page_number=7, images=["diagram.png"]
        ),
        "unpaged": RetrievalSource(unit_id="unpaged", document_id="manual", images=["shared.png"]),
        "appendix": RetrievalSource(unit_id="appendix", document_id="bulletin", text="Appendix"),
    }


def test_localization_uses_canonical_pages_and_actual_attachment_positions(mixed_sources):
    context = models.GenerationContext(
        context_id="mixed", unit_ids=["intro", "diagram", "appendix", "bulletin", "unpaged"]
    )
    request = source_request(
        prompts.LOCALIZE, models.Localization, "judge", context, mixed_sources, judge_stage="localization", query="Why?"
    )
    payload = json.loads(request.text.removeprefix(prompts.LOCALIZE + "\n"))
    assert payload == {
        "sources": [
            {
                "unit_id": "intro",
                "text": "Introduction",
                "document_id": "manual",
                "page_number": 42,
                "image_number": None,
            },
            {
                "unit_id": "diagram",
                "text": "Diagram caption",
                "document_id": "manual",
                "page_number": 7,
                "image_number": 1,
            },
            {
                "unit_id": "appendix",
                "text": "Appendix",
                "document_id": "bulletin",
                "page_number": None,
                "image_number": None,
            },
            {
                "unit_id": "bulletin",
                "text": "Bulletin detail",
                "document_id": "bulletin",
                "page_number": 7,
                "image_number": 2,
            },
            {"unit_id": "unpaged", "text": "", "document_id": "manual", "page_number": None, "image_number": 3},
        ],
        "image_unit_ids": ["diagram", "bulletin", "unpaged"],
        "query": "Why?",
    }
    assert request.images == (Path("diagram.png"), Path("shared.png"), Path("shared.png"))
    assert request.image_unit_ids == ("diagram", "bulletin", "unpaged")
    assert request.schema is models.Localization
    assert request.context == context
    assert request.query == "Why?"


@pytest.mark.parametrize(
    "stage,schema,role", [(None, models.QueryBatch, "generator"), ("relevance", models.RelevanceJudgment, "judge")]
)
def test_other_source_requests_keep_the_original_payload(mixed_sources, stage, schema, role):
    context = models.GenerationContext(context_id="single", unit_ids=["diagram"])
    request = source_request("Task", schema, role, context, mixed_sources, judge_stage=stage, query="Why?")
    assert request.text == (
        'Task\n{"sources": [{"unit_id": "diagram", "text": "Diagram caption"}], '
        '"image_unit_ids": ["diagram"], "query": "Why?"}'
    )
    assert request.images == (Path("diagram.png"),)
    assert request.image_unit_ids == ("diagram",)
