# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint evidence selection and support grades through the public workflow."""

import json

import pytest
from PIL import Image, ImageDraw
from test_multimodal_sdg import fixture_config
from test_multimodal_sections import SectionInference
from workflow_test_utils import install_inference

from data_designer_retrieval_sdg.multimodal import models, run_multimodal_sdg
from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource
from data_designer_retrieval_sdg.multimodal.storage import digest
from data_designer_retrieval_sdg.multimodal.workflow import localization_reasons, normalize_localization


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("valid_grade", [1, 2])
@pytest.mark.parametrize(
    "text,images,valid_modality,invalid_modality,invalid_quote,strict",
    [
        ("Valve V7 opens at 24 bar.", [], "text", "image", "", False),
        ("Valve V7 opens at 24 bar.", [], "text", "text_and_image", "Valve V7", False),
        ("", ["scan.png"], "image", "text", "Valve V7", False),
        ("", ["scan.png"], "image", "text_and_image", "Valve V7", False),
        ("Valve V7 opens at 24 bar.", [], "text", "text", "Unrelated quote", True),
        ("Valve V7 opens at 24 bar.", [], "text", "text", "", True),
    ],
)
def test_invalid_duplicate_cannot_displace_valid_support(
    reverse, valid_grade, text, images, valid_modality, invalid_modality, invalid_quote, strict
):
    source = RetrievalSource(unit_id="valve", document_id="manual", text=text, images=images)
    context = models.GenerationContext(context_id="manual", unit_ids=[source.unit_id])
    valid = models.Support(
        unit_id=source.unit_id,
        grade=valid_grade,
        modality=valid_modality,
        quote="Valve V7" if text else "",
        visual_evidence="Valve V7 opens at 24 bar." if images else "",
        contribution="Pressure specification",
    )
    invalid = valid.model_copy(
        update={"grade": 2, "modality": invalid_modality, "quote": invalid_quote, "visual_evidence": "V7: 24 bar"}
    )
    entries = [valid, invalid] if not reverse else [invalid, valid]
    sources = {source.unit_id: source}
    result = normalize_localization(models.Localization(supports=entries), context, sources, strict)
    assert result.supports == [valid]
    assert not localization_reasons(result, context, sources, strict)
    invalid_only = normalize_localization(
        models.Localization(supports=[invalid.model_copy(update={"grade": 1}), invalid]), context, sources, strict
    )
    assert invalid_only.supports == [invalid]
    assert localization_reasons(invalid_only, context, sources, strict)
    assert localization_reasons(invalid_only, context, sources, strict) == localization_reasons(
        models.Localization(supports=[invalid]), context, {source.unit_id: source}, strict
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_invalid_distinct_unit_remains_subject_to_localization_gate(reverse):
    sources = {
        key: RetrievalSource(unit_id=key, document_id="manual", text="Valve V7 opens at 24 bar.")
        for key in ["valid", "invalid"]
    }
    context = models.GenerationContext(context_id="manual", unit_ids=list(sources))
    valid = models.Support(
        unit_id="valid", grade=1, modality="text", quote="Valve V7", visual_evidence="", contribution="Pressure"
    )
    invalid = valid.model_copy(update={"unit_id": "invalid", "grade": 2, "modality": "image"})
    entries = [invalid, valid] if reverse else [valid, invalid]
    result = normalize_localization(models.Localization(supports=entries), context, sources)
    assert result.supports == entries
    assert localization_reasons(result, context, sources) == ["missing_visual_evidence"]


def test_duplicate_selection_preserves_first_unit_order_and_equal_grade_ties():
    sources = {
        key: RetrievalSource(unit_id=key, document_id="manual", text="Valve V7 opens at 24 bar.")
        for key in ["first", "second"]
    }
    context = models.GenerationContext(context_id="manual", unit_ids=list(sources))
    first = models.Support(
        unit_id="first", grade=1, modality="text", quote="Valve V7", visual_evidence="", contribution="Pressure"
    )
    invalid = first.model_copy(update={"grade": 2, "modality": "image"})
    second = first.model_copy(update={"unit_id": "second"})
    tied = first.model_copy(update={"quote": "24 bar"})
    unknown = first.model_copy(update={"unit_id": "unknown"})
    result = normalize_localization(
        models.Localization(supports=[unknown, invalid, second, first, tied]), context, sources
    )
    assert result.supports == [first, second]


class DistinctInvalidSupportInference(SectionInference):
    """Keep one valid unit and claim a missing image carrier for another unit."""

    def generate(self, requests):
        responses = super().generate(requests)
        for response in responses:
            if isinstance(response, models.Localization):
                response.supports[1] = response.supports[1].model_copy(
                    update={"grade": 2, "modality": "image", "visual_evidence": "V7: 24 bar"}
                )
        return responses


def test_workflow_rejects_invalid_support_for_a_distinct_unit(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    sources = [json.loads(line) for line in config.sources_file.read_text().splitlines()[:2]]
    for source in sources:
        source["images"] = []
    config.sources_file.write_text("".join(json.dumps(source) + "\n" for source in sources))
    config.contexts_file.write_text(json.dumps({"context_id": "text", "unit_ids": ["unit-0", "unit-1"]}) + "\n")
    install_inference(monkeypatch, DistinctInvalidSupportInference())
    with pytest.raises(ValueError, match="No candidates"):
        run_multimodal_sdg(config)
    (candidate,) = json.loads((config.output_dir / "candidates.json").read_text())
    assert not candidate["accepted"]
    assert candidate["rejection_reasons"] == ["missing_visual_evidence"]
    assert [support["unit_id"] for support in candidate["localization"]["supports"]] == ["unit-0", "unit-1"]


class DuplicateSupportInference(SectionInference):
    """Return a missing image carrier alongside a valid text positive."""

    def __init__(self, reverse, valid_grade):
        super().__init__()
        self.reverse = reverse
        self.valid_grade = valid_grade

    def generate(self, requests):
        responses = super().generate(requests)
        for response in responses:
            if isinstance(response, models.Localization):
                valid = response.supports[0].model_copy(update={"grade": self.valid_grade})
                invalid = valid.model_copy(update={"grade": 2, "modality": "image", "visual_evidence": "V7: 24 bar"})
                response.supports = [invalid, valid] if self.reverse else [valid, invalid]
        return responses


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("valid_grade", [1, 2])
def test_workflow_duplicate_support_keeps_valid_positive_and_grade(tmp_path, monkeypatch, reverse, valid_grade):
    config = fixture_config(tmp_path)
    source = json.loads(config.sources_file.read_text().splitlines()[0])
    source["images"] = []
    config.sources_file.write_text(json.dumps(source) + "\n")
    config.contexts_file.write_text(json.dumps({"context_id": "text", "unit_ids": ["unit-0"]}) + "\n")
    completion = install_inference(monkeypatch, DuplicateSupportInference(reverse, valid_grade))
    assert run_multimodal_sdg(config).is_file()
    (candidate,) = json.loads((config.output_dir / "candidates.json").read_text())
    assert candidate["accepted"]
    (support,) = candidate["localization"]["supports"]
    assert (support["unit_id"], support["modality"], support["grade"]) == ("unit-0", "text", valid_grade)
    calls = completion.await_count
    assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
    assert completion.await_count == calls


class JointEvidenceInference(SectionInference):
    """Control judgments to test execution policy, without claiming model accuracy."""

    def __init__(self, modality, relevance=5):
        super().__init__()
        self.modality = modality
        self.relevance = relevance

    def generate(self, requests):
        responses = super().generate(requests)
        for request, response in zip(requests, responses, strict=True):
            if isinstance(response, models.QueryBatch):
                response.queries[0].query = "What is the relief valve pressure?"
                response.queries[0].evidence_modality = "text_and_image"
            elif isinstance(response, models.RelevanceJudgment):
                response.relevance = self.relevance
            elif isinstance(response, models.Localization):
                response.supports = (
                    []
                    if self.modality is None
                    else [
                        models.Support(
                            unit_id="unit-0",
                            grade=2,
                            modality=self.modality,
                            quote="The relief valve is instrument V7." if self.modality != "image" else "",
                            visual_evidence="V7: 24 bar" if self.modality != "text" else "",
                            contribution="The text identifies V7; its pressure is shown in the image.",
                        )
                    ]
                )
        return responses


@pytest.mark.parametrize("modality", ["text", "image", "text_and_image", None])
@pytest.mark.parametrize("relevance", [4, 5])
def test_joint_localization_preserves_evidence_selection_grades_and_resume(tmp_path, monkeypatch, modality, relevance):
    config = fixture_config(tmp_path).model_copy(update={"require_verbatim_quotes": True})
    source = json.loads(config.sources_file.read_text().splitlines()[0])
    source["text"] = "The relief valve is instrument V7."
    source["page_number"] = 17
    config.sources_file.write_text(json.dumps(source) + "\n")
    config.contexts_file.write_text(json.dumps({"context_id": "paired", "unit_ids": ["unit-0"]}) + "\n")
    image = Image.new("RGB", (240, 80), "white")
    ImageDraw.Draw(image).text((10, 20), "V7: 24 bar", fill="black")
    image.save(tmp_path / "page.png")
    inference = JointEvidenceInference(modality, relevance)
    completion = install_inference(monkeypatch, inference)
    if modality is None:
        with pytest.raises(ValueError, match="No candidates"):
            run_multimodal_sdg(config)
    else:
        assert run_multimodal_sdg(config).is_file()
    (candidate,) = json.loads((config.output_dir / "candidates.json").read_text())
    assert candidate["accepted"] == (modality is not None)
    requests = [request for request in inference.requests if request.schema is models.Localization]
    assert len(requests) == 1
    assert len(requests[0].images) == 1
    payload = json.loads(requests[0].text.splitlines()[-1])
    assert payload["sources"] == [
        {
            "unit_id": "unit-0",
            "text": source["text"],
            "document_id": source["document_id"],
            "page_number": 17,
            "image_number": 1,
        }
    ]
    assert payload["image_unit_ids"] == ["unit-0"]
    assert not list(config.output_dir.rglob("stage-*-joint-localization"))
    if modality is None:
        assert candidate["localization"]["supports"] == []
        assert candidate["rejection_reasons"] == ["no_localized_positives"]
    else:
        (support,) = candidate["localization"]["supports"]
        assert (support["unit_id"], support["modality"], support["grade"]) == ("unit-0", modality, 2)
        saved = {path: digest(path) for path in config.output_dir.rglob("*") if path.is_file()}
        calls = completion.await_count
        assert run_multimodal_sdg(config.model_copy(update={"resume": True})).is_file()
        assert completion.await_count == calls
        assert all(digest(path) == checksum for path, checksum in saved.items())
