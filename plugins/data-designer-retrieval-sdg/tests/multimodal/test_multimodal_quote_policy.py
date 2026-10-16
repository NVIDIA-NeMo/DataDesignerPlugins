# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quote diagnostics do not replace quality or source-evidence checks."""

import json

import pytest
from test_multimodal_sdg import generated_fixture

from data_designer_retrieval_sdg.multimodal.export import export_multimodal_bundle
from data_designer_retrieval_sdg.multimodal.models import Localization
from data_designer_retrieval_sdg.multimodal.workflow import load_contexts, localization_reasons, quote_verified


@pytest.mark.parametrize("modality", ["text", "image", "text_and_image"])
@pytest.mark.parametrize("quote", ["", "  \n", "Unrelated statement", "opens at 20 bar"])
@pytest.mark.parametrize("strict", [False, True])
def test_quote_policy_by_modality(tmp_path, modality, quote, strict):
    config, sources, candidates, _, _ = generated_fixture(tmp_path)
    support = (
        candidates[0]
        .localization.supports[0]
        .model_copy(update={"modality": modality, "quote": quote, "visual_evidence": "Visible pressure curve"})
    )
    reasons = localization_reasons(
        Localization(supports=[support]),
        load_contexts(config, sources)[0],
        {s.unit_id: s for s in sources},
        strict,
    )
    expected = []
    if strict and modality != "image":
        if not quote.strip():
            expected = ["missing_text_evidence"]
        elif not quote_verified(quote, sources[0].text):
            expected = ["unverified_text_evidence"]
    assert reasons == expected


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "modality,source_changes,support_changes,duplicate,reason",
    [
        ("text", {"text": " "}, {}, False, "missing_text_evidence"),
        ("text_and_image", {"text": ""}, {}, False, "missing_text_evidence"),
        ("image", {"images": []}, {}, False, "missing_visual_evidence"),
        ("text_and_image", {"images": []}, {}, False, "missing_visual_evidence"),
        ("image", {}, {"visual_evidence": ""}, False, "missing_visual_evidence"),
        ("text_and_image", {}, {"visual_evidence": ""}, False, "missing_visual_evidence"),
        ("text", {}, {"unit_id": "unknown"}, False, "invalid_localized_identity"),
        ("text", {}, {}, True, "invalid_localized_identity"),
    ],
)
def test_optional_quotes_preserve_evidence_and_identity_gates(
    tmp_path, strict, modality, source_changes, support_changes, duplicate, reason
):
    config, sources, candidates, _, _ = generated_fixture(tmp_path)
    support = (
        candidates[0]
        .localization.supports[0]
        .model_copy(update={"modality": modality, "visual_evidence": "Visible pressure curve", **support_changes})
    )
    by_id = {s.unit_id: s.model_copy(update=source_changes) for s in sources}
    localized = Localization(supports=[support, support] if duplicate else [support])
    assert localization_reasons(localized, load_contexts(config, sources)[0], by_id, strict) == [reason]


@pytest.mark.parametrize("modality", ["text", "image", "text_and_image"])
def test_export_preserves_optional_quote_diagnostics_and_strict_gate(tmp_path, modality):
    config, sources, candidates, outcomes, _ = generated_fixture(tmp_path)
    for candidate in candidates:
        for support in candidate.localization.supports:
            support.modality = modality
            support.quote = ""
            support.visual_evidence = "Visible pressure curve" if modality != "text" else ""
            candidate.quote_verification[support.unit_id] = False
    root = tmp_path / "optional"
    export_multimodal_bundle(root, sources, candidates, outcomes, config)
    report = json.loads((root / "report.json").read_text())
    assert report["accepted_query_ids"] == 10
    assert report["positive_annotations"] == 20
    assert report["quote_verification"] == {"false": 20}
    strict_config = config.model_copy(update={"require_verbatim_quotes": True})
    if modality == "image":
        export_multimodal_bundle(tmp_path / "strict", sources, candidates, outcomes, strict_config)
    else:
        with pytest.raises(ValueError, match="Recorded acceptance"):
            export_multimodal_bundle(tmp_path / "strict", sources, candidates, outcomes, strict_config)
