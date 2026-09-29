# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional cross-repository contract test against the Nemotron recipe consumer."""

from pathlib import Path

import pytest
from test_retrieval_export import _candidate, _record, _unit, _write_rows

from data_designer_retrieval_sdg.retrieval import SplitRatios, export_retrieval_data


def test_public_export_passes_full_recipe_validation(tmp_path: Path) -> None:
    """Exercise real producer and consumer together when the recipe is installed."""
    retrieval_vl = pytest.importorskip("nemotron.recipes.retrieval_vl")
    handoff = pytest.importorskip("nemotron.recipes.embed.sdg_manifest")
    records = [
        _record(
            f"source-{index}",
            [_unit(f"page-{index}", "shared-document", text=f"Service policy number {index}.")],
            _candidate(
                f"What does service policy {index} require?", "The stated service terms.", "question", [f"page-{index}"]
            ),
            independent_answer="The applicable terms are in the service policy.",
        )
        for index in range(4)
    ]
    source = tmp_path / "generated.jsonl"
    _write_rows(source, records)
    summary = export_retrieval_data(
        source,
        tmp_path / "bundle",
        dataset_id="service-policies",
        ratios=SplitRatios(train=0.5, validation=0, evaluation=0.5),
    )
    paths = retrieval_vl.inspect_vl_bundle(Path(summary.run_manifest_path), view="text")
    assert paths.train_corpus == paths.evaluation_corpus
    manifest = handoff.write_generation_manifest(
        output_dir=tmp_path,
        output_path=paths.train,
        dataset_name="service-policies",
        portable_bundle=tmp_path / "bundle",
    )
    assert handoff.resolve_portable_training_input(manifest, "text", "grouped_query_disjoint") == paths.train
    assert handoff.resolve_portable_evaluation_input(manifest, "text", "grouped_query_disjoint") == (
        paths.synthetic_eval,
        tmp_path / "bundle",
    )
