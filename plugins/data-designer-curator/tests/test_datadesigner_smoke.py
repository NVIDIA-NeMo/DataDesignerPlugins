# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner

from data_designer_curator.config import ExactDedupProcessorConfig


def test_exact_dedup_runs_through_data_designer_preview(tmp_path: Path) -> None:
    builder = dd.DataDesignerConfigBuilder()
    builder.add_column(
        dd.SamplerColumnConfig(
            name="text",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["same"]),
        )
    )
    builder.add_processor(ExactDedupProcessorConfig(name="dedup", text_columns=["text"]))

    result = DataDesigner(artifact_path=tmp_path).preview(builder, num_records=3)

    assert result.dataset is not None
    assert result.dataset["text"].tolist() == ["same"]
    assert result.processor_artifacts is not None
    assert result.processor_artifacts["dedup"][1]["_dd_action"] == "duplicate"
