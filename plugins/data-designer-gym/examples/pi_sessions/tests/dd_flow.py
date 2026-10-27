# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import data_designer.config as dd
import pandas as pd
from data_designer.interface import DataDesigner

from data_designer_gym.config import GymHarnessColumnConfig


def main():
    endpoint, phase, directory = sys.argv[1:]
    root = Path(directory)
    rows = (
        pd.DataFrame(
            [
                {"row_id": f"run-{token}", "prompt": f"Create solution.py with TOKEN = '{token}' and VALUE = 1"}
                for token in ("alpha", "beta")
            ]
        )
        if phase == "1"
        else pd.read_json(root / "phase-1.jsonl", lines=True)
    )
    builder = dd.DataDesignerConfigBuilder().with_seed_dataset(
        dd.DataFrameSeedSource(df=rows), sampling_strategy="ordered"
    )
    for model in tuple(builder.model_configs):
        builder.delete_model_config(model.alias)
    builder.add_column(
        GymHarnessColumnConfig(
            name=f"reply_{phase}",
            evidence_column=f"reply_{phase}_evidence",
            endpoint=endpoint,
            row_id_column="row_id",
            request_columns={"input": "prompt" if phase == "1" else "user_reaction", "user": "row_id"},
            request_values={"metadata": {"turn": phase}},
            timeout=60,
            max_parallel_requests=2,
            skip=dd.SkipConfig(when="{{ reply_1_evidence.status != 'completed' }}") if phase == "2" else None,
        )
    )
    if phase == "1":
        builder.add_column(
            dd.ExpressionColumnConfig(
                name="user_reaction",
                expr="Your reply was: {{ reply_1 }}. For the second turn set VALUE = 2 and preserve TOKEN.",
                skip=dd.SkipConfig(when="{{ reply_1_evidence.status != 'completed' }}"),
            )
        )
    result = DataDesigner(artifact_path=str(root / f"dd-artifacts-{phase}")).create(builder, num_records=2)
    data = result.load_dataset()
    assert data[f"reply_{phase}_evidence"].map(lambda item: item["status"] == "completed").all()
    assert data[f"reply_{phase}"].notna().all()
    data.to_json(root / f"phase-{phase}.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
