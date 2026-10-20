# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import data_designer.config as dd
import pandas as pd
from data_designer.interface import DataDesigner

from data_designer_gym.config import GymHarnessColumnConfig


def main():
    endpoint, directory = sys.argv[1:]
    root = Path(directory)
    rows = pd.DataFrame([{"row_id": "run-alpha", "prompt": "Create solution.py with TOKEN = 'alpha' and VALUE = 1"}])
    for phase in (1, 2):
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
                request_columns={"input": "prompt" if phase == 1 else "user_reaction", "user": "row_id"},
                request_values={"metadata": {"turn": str(phase)}},
                timeout=0.15 if phase == 1 else 15,
                max_parallel_requests=1,
                skip=dd.SkipConfig(when="{{ reply_1_evidence.status != 'completed' }}") if phase == 2 else None,
            )
        )
        if phase == 1:
            builder.add_column(
                dd.ExpressionColumnConfig(
                    name="user_reaction",
                    expr="Your reply was: {{ reply_1 }}. For the second turn set VALUE = 2 and preserve TOKEN.",
                    skip=dd.SkipConfig(when="{{ reply_1_evidence.status != 'completed' }}"),
                )
            )
        result = DataDesigner(artifact_path=root / f"dd-timeout-{phase}").create(builder, num_records=1)
        rows = result.load_dataset()
        row = rows.iloc[0]
        assert row["reply_1_evidence"]["failure_kind"] == "timeout"
        assert pd.isna(row["reply_1"])
        assert pd.isna(row["user_reaction"])
        if phase == 2:
            assert pd.isna(row["reply_2"])
            assert pd.isna(row["reply_2_evidence"])
        rows.to_json(root / f"timeout-{phase}.jsonl", orient="records", lines=True)
        rows = pd.read_json(root / f"timeout-{phase}.jsonl", lines=True)


if __name__ == "__main__":
    main()
