# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from data_designer_retrieval_sdg.postprocess import filter_qa_pairs_by_quality, postprocess_retriever_data

# ---------------------------------------------------------------------------
# postprocess_retriever_data
# ---------------------------------------------------------------------------


def test_postprocess_basic() -> None:
    df = pd.DataFrame(
        [
            {
                "file_name": ["doc.txt"],
                "deduplicated_qa_pairs": [
                    {
                        "question": "What is X?",
                        "answer": "X is Y.",
                        "query_type": "structural",
                        "reasoning_type": "factual",
                        "question_complexity": 4,
                        "segment_ids": [1],
                        "hop_count": 1,
                        "hop_contexts": [],
                    }
                ],
            }
        ]
    )
    queries_df, qrels_df, splits = postprocess_retriever_data(df)
    assert len(queries_df) == 1
    assert queries_df.iloc[0]["text"] == "What is X?"
    assert len(qrels_df) == 1
    assert "text" in splits


def test_postprocess_skips_missing() -> None:
    df = pd.DataFrame([{"file_name": ["x.txt"]}])
    queries_df, _, _ = postprocess_retriever_data(df)
    assert len(queries_df) == 0


# ---------------------------------------------------------------------------
# filter_qa_pairs_by_quality
# ---------------------------------------------------------------------------


def test_filter_by_quality() -> None:
    df = pd.DataFrame(
        [
            {
                "file_name": ["a.txt"],
                "deduplicated_qa_pairs": [
                    {"question": "Q1", "answer": "A1"},
                    {"question": "Q2", "answer": "A2"},
                ],
                "qa_evaluations": {
                    "evaluations": [
                        {"overall": {"score": 9.0}},
                        {"overall": {"score": 3.0}},
                    ]
                },
            }
        ]
    )
    filtered_df, skipped = filter_qa_pairs_by_quality(df, quality_threshold=7.0)
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["question"] == "Q1"
    assert skipped == []


def test_filter_skips_mismatched() -> None:
    df = pd.DataFrame(
        [
            {
                "file_name": ["bad.txt"],
                "deduplicated_qa_pairs": [{"question": "Q1", "answer": "A1"}],
                "qa_evaluations": {"evaluations": []},
            }
        ]
    )
    filtered_df, skipped = filter_qa_pairs_by_quality(df, quality_threshold=5.0)
    assert len(filtered_df) == 0
    assert len(skipped) == 1
