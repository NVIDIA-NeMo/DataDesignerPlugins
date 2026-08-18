# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd
import pytest

from data_designer_retrieval_sdg.postprocess import (
    filter_qa_pairs_by_quality,
    load_positive_docs_with_modality,
    postprocess_retriever_data,
)

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


# ---------------------------------------------------------------------------
# load_positive_docs_with_modality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus_id_header", ["corpus-id", "corpus-id "])
def test_load_positive_docs_with_modality_accepts_corpus_id_headers(tmp_path: Path, corpus_id_header: str) -> None:
    """Canonical and legacy corpus ID headers produce the same result."""
    qrels_path = tmp_path / "test.tsv"
    qrels_path.write_text(f"query-id\t{corpus_id_header}\tscore\nq1\td1\t1\n", encoding="utf-8")

    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(json.dumps({"_id": "d1", "text": "hello world", "title": "T"}) + "\n", encoding="utf-8")

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"text": ["q1"]}), encoding="utf-8")

    docs_df, modality_map = load_positive_docs_with_modality(qrels_path, corpus_path, split_path)

    assert len(docs_df) == 1
    assert docs_df.iloc[0]["doc_id"] == "d1"
    assert docs_df.iloc[0]["modality"] == "text"
    assert modality_map == {"d1": "text"}


def test_load_positive_docs_with_modality_uses_most_common_modality(tmp_path: Path) -> None:
    """A document shared across modalities uses its most frequent modality."""
    qrels_path = tmp_path / "test.tsv"
    qrels_path.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td1\t1\nq3\td1\t1\n", encoding="utf-8")

    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(json.dumps({"_id": "d1", "text": "hello world", "title": "T"}) + "\n", encoding="utf-8")

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"text": ["q1", "q2"], "image": ["q3"]}), encoding="utf-8")

    docs_df, modality_map = load_positive_docs_with_modality(qrels_path, corpus_path, split_path)

    assert docs_df.iloc[0]["modality"] == "text"
    assert modality_map == {"d1": "text"}
