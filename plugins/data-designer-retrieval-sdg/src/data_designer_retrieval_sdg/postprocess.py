# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-processing utilities for generated retriever SDG data.

Includes BEIR-format export, quality-based filtering, and a helper
for loading positive documents with modality metadata.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# BEIR-format post-processing
# ---------------------------------------------------------------------------


def postprocess_retriever_data(
    generated_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Flatten generated data into BEIR-style queries, qrels, and splits.

    Args:
        generated_df: DataFrame produced by the pipeline, containing
            ``file_name``, ``deduplicated_qa_pairs`` (or ``qa_generation``),
            and metadata columns.

    Returns:
        Tuple of ``(queries_df, qrels_df, splits)`` where *splits* maps
        modality names to lists of query IDs.
    """
    print(f"Processing {len(generated_df)} generated records...")

    queries_data: list[dict] = []
    qrels_data: list[dict] = []
    splits: dict[str, list[str]] = defaultdict(list)
    query_counter = 0
    reasoning_types: list[str] = []
    query_types: list[str] = []

    for _, row in generated_df.iterrows():
        if "file_name" not in row:
            print("Warning: Skipping row without file_name")
            continue

        file_name = row["file_name"]
        qa_pairs = _extract_qa_pairs(row, file_name)
        if qa_pairs is None:
            continue

        for qa_pair in qa_pairs:
            parsed = _parse_qa_pair(qa_pair)
            if not parsed["question"] or not isinstance(parsed["question"], str):
                continue

            query_id = f"q{query_counter:08d}"
            query_counter += 1
            reasoning_types.append(parsed["reasoning_type"])
            query_types.append(parsed["query_type"])

            metadata = {
                "query_type": parsed["query_type"],
                "reasoning_type": parsed["reasoning_type"],
                "question_complexity": parsed["question_complexity"],
                "hop_count": parsed["hop_count"],
                "segment_ids": parsed["segment_ids"],
                "source_file": file_name,
                "answer": parsed["answer"],
            }
            if parsed["hop_contexts"]:
                metadata["hop_contexts"] = parsed["hop_contexts"]

            queries_data.append({"_id": query_id, "metadata": metadata, "text": parsed["question"]})
            qrels_data.append({"query-id": query_id, "corpus-id": file_name, "score": 1})
            splits["text"].append(query_id)

    queries_df = pd.DataFrame(queries_data)
    qrels_df = pd.DataFrame(qrels_data)

    total_queries = len(queries_df)
    if total_queries > 0:
        print(f"\nGenerated {total_queries} queries from {len(generated_df)} documents")
        _print_distribution("Reasoning type", reasoning_types, total_queries)
        _print_distribution("Query type", query_types, total_queries)
    else:
        print("\nWarning: No queries generated!")

    return queries_df, qrels_df, dict(splits)


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------


def filter_qa_pairs_by_quality(
    generated_df: pd.DataFrame,
    quality_threshold: float = 7.0,
) -> tuple[pd.DataFrame, list[dict]]:
    """Filter deduplicated QA pairs using evaluation scores.

    Each pair's ``overall.score`` from the ``qa_evaluations`` column is
    compared against *quality_threshold*.  Rows with mismatched
    evaluation/pair counts are skipped.

    Args:
        generated_df: DataFrame with ``deduplicated_qa_pairs``,
            ``qa_evaluations``, and ``file_name`` columns.
        quality_threshold: Minimum overall quality score to retain a pair.

    Returns:
        Tuple of ``(filtered_df, skipped_files)`` where *skipped_files* is
        a list of ``{"file_name": ..., "reason": ...}`` dicts.
    """
    print(f"Filtering QA pairs based on quality threshold: {quality_threshold}")

    total_pairs = 0
    filtered_pairs = 0
    all_filtered: list[dict] = []
    skipped_files: list[dict] = []

    for _, row in generated_df.iterrows():
        file_name = row.get("file_name", "unknown")
        dedup_pairs = _to_list(row.get("deduplicated_qa_pairs"))
        if dedup_pairs is None:
            print(f"Warning: Skipping {file_name} - deduplicated_qa_pairs is None")
            continue
        if not dedup_pairs:
            print(f"Warning: Skipping {file_name} - no valid deduplicated pairs found")
            continue

        scores = _extract_evaluation_scores(row.get("qa_evaluations"))

        if len(scores) != len(dedup_pairs):
            reason = f"deduplicated_qa_pairs has {len(dedup_pairs)} items but qa_evaluations has {len(scores)} items"
            print(f"Warning: Skipping {file_name} - data integrity error: {reason}")
            skipped_files.append({"file_name": file_name, "reason": reason})
            continue

        for pair_idx, qa_pair in enumerate(dedup_pairs):
            total_pairs += 1
            quality_score = scores[pair_idx] if pair_idx < len(scores) else 0
            if quality_score >= quality_threshold:
                pair_dict = _qa_pair_to_dict(qa_pair)
                pair_dict["file_name"] = file_name
                pair_dict["quality_score"] = quality_score
                all_filtered.append(pair_dict)
            else:
                filtered_pairs += 1

    filtered_df = pd.DataFrame(all_filtered)

    print("\nQuality Filtering Results:")
    print(f"  Total QA pairs: {total_pairs}")
    print(f"  Filtered out (score < {quality_threshold}): {filtered_pairs}")
    print(f"  Remaining high-quality pairs: {len(filtered_df)}")
    print(f"  Files skipped due to data issues: {len(skipped_files)}")
    retention = len(filtered_df) / total_pairs * 100 if total_pairs > 0 else 0
    print(f"  Retention rate: {retention:.1f}%")

    return filtered_df, skipped_files


# ---------------------------------------------------------------------------
# Modality / BEIR loader
# ---------------------------------------------------------------------------


def load_positive_docs_with_modality(
    test_tsv_path: Path,
    corpus_jsonl_path: Path,
    split_json_path: Path,
    min_text_length: int = 0,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load positive documents and map them to their modalities.

    Args:
        test_tsv_path: Path to ``qrels/test.tsv``.
        corpus_jsonl_path: Path to ``corpus.jsonl``.
        split_json_path: Path to ``split.json``.
        min_text_length: Minimum text length to include a document.

    Returns:
        Tuple of ``(positive_docs_df, doc_to_modality_final)``.
    """
    qrels_df = pd.read_csv(test_tsv_path, sep="\t")

    with open(split_json_path, encoding="utf-8") as f:
        splits = json.load(f)

    query_to_modality: dict[str, str] = {}
    for modality, query_ids in splits.items():
        for query_id in query_ids:
            query_to_modality[query_id] = modality

    doc_to_modality: dict[str, set[str]] = defaultdict(set)
    for _, row in qrels_df.iterrows():
        query_id = row["query-id"]
        corpus_id = row["corpus-id "]  # trailing space in column name
        if query_id in query_to_modality:
            doc_to_modality[corpus_id].add(query_to_modality[query_id])

    doc_to_modality_final: dict[str, str] = {}
    for doc_id, modalities in doc_to_modality.items():
        if len(modalities) == 1:
            doc_to_modality_final[doc_id] = next(iter(modalities))
        else:
            modality_counts: dict[str, int] = defaultdict(int)
            for _, r in qrels_df[qrels_df["corpus-id "] == doc_id].iterrows():
                qid = r["query-id"]
                if qid in query_to_modality:
                    modality_counts[query_to_modality[qid]] += 1
            doc_to_modality_final[doc_id] = max(modality_counts, key=modality_counts.get)  # type: ignore[arg-type]

    unique_group_ids = set(doc_to_modality_final.keys())
    corpus_docs_by_group: dict[str, dict] = {}
    with open(corpus_jsonl_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            group_id = doc.get("group_id", doc["_id"])
            if group_id in unique_group_ids and group_id not in corpus_docs_by_group:
                corpus_docs_by_group[group_id] = doc

    positive_docs_data: list[dict] = []
    for group_id, modality in doc_to_modality_final.items():
        if group_id in corpus_docs_by_group:
            doc = corpus_docs_by_group[group_id]
            positive_docs_data.append(
                {
                    "doc_id": doc["_id"],
                    "text": doc["text"],
                    "title": doc.get("title", ""),
                    "modality": modality,
                    "group_id": group_id,
                }
            )

    positive_docs_df = pd.DataFrame(positive_docs_data)

    if min_text_length > 0 and len(positive_docs_df) > 0:
        original_count = len(positive_docs_df)
        positive_docs_df = positive_docs_df[positive_docs_df["text"].str.len() >= min_text_length]
        filtered_count = original_count - len(positive_docs_df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} documents shorter than {min_text_length} characters")

    return positive_docs_df, doc_to_modality_final


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_list(value: object) -> list | None:
    """Coerce *value* to a Python list, handling numpy arrays."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return None


def _extract_qa_pairs(row: pd.Series, file_name: object) -> list | None:
    """Pull the QA pairs list from a generated row."""
    if "deduplicated_qa_pairs" in row and row["deduplicated_qa_pairs"] is not None:
        pairs = row["deduplicated_qa_pairs"]
    elif "qa_generation" in row:
        qa_gen = row.get("qa_generation")
        if qa_gen is None:
            print(f"Warning: Skipping {file_name} - qa_generation is None")
            return None
        if isinstance(qa_gen, dict):
            pairs = qa_gen.get("pairs", [])
        else:
            pairs = getattr(qa_gen, "pairs", [])
    else:
        print(f"Warning: Skipping {file_name} - no qa_generation or deduplicated_qa_pairs found")
        return None

    pairs = _to_list(pairs) if not isinstance(pairs, list) else pairs
    if not pairs:
        print(f"Warning: Skipping {file_name} - no valid pairs found")
        return None
    return pairs


def _parse_qa_pair(qa_pair: object) -> dict:
    """Normalise a QA pair (dict or Pydantic model) to a plain dict."""
    fields = (
        "question",
        "answer",
        "query_type",
        "reasoning_type",
        "question_complexity",
        "segment_ids",
        "hop_count",
        "hop_contexts",
    )
    defaults = ("", "", "", "", 0, [], 1, [])

    result: dict = {}
    for field, default in zip(fields, defaults):
        if isinstance(qa_pair, dict):
            val = qa_pair.get(field, default)
        else:
            val = getattr(qa_pair, field, default)
        if isinstance(val, np.ndarray):
            val = val.tolist()
        result[field] = val
    return result


def _qa_pair_to_dict(qa_pair: object) -> dict:
    """Convert a QA pair to a plain dict for DataFrame construction."""
    keys = (
        "question",
        "answer",
        "query_type",
        "reasoning_type",
        "question_complexity",
        "segment_ids",
        "hop_count",
        "hop_contexts",
    )
    if isinstance(qa_pair, dict):
        return {k: qa_pair.get(k, None) for k in keys}
    return {k: getattr(qa_pair, k, None) for k in keys}


def _extract_evaluation_scores(qa_evaluations: object) -> list[float]:
    """Pull overall scores from the qa_evaluations object."""
    scores: list[float] = []
    if qa_evaluations is None:
        return scores

    if isinstance(qa_evaluations, dict):
        evaluations_list = qa_evaluations.get("evaluations", [])
    else:
        evaluations_list = getattr(qa_evaluations, "evaluations", [])

    if isinstance(evaluations_list, np.ndarray):
        evaluations_list = evaluations_list.tolist()

    for eval_item in evaluations_list:
        if isinstance(eval_item, dict):
            overall = eval_item.get("overall", {})
        else:
            overall = getattr(eval_item, "overall", None)

        if isinstance(overall, dict):
            scores.append(overall.get("score", 0))
        elif overall is not None:
            scores.append(getattr(overall, "score", 0))
        else:
            scores.append(0)

    return scores


def _print_distribution(label: str, values: list[str], total: int) -> None:
    """Print a frequency distribution to stdout."""
    print(f"\n{label} distribution:")
    dist = pd.Series(values).value_counts()
    for name, count in dist.items():
        pct = count / total * 100
        print(f"  {name}: {count} queries ({pct:.1f}%)")
