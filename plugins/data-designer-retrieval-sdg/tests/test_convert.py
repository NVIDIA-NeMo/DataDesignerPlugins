# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd

from data_designer_retrieval_sdg.convert import (
    UnionFind,
    build_corpus_and_mappings,
    create_train_val_test_split,
    extract_base_filename,
    file_tuple_in_set,
    filter_mismatched_records,
    generate_eval_set,
    generate_training_set,
    get_corpus_id,
    get_file_identifier,
    load_generated_json_files,
    merge_groups_union_find,
    normalize_file_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_get_corpus_id_deterministic() -> None:
    assert get_corpus_id("hello") == get_corpus_id("hello")
    assert get_corpus_id("hello") != get_corpus_id("world")
    assert get_corpus_id("hello").startswith("d_")


def test_extract_base_filename() -> None:
    assert extract_base_filename("path/to/file.txt") == "file"
    assert extract_base_filename("README") == "README"


def test_normalize_file_name() -> None:
    assert normalize_file_name("file.txt") == ["file.txt"]
    assert normalize_file_name(["a.txt", "b.txt"]) == ["a.txt", "b.txt"]
    assert normalize_file_name(42) == ["42"]


def test_get_file_identifier_single() -> None:
    assert get_file_identifier(["path/to/doc.txt"]) == "doc"


def test_get_file_identifier_multi() -> None:
    ident = get_file_identifier(["a.txt", "b.txt"])
    assert len(ident) == 16  # MD5 truncated


def test_file_tuple_in_set() -> None:
    s = {("a.txt",), ("b.txt", "c.txt")}
    assert file_tuple_in_set(["a.txt"], s) is True
    assert file_tuple_in_set(["b.txt", "c.txt"], s) is True
    assert file_tuple_in_set(["d.txt"], s) is False


# ---------------------------------------------------------------------------
# filter_mismatched_records
# ---------------------------------------------------------------------------


def test_filter_mismatched_records() -> None:
    records = [
        {"file_name": "ok", "deduplicated_qa_pairs": [1], "qa_evaluations": {"evaluations": [1]}},
        {"file_name": "bad", "deduplicated_qa_pairs": [1, 2], "qa_evaluations": {"evaluations": [1]}},
    ]
    filtered, dropped = filter_mismatched_records(records)
    assert len(filtered) == 1
    assert dropped == 1


# ---------------------------------------------------------------------------
# build_corpus_and_mappings
# ---------------------------------------------------------------------------


def test_build_corpus_and_mappings() -> None:
    df = pd.DataFrame(
        [
            {
                "file_name": ["a.txt"],
                "chunks": [{"chunk_id": 1, "text": "hello"}, {"chunk_id": 2, "text": "world"}],
            }
        ]
    )
    corpus, mapping = build_corpus_and_mappings(df)
    assert len(corpus) == 2
    assert ("a", 1) in mapping
    assert mapping[("a", 1)] == "hello"


# ---------------------------------------------------------------------------
# create_train_val_test_split
# ---------------------------------------------------------------------------


def test_split_basic() -> None:
    rows = [{"file_name": [f"f{i}.txt"], "question": f"Q{i}"} for i in range(10)]
    df = pd.DataFrame(rows)
    train, val, test = create_train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, seed=42)
    assert len(train) + len(val) + len(test) == 10


# ---------------------------------------------------------------------------
# UnionFind
# ---------------------------------------------------------------------------


def test_union_find() -> None:
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("d") != uf.find("a")


def test_merge_groups_union_find() -> None:
    groups = {"g1": ["a", "b"], "g2": ["b", "c"]}
    merged = merge_groups_union_find(groups)
    assert len(merged) == 1
    members = list(merged.values())[0]
    assert set(members) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# load_generated_json_files
# ---------------------------------------------------------------------------


def test_load_from_single_file(tmp_path: Path) -> None:
    data = [
        {
            "file_name": "doc.txt",
            "deduplicated_qa_pairs": [{"question": "Q"}],
            "qa_evaluations": {"evaluations": [{"overall": {"score": 8}}]},
        }
    ]
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data))
    df = load_generated_json_files(str(p))
    assert len(df) == 1
    assert df.iloc[0]["file_name"] == ["doc.txt"]


def test_load_from_directory(tmp_path: Path) -> None:
    for i in range(2):
        data = [{"file_name": f"d{i}.txt", "deduplicated_qa_pairs": [], "qa_evaluations": {"evaluations": []}}]
        (tmp_path / f"generated_batch{i}.json").write_text(json.dumps(data))
    df = load_generated_json_files(str(tmp_path))
    assert len(df) == 2


def test_load_from_jsonl_file(tmp_path: Path) -> None:
    records = [
        {"file_name": "a.txt", "deduplicated_qa_pairs": [], "qa_evaluations": {"evaluations": []}},
        {"file_name": "b.txt", "deduplicated_qa_pairs": [], "qa_evaluations": {"evaluations": []}},
    ]
    path = tmp_path / "generated.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    df = load_generated_json_files(str(path))

    assert len(df) == 2
    assert df.iloc[0]["file_name"] == ["a.txt"]


def test_load_from_jsonl_directory(tmp_path: Path) -> None:
    for name in ("generated-a.jsonl", "generated-b.jsonl"):
        record = {"file_name": name, "deduplicated_qa_pairs": [], "qa_evaluations": {"evaluations": []}}
        (tmp_path / name).write_text(json.dumps(record) + "\n", encoding="utf-8")

    df = load_generated_json_files(str(tmp_path))

    assert len(df) == 2


def test_load_from_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "generated.parquet"
    pd.DataFrame(
        [
            {
                "file_name": ["doc.txt"],
                "deduplicated_qa_pairs": [],
                "qa_evaluations": {"evaluations": []},
            }
        ]
    ).to_parquet(path, index=False)

    df = load_generated_json_files(str(path))

    assert len(df) == 1
    assert df.iloc[0]["file_name"] == ["doc.txt"]


# ---------------------------------------------------------------------------
# generate_training_set / generate_eval_set
# ---------------------------------------------------------------------------


def test_generate_training_set(tmp_path: Path) -> None:
    corpus = {"hello": "d_abc"}
    chunk_mapping = {("doc", 1): "hello"}
    df = pd.DataFrame([{"file_name": ["doc.txt"], "question": "Q?", "segment_ids": [1]}])
    generate_training_set(corpus, chunk_mapping, df, str(tmp_path), "my_corpus")
    train_path = tmp_path / "train.json"
    assert train_path.exists()
    payload = json.loads(train_path.read_text())
    assert len(payload["data"]) == 1


def test_generate_eval_set(tmp_path: Path) -> None:
    corpus = {"hello": "d_abc"}
    chunk_mapping = {("doc", 1): "hello"}
    df = pd.DataFrame([{"file_name": ["doc.txt"], "question": "Q?", "segment_ids": [1]}])
    generate_eval_set(corpus, chunk_mapping, df, str(tmp_path), eval_only=True)
    assert (tmp_path / "corpus.jsonl").exists()
    assert (tmp_path / "queries.jsonl").exists()
    assert (tmp_path / "qrels" / "test.tsv").exists()
