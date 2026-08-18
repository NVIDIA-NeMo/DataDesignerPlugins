# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd
import pytest

import data_designer_retrieval_sdg.convert as conversion_module
from data_designer_retrieval_sdg.convert import (
    UnionFind,
    build_corpus_and_mappings,
    build_file_to_group_mapping,
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
    run_conversion,
    run_conversion_with_config,
)
from data_designer_retrieval_sdg.run_config import ConversionRunConfig

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
    assert get_file_identifier(["path/to/doc.txt"]) == "path/to/doc.txt"
    assert get_file_identifier([r"path\to\doc.txt"]) == "path/to/doc.txt"


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
    assert ("a.txt", 1) in mapping
    assert mapping[("a.txt", 1)] == "hello"


def test_source_id_prevents_same_basename_collision() -> None:
    df = pd.DataFrame(
        [
            {
                "file_name": ["/mnt/corpus/finance/report.txt"],
                "source_id": "finance/report.txt",
                "chunks": [{"chunk_id": 1, "text": "finance"}],
            },
            {
                "file_name": ["/mnt/corpus/hr/report.txt"],
                "source_id": "hr/report.txt",
                "chunks": [{"chunk_id": 1, "text": "hr"}],
            },
        ]
    )

    _, mapping = build_corpus_and_mappings(df)

    assert mapping[("finance/report.txt", 1)] == "finance"
    assert mapping[("hr/report.txt", 1)] == "hr"


# ---------------------------------------------------------------------------
# create_train_val_test_split
# ---------------------------------------------------------------------------


def test_split_basic() -> None:
    rows = [{"file_name": [f"f{i}.txt"], "question": f"Q{i}"} for i in range(10)]
    df = pd.DataFrame(rows)
    train, val, test = create_train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, seed=42)
    assert len(train) + len(val) + len(test) == 10


def test_split_is_stable_across_input_order() -> None:
    rows = [{"file_name": [f"f{i}.txt"], "question": f"Q{i}"} for i in range(20)]
    forward = pd.DataFrame(rows)
    reverse = pd.DataFrame(reversed(rows))

    forward_splits = create_train_val_test_split(forward, train_ratio=0.6, val_ratio=0.2, seed=42)
    reverse_splits = create_train_val_test_split(reverse, train_ratio=0.6, val_ratio=0.2, seed=42)

    for forward_split, reverse_split in zip(forward_splits, reverse_splits, strict=True):
        assert set(forward_split["question"]) == set(reverse_split["question"])


def test_group_matching_rejects_ambiguous_basename() -> None:
    groups = {
        "finance": ["finance/annual/report.pdf"],
        "hr": ["hr/annual/report.pdf"],
    }

    with pytest.raises(ValueError, match="Ambiguous basename match"):
        build_file_to_group_mapping(groups, {"legacy/report.txt"})


def test_group_matching_allows_unique_extension_fallback() -> None:
    groups = {"finance": ["finance/annual/report.pdf"]}

    mapping = build_file_to_group_mapping(groups, {"finance/annual/report.txt"})

    assert mapping == {"finance/annual/report.txt": "finance"}


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


def test_load_rejects_mixed_format_directory(tmp_path: Path) -> None:
    record = {"file_name": "doc.txt", "deduplicated_qa_pairs": [], "qa_evaluations": {"evaluations": []}}
    (tmp_path / "retrieval.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "generated_batch0.json").write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(ValueError, match="Mixed generated-data formats"):
        load_generated_json_files(str(tmp_path))


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


def test_load_from_parquet_normalizes_nested_arrays_for_chunk_mapping(tmp_path: Path) -> None:
    path = tmp_path / "generated.parquet"
    pd.DataFrame(
        [
            {
                "file_name": ["doc.txt"],
                "chunks": [{"chunk_id": 1, "text": "hello"}, {"chunk_id": 2, "text": "world"}],
                "deduplicated_qa_pairs": [],
                "qa_evaluations": {"evaluations": []},
            }
        ]
    ).to_parquet(path, index=False)

    df = load_generated_json_files(str(path))
    corpus, mapping = build_corpus_and_mappings(df)

    assert isinstance(df.iloc[0]["chunks"], list)
    assert len(corpus) == 2
    assert mapping[("doc.txt", 1)] == "hello"
    assert mapping[("doc.txt", 2)] == "world"


# ---------------------------------------------------------------------------
# generate_training_set / generate_eval_set
# ---------------------------------------------------------------------------


def test_generate_training_set(tmp_path: Path) -> None:
    corpus = {"hello": "d_abc"}
    chunk_mapping = {("doc.txt", 1): "hello"}
    df = pd.DataFrame([{"file_name": ["doc.txt"], "question": "Q?", "segment_ids": [1]}])
    count = generate_training_set(corpus, chunk_mapping, df, str(tmp_path), "my_corpus")
    train_path = tmp_path / "train.json"
    assert train_path.exists()
    payload = json.loads(train_path.read_text())
    assert len(payload["data"]) == 1
    assert count == 1


def test_generate_eval_set(tmp_path: Path) -> None:
    corpus = {"hello": "d_abc"}
    chunk_mapping = {("doc.txt", 1): "hello"}
    df = pd.DataFrame([{"file_name": ["doc.txt"], "question": "Q?", "segment_ids": [1]}])
    count = generate_eval_set(corpus, chunk_mapping, df, str(tmp_path), eval_only=True)
    assert (tmp_path / "corpus.jsonl").exists()
    assert (tmp_path / "queries.jsonl").exists()
    assert (tmp_path / "qrels" / "test.tsv").exists()
    assert count == 1


@pytest.mark.parametrize("use_group_id_in_eval", [False, True])
def test_generate_eval_set_qrels_reference_corpus_ids(
    tmp_path: Path,
    use_group_id_in_eval: bool,
) -> None:
    corpus = {
        "first document": get_corpus_id("first document"),
        "second document": get_corpus_id("second document"),
    }
    chunk_mapping = {
        ("src.txt", 1): "first document",
        ("src.txt", 2): "second document",
    }
    eval_df = pd.DataFrame([{"file_name": ["src.txt"], "question": "Q?", "segment_ids": [1, 2]}])

    generate_eval_set(
        corpus,
        chunk_mapping,
        eval_df,
        str(tmp_path),
        eval_only=True,
        use_group_id_in_eval=use_group_id_in_eval,
    )

    corpus_entries = [json.loads(line) for line in (tmp_path / "corpus.jsonl").read_text().splitlines()]
    corpus_ids = {entry["_id"] for entry in corpus_entries}
    qrels_lines = (tmp_path / "qrels" / "test.tsv").read_text().splitlines()[1:]
    qrels_ids = {line.split("\t")[1] for line in qrels_lines}
    expected_ids = set(corpus.values()) if use_group_id_in_eval else {"d0", "d1"}

    assert corpus_ids == expected_ids
    assert qrels_ids == expected_ids
    if use_group_id_in_eval:
        assert {entry["group_id"] for entry in corpus_entries} == expected_ids
    else:
        assert all("group_id" not in entry for entry in corpus_entries)


def test_run_conversion_returns_generated_paths_and_counts(tmp_path: Path) -> None:
    input_path = tmp_path / "generated.jsonl"
    record = {
        "file_name": ["nested/doc.txt"],
        "source_id": "nested/doc.txt",
        "chunks": [{"chunk_id": 1, "text": "hello"}],
        "deduplicated_qa_pairs": [{"question": "Q?", "segment_ids": [1]}],
        "qa_evaluations": {"evaluations": [{"overall": {"score": 9.0}}]},
    }
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = run_conversion(
        input_path=str(input_path),
        corpus_id="my_corpus",
        output_dir=str(tmp_path / "converted"),
        eval_only=True,
    )

    assert result.output_dir == tmp_path / "converted"
    assert result.train_file is None
    assert result.validation_file is None
    assert result.corpus_dir is None
    assert result.evaluation_dir == tmp_path / "converted"
    assert result.training_examples == 0
    assert result.validation_examples == 0
    assert result.evaluation_queries == 1


def test_run_conversion_with_config_writes_run_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "generated.jsonl"
    record = {
        "file_name": ["nested/doc.txt"],
        "source_id": "nested/doc.txt",
        "chunks": [{"chunk_id": 1, "text": "hello"}],
        "deduplicated_qa_pairs": [{"question": "Q?", "segment_ids": [1]}],
        "qa_evaluations": {"evaluations": [{"overall": {"score": 9.0}}]},
    }
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = run_conversion_with_config(
        ConversionRunConfig(
            input_path=input_path,
            corpus_id="my_corpus",
            output_dir=tmp_path / "converted",
            eval_only=True,
        ),
        override_paths=["eval_only"],
    )

    assert result.resolved_config_path == tmp_path / "converted" / ".retrieval_sdg_run" / "resolved_config.yaml"
    assert result.provenance_path == tmp_path / "converted" / ".retrieval_sdg_run" / "config_provenance.json"
    assert result.resolved_config_path.exists()
    assert result.provenance_path.exists()


def test_run_conversion_with_config_does_not_write_metadata_when_conversion_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "generated.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "converted"

    def fail_conversion(**_: object) -> conversion_module.ConversionResult:
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(conversion_module, "run_conversion", fail_conversion)
    with pytest.raises(RuntimeError, match="conversion failed"):
        conversion_module.run_conversion_with_config(
            ConversionRunConfig(
                input_path=input_path,
                corpus_id="my_corpus",
                output_dir=output_dir,
            )
        )

    assert not (output_dir / ".retrieval_sdg_run").exists()
