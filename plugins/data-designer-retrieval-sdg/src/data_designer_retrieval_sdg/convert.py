# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert raw SDG output to Automodel-compatible retriever training formats.

Produces:
- ``train.json`` / ``val.json`` -- NeMo Retriever training format
- ``eval_beir/`` -- BEIR-compatible evaluation format
- ``corpus/`` -- parquet corpus + merlin metadata

Supports random, dedupped (union-find merged), and cluster split strategies.
"""

from __future__ import annotations

import glob as glob_mod
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from data_designer_retrieval_sdg.postprocess import filter_qa_pairs_by_quality

# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------


def filter_mismatched_records(records: list[dict]) -> tuple[list[dict], int]:
    """Drop records where evaluation and pair counts disagree.

    Args:
        records: Raw JSON records from the SDG pipeline.

    Returns:
        Tuple of ``(filtered_records, dropped_count)``.
    """
    filtered: list[dict] = []
    dropped_count = 0

    for record in records:
        qa_evals = record.get("qa_evaluations", {}).get("evaluations", [])
        dedup_pairs = record.get("deduplicated_qa_pairs", [])
        if len(qa_evals) == len(dedup_pairs):
            filtered.append(record)
        else:
            dropped_count += 1
            file_name = record.get("file_name", "unknown")
            display = file_name if isinstance(file_name, str) else ", ".join(file_name) if file_name else "unknown"
            print(
                f"  Dropping record '{display}': "
                f"qa_evaluations={len(qa_evals)}, deduplicated_qa_pairs={len(dedup_pairs)}"
            )

    return filtered, dropped_count


def normalize_file_name(file_name: object) -> list[str]:
    """Normalise *file_name* to a list of strings.

    Provides backward compatibility for old data where ``file_name`` was a
    plain string.

    Args:
        file_name: String, list of strings, or other.

    Returns:
        List of file-name strings.
    """
    if isinstance(file_name, str):
        return [file_name]
    if isinstance(file_name, list):
        return file_name
    if hasattr(file_name, "tolist"):
        value = file_name.tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
    return [str(file_name)]


def _load_json_records(input_file: Path) -> list[dict]:
    """Load records from a JSON file containing one object or a list of objects."""
    with input_file.open(encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, list):
        return records
    return [records]


def _load_jsonl_records(input_file: Path) -> list[dict]:
    """Load records from a JSONL file containing one JSON object per line."""
    records: list[dict] = []
    with input_file.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, list):
                records.extend(record)
            else:
                records.append(record)
    return records


def _load_parquet_records(input_file: Path) -> list[dict]:
    """Load records from a parquet file exported by DataDesigner."""
    return pd.read_parquet(input_file).to_dict(orient="records")


def _load_generated_records_file(input_file: Path) -> list[dict]:
    """Load generated records from one supported file path."""
    suffix = input_file.suffix.lower()
    if suffix == ".json":
        print(f"Loading JSON file: {input_file}")
        return _load_json_records(input_file)
    if suffix == ".jsonl":
        print(f"Loading JSONL file: {input_file}")
        return _load_jsonl_records(input_file)
    if suffix == ".parquet":
        print(f"Loading parquet file: {input_file}")
        return _load_parquet_records(input_file)
    raise ValueError(f"Unsupported generated data file format: {input_file}")


def _discover_generated_record_files(input_dir: Path) -> list[Path]:
    """Discover generated output files in a directory, preferring the newest output contract."""
    pattern_groups = [
        "*.jsonl",
        "generated_batch*.json",
        "*.json",
        "*.parquet",
    ]
    for pattern in pattern_groups:
        files = sorted(Path(p) for p in glob_mod.glob(str(input_dir / pattern)))
        if files:
            return files
    return []


def load_generated_json_files(input_path: str) -> pd.DataFrame:
    """Load generated records from a file or output directory.

    Args:
        input_path: Path to a generated ``.jsonl``, ``.json``, or ``.parquet``
            file, or a directory containing generated output files.

    Returns:
        Combined DataFrame with all records.

    Raises:
        ValueError: If no supported generated files are found.
    """
    all_records: list[dict] = []
    path = Path(input_path)

    if path.is_file():
        all_records.extend(_load_generated_records_file(path))
    else:
        generated_files = _discover_generated_record_files(path)
        if not generated_files:
            raise ValueError(f"No generated JSONL, JSON, or parquet files found in {input_path}")

        print(f"Found {len(generated_files)} generated file(s)")
        for generated_file in generated_files:
            all_records.extend(_load_generated_records_file(generated_file))

    print("Normalizing file_name fields...")
    for record in all_records:
        if "file_name" in record:
            record["file_name"] = normalize_file_name(record["file_name"])

    print("Filtering mismatched records...")
    all_records, dropped_count = filter_mismatched_records(all_records)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} records with mismatched qa_evaluations/deduplicated_qa_pairs sizes")

    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} total records")
    return df


# ---------------------------------------------------------------------------
# Corpus / chunk mapping
# ---------------------------------------------------------------------------


def get_corpus_id(text: str) -> str:
    """Generate a hash-based corpus ID from text content.

    Args:
        text: Document text.

    Returns:
        ID in ``d_<16-hex-char>`` format.
    """
    return "d_" + hashlib.sha256(text.encode()).hexdigest()[:16]


def extract_base_filename(file_path: str) -> str:
    """Return the base filename without extension.

    Args:
        file_path: Absolute or relative file path.

    Returns:
        Filename stem.
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def get_file_identifier(file_name_list: list[str]) -> str:
    """Derive a canonical identifier from a file-name list.

    Single-document bundles use the base filename; multi-document bundles
    use a truncated hash of sorted paths.

    Args:
        file_name_list: List of file names in the bundle.

    Returns:
        String identifier for chunk-mapping lookups.
    """
    if not file_name_list:
        return ""
    if len(file_name_list) == 1:
        return extract_base_filename(file_name_list[0])
    return hashlib.md5("||".join(sorted(file_name_list)).encode()).hexdigest()[:16]


def build_corpus_and_mappings(
    generated_df: pd.DataFrame,
) -> tuple[dict[str, str], dict[tuple[str, int], str]]:
    """Build a deduplicated corpus and chunk-mapping from generated data.

    Args:
        generated_df: DataFrame with ``file_name`` and ``chunks`` columns.

    Returns:
        Tuple of ``(corpus, chunk_mapping)`` where *corpus* maps
        ``text -> corpus_id`` and *chunk_mapping* maps
        ``(file_identifier, chunk_id) -> text``.
    """
    corpus: dict[str, str] = {}
    chunk_mapping: dict[tuple[str, int], str] = {}

    print("Building corpus and chunk mappings...")

    for _, row in generated_df.iterrows():
        file_name_list = row.get("file_name", [])
        chunks = row.get("chunks", [])

        if not chunks or not file_name_list:
            continue

        file_identifier = get_file_identifier(file_name_list)

        if hasattr(chunks, "tolist"):
            chunks = chunks.tolist()

        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id")
                text = chunk.get("text", "")
            else:
                chunk_id = getattr(chunk, "chunk_id", None)
                text = getattr(chunk, "text", "")

            if chunk_id is None or not text:
                continue

            chunk_mapping[(file_identifier, chunk_id)] = text
            if text not in corpus:
                corpus[text] = get_corpus_id(text)

    print(f"Built corpus with {len(corpus)} unique documents from {len(chunk_mapping)} total chunks")
    return corpus, chunk_mapping


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------


def file_tuple_in_set(file_name: object, file_set: set[tuple[str, ...]]) -> bool:
    """Check whether *file_name* (list or str) belongs to *file_set*.

    Args:
        file_name: A list of strings or a single string.
        file_set: Set of tuples to test membership against.

    Returns:
        ``True`` when the normalised tuple is in *file_set*.
    """
    file_tuple = tuple(file_name) if isinstance(file_name, list) else (file_name,)
    return file_tuple in file_set


def create_train_val_test_split(
    filtered_qa_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly split QA pairs by file/bundle into train, val, and test.

    Args:
        filtered_qa_df: DataFrame with filtered QA pairs.
        train_ratio: Fraction of files for training.
        val_ratio: Fraction of files for validation.
        seed: Random seed.

    Returns:
        ``(train_df, val_df, test_df)``

    Raises:
        ValueError: If ``train_ratio + val_ratio > 1.0``.
    """
    random.seed(seed)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be <= 1.0")

    unique_file_tuples = list({tuple(f) if isinstance(f, list) else (f,) for f in filtered_qa_df["file_name"]})
    random.shuffle(unique_file_tuples)

    n_train = int(len(unique_file_tuples) * train_ratio)
    n_val = int(len(unique_file_tuples) * val_ratio)

    train_files = set(unique_file_tuples[:n_train])
    val_files = set(unique_file_tuples[n_train : n_train + n_val])
    test_files = set(unique_file_tuples[n_train + n_val :])

    train_df = filtered_qa_df[filtered_qa_df["file_name"].apply(lambda f: file_tuple_in_set(f, train_files))]
    val_df = filtered_qa_df[filtered_qa_df["file_name"].apply(lambda f: file_tuple_in_set(f, val_files))]
    test_df = filtered_qa_df[filtered_qa_df["file_name"].apply(lambda f: file_tuple_in_set(f, test_files))]

    print(
        f"Split: {len(train_files)} train files/bundles ({len(train_df)} QA pairs), "
        f"{len(val_files)} val files/bundles ({len(val_df)} QA pairs), "
        f"{len(test_files)} test files/bundles ({len(test_df)} QA pairs)"
    )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Group-aware split helpers (dedupped / cluster)
# ---------------------------------------------------------------------------


class UnionFind:
    """Disjoint-set / Union-Find with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find the root representative of *x*."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge the sets containing *x* and *y*."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def load_dedup_groups(json_paths: list[str]) -> dict[str, list[str]]:
    """Load groups/clusters from dedup_groups.json files.

    Auto-detects method keys (``exact``, ``fuzzy``, ``semantic``) and
    extracts ``groups`` or ``clusters``.

    Args:
        json_paths: Paths to dedup group JSON files.

    Returns:
        Unified mapping of ``group_id -> [doc_id, ...]``.
    """
    all_groups: dict[str, list[str]] = {}

    for path in json_paths:
        print(f"  Loading dedup groups from: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for method_key in ("exact", "fuzzy", "semantic"):
            if method_key not in data:
                continue
            method_data = data[method_key]
            groups = method_data.get("groups") or method_data.get("clusters", {})
            n_before = len(all_groups)
            for group_id, doc_list in groups.items():
                all_groups[group_id] = doc_list
            n_added = len(all_groups) - n_before
            n_docs = sum(len(v) for v in groups.values())
            print(f"    {method_key}: {n_added} groups, {n_docs} docs")

    print(f"  Total loaded: {len(all_groups)} groups")
    return all_groups


def merge_groups_union_find(all_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Transitively merge overlapping groups via Union-Find.

    Args:
        all_groups: ``group_id -> [doc_id, ...]`` mapping.

    Returns:
        Merged super-groups (only groups with 2+ members).
    """
    uf = UnionFind()

    for doc_list in all_groups.values():
        if len(doc_list) < 2:
            continue
        anchor = doc_list[0]
        for doc_id in doc_list[1:]:
            uf.union(anchor, doc_id)

    all_docs: set[str] = set()
    for doc_list in all_groups.values():
        all_docs.update(doc_list)

    components: dict[str, set[str]] = defaultdict(set)
    for doc_id in all_docs:
        root = uf.find(doc_id)
        components[root].add(doc_id)

    merged: dict[str, list[str]] = {}
    for i, (_, members) in enumerate(sorted(components.items(), key=lambda x: -len(x[1])), 1):
        if len(members) >= 2:
            merged[f"merged_{i:04d}"] = sorted(members)

    total_docs = sum(len(v) for v in merged.values())
    print(f"  Merged into {len(merged)} super-groups covering {total_docs} docs (from {len(all_groups)} input groups)")
    return merged


def build_file_to_group_mapping(
    groups: dict[str, list[str]],
    qa_file_names: set[str],
) -> dict[str, str]:
    """Map QA file names to group IDs with fallback matching.

    Matching order: exact string, strip extension, basename.

    Args:
        groups: ``group_id -> [doc_id, ...]``.
        qa_file_names: Set of individual file paths from the QA DataFrame.

    Returns:
        Mapping of ``file_name -> group_id`` (only matched files).
    """
    doc_to_group: dict[str, str] = {}
    for group_id, doc_list in groups.items():
        for doc_id in doc_list:
            doc_to_group[doc_id] = group_id

    noext_to_doc = {os.path.splitext(d)[0]: d for d in doc_to_group}
    basename_to_doc = {extract_base_filename(d): d for d in doc_to_group}

    file_to_group: dict[str, str] = {}
    matched = 0
    unmatched = 0

    for fname in qa_file_names:
        if fname in doc_to_group:
            file_to_group[fname] = doc_to_group[fname]
            matched += 1
            continue

        fname_noext = os.path.splitext(fname)[0]
        if fname_noext in doc_to_group:
            file_to_group[fname] = doc_to_group[fname_noext]
            matched += 1
            continue
        if fname_noext in noext_to_doc:
            file_to_group[fname] = doc_to_group[noext_to_doc[fname_noext]]
            matched += 1
            continue

        bn = extract_base_filename(fname)
        if bn in basename_to_doc:
            file_to_group[fname] = doc_to_group[basename_to_doc[bn]]
            matched += 1
            continue

        unmatched += 1

    print(f"  File matching: {matched} matched, {unmatched} unmatched (out of {len(qa_file_names)} QA files)")
    return file_to_group


def create_group_aware_split(
    filtered_qa_df: pd.DataFrame,
    file_to_group: dict[str, str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split QA pairs into train/val/test respecting group boundaries.

    Uses greedy bin-packing sorted by weight (QA-pair count) descending.

    Args:
        filtered_qa_df: DataFrame with filtered QA pairs.
        file_to_group: Mapping from individual file paths to group IDs.
        train_ratio: Target ratio for training.
        val_ratio: Target ratio for validation.
        seed: Random seed.

    Returns:
        ``(train_df, val_df, test_df)``

    Raises:
        ValueError: If ``train_ratio + val_ratio > 1.0``.
    """
    random.seed(seed)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be <= 1.0")

    unique_file_tuples = list({tuple(f) if isinstance(f, list) else (f,) for f in filtered_qa_df["file_name"]})

    file_tuple_counts: dict[tuple[str, ...], int] = {}
    for ft in unique_file_tuples:
        mask = filtered_qa_df["file_name"].apply(lambda f, _ft=ft: (tuple(f) if isinstance(f, list) else (f,)) == _ft)
        file_tuple_counts[ft] = int(mask.sum())

    group_to_file_tuples: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    singleton_file_tuples: list[tuple[str, ...]] = []

    for ft in unique_file_tuples:
        matched_group = None
        for fname in ft:
            if fname in file_to_group:
                matched_group = file_to_group[fname]
                break
        if matched_group is not None:
            group_to_file_tuples[matched_group].append(ft)
        else:
            singleton_file_tuples.append(ft)

    units: list[tuple[str, list[tuple[str, ...]], int]] = []
    for group_id, file_tuples in group_to_file_tuples.items():
        weight = sum(file_tuple_counts[ft] for ft in file_tuples)
        units.append((group_id, file_tuples, weight))
    for ft in singleton_file_tuples:
        units.append((f"singleton_{ft}", [ft], file_tuple_counts[ft]))

    random.shuffle(units)
    units.sort(key=lambda x: -x[2])

    total_qa = sum(u[2] for u in units)
    targets = {"train": total_qa * train_ratio, "val": total_qa * val_ratio, "test": total_qa * test_ratio}
    current: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_assignments: dict[str, set[tuple[str, ...]]] = {"train": set(), "val": set(), "test": set()}

    for _, file_tuples, weight in units:
        deficits = {s: targets[s] - current[s] for s in targets}
        best_split = max(deficits, key=deficits.get)  # type: ignore[arg-type]
        for ft in file_tuples:
            split_assignments[best_split].add(ft)
        current[best_split] += weight

    train_df = filtered_qa_df[
        filtered_qa_df["file_name"].apply(lambda f: file_tuple_in_set(f, split_assignments["train"]))
    ]
    val_df = filtered_qa_df[filtered_qa_df["file_name"].apply(lambda f: file_tuple_in_set(f, split_assignments["val"]))]
    test_df = filtered_qa_df[
        filtered_qa_df["file_name"].apply(lambda f: file_tuple_in_set(f, split_assignments["test"]))
    ]

    n_groups = len(group_to_file_tuples)
    n_singletons = len(singleton_file_tuples)
    print(f"  Groups: {n_groups} multi-file groups, {n_singletons} singletons")
    print(
        f"  Split: train={len(train_df)} QA pairs ({len(split_assignments['train'])} files), "
        f"val={len(val_df)} ({len(split_assignments['val'])} files), "
        f"test={len(test_df)} ({len(split_assignments['test'])} files)"
    )
    if total_qa > 0:
        print(
            f"  Actual ratios: train={len(train_df) / total_qa:.3f}, "
            f"val={len(val_df) / total_qa:.3f}, test={len(test_df) / total_qa:.3f}"
        )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_training_set(
    corpus: dict[str, str],
    chunk_mapping: dict[tuple[str, int], str],
    train_df: pd.DataFrame,
    output_dir: str,
    corpus_id: str,
    max_pos_docs: int = 5,
    output_filename: str = "train.json",
    set_name: str = "training",
    write_corpus: bool = True,
) -> None:
    """Generate a training/validation set in NeMo Retriever format.

    Args:
        corpus: ``text -> corpus_id`` mapping.
        chunk_mapping: ``(file_identifier, chunk_id) -> text`` mapping.
        train_df: DataFrame with QA pairs for this split.
        output_dir: Output directory path.
        corpus_id: Corpus identifier string.
        max_pos_docs: Maximum positive docs per query.
        output_filename: Name of the output JSON file.
        set_name: Label for log messages (e.g. ``"training"``).
        write_corpus: Whether to write corpus parquet and metadata.
    """
    print(f"Generating {set_name} set...")

    corpus_dir = os.path.join(output_dir, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    training_data: list[dict] = []
    question_counter = 0
    skipped_queries = 0
    skipped_too_many_pos = 0

    for _, qa_pair in train_df.iterrows():
        file_name_list = qa_pair.get("file_name", [])
        file_identifier = get_file_identifier(file_name_list) if file_name_list else ""
        segment_ids = qa_pair.get("segment_ids", [])
        question = qa_pair.get("question", "")

        if not question:
            skipped_queries += 1
            continue

        if hasattr(segment_ids, "tolist"):
            segment_ids = segment_ids.tolist()

        if len(segment_ids) > max_pos_docs:
            skipped_too_many_pos += 1
            continue

        pos_docs: list[dict] = []
        all_segments_exist = True
        for segment_id in segment_ids:
            key = (file_identifier, segment_id)
            if key not in chunk_mapping:
                all_segments_exist = False
                break
            text = chunk_mapping[key]
            pos_docs.append({"id": corpus[text]})

        if not all_segments_exist or not pos_docs:
            skipped_queries += 1
            continue

        training_data.append(
            {
                "question_id": f"q{question_counter}",
                "question": question,
                "corpus_id": corpus_id,
                "pos_doc": pos_docs,
                "neg_doc": [],
            }
        )
        question_counter += 1

    print(f"  Generated {len(training_data)} {set_name} queries")
    if skipped_queries > 0:
        print(f"  Skipped {skipped_queries} queries (missing segments or empty question)")
    if skipped_too_many_pos > 0:
        print(f"  Skipped {skipped_too_many_pos} queries (exceeded max_pos_docs={max_pos_docs})")

    train_json_path = os.path.join(output_dir, output_filename)
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump({"corpus": {"path": "./corpus/"}, "data": training_data}, f, indent=2, sort_keys=False)
    print(f"  Wrote {train_json_path}")

    if write_corpus:
        corpus_list = [{"id": doc_id, "text": text} for text, doc_id in corpus.items()]
        corpus_df = pd.DataFrame(corpus_list)
        parquet_path = os.path.join(corpus_dir, "train.parquet")
        corpus_df.to_parquet(parquet_path, index=False)
        print(f"  Wrote {parquet_path} with {len(corpus_list)} documents")

        metadata_path = os.path.join(corpus_dir, "merlin_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"corpus_id": corpus_id, "class": "TextQADataset"}, f, indent=2, sort_keys=False)
        print(f"  Wrote {metadata_path}")


def generate_eval_set(
    corpus: dict[str, str],
    chunk_mapping: dict[tuple[str, int], str],
    eval_df: pd.DataFrame,
    output_dir: str,
    max_pos_docs: int = 5,
    eval_only: bool = False,
    use_group_id_in_eval: bool = False,
) -> None:
    """Generate an evaluation set in BEIR format.

    Args:
        corpus: ``text -> corpus_id`` mapping.
        chunk_mapping: ``(file_identifier, chunk_id) -> text`` mapping.
        eval_df: DataFrame with QA pairs for evaluation.
        output_dir: Output directory path.
        max_pos_docs: Maximum positive docs per query.
        eval_only: If ``True`` write directly to *output_dir* instead of
            an ``eval_beir/`` sub-directory.
        use_group_id_in_eval: Use hash-based group ID in qrels instead of
            sequential BEIR IDs.
    """
    print("Generating evaluation set...")

    eval_dir = output_dir if eval_only else os.path.join(output_dir, "eval_beir")
    os.makedirs(eval_dir, exist_ok=True)

    corpus_path = os.path.join(eval_dir, "corpus.jsonl")
    corpus_id_counter = 0
    text_to_beir_id: dict[str, str] = {}

    with open(corpus_path, "w", encoding="utf-8") as corpus_file:
        for text, hash_id in corpus.items():
            beir_id = f"d{corpus_id_counter}"
            text_to_beir_id[text] = beir_id

            corpus_entry: dict = {"_id": beir_id, "metadata": {}, "text": text, "title": ""}
            if use_group_id_in_eval:
                corpus_entry["group_id"] = hash_id
            corpus_file.write(json.dumps(corpus_entry) + "\n")
            corpus_id_counter += 1

    print(f"  Wrote {corpus_path} with {corpus_id_counter} documents")

    queries_path = os.path.join(eval_dir, "queries.jsonl")
    query_mappings: list[tuple[str, str, list]] = []
    query_counter = 0
    skipped_queries = 0
    skipped_too_many_pos = 0

    with open(queries_path, "w", encoding="utf-8") as queries_file:
        for _, qa_pair in eval_df.iterrows():
            file_name_list = qa_pair.get("file_name", [])
            file_identifier = get_file_identifier(file_name_list) if file_name_list else ""
            segment_ids = qa_pair.get("segment_ids", [])
            question = qa_pair.get("question", "")

            if not question:
                skipped_queries += 1
                continue

            if hasattr(segment_ids, "tolist"):
                segment_ids = segment_ids.tolist()

            if len(segment_ids) > max_pos_docs:
                skipped_too_many_pos += 1
                continue

            all_segments_exist = True
            for segment_id in segment_ids:
                key = (file_identifier, segment_id)
                if key not in chunk_mapping:
                    all_segments_exist = False
                    break

            if not all_segments_exist:
                skipped_queries += 1
                continue

            query_id = f"q{query_counter}"
            query_mappings.append((query_id, file_identifier, segment_ids))

            metadata: dict = {}
            for field in (
                "query_type",
                "reasoning_type",
                "hop_count",
                "question_complexity",
                "quality_score",
                "answer",
                "hop_contexts",
            ):
                val = qa_pair.get(field)
                if val is not None:
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    metadata[field] = val

            metadata["file_name"] = file_name_list
            metadata["segment_ids"] = segment_ids

            query_entry = {"_id": query_id, "metadata": metadata, "text": question}
            queries_file.write(json.dumps(query_entry) + "\n")
            query_counter += 1

    print(f"  Wrote {queries_path} with {query_counter} queries")
    if skipped_queries > 0:
        print(f"  Skipped {skipped_queries} queries (missing segments or empty question)")
    if skipped_too_many_pos > 0:
        print(f"  Skipped {skipped_too_many_pos} queries (exceeded max_pos_docs={max_pos_docs})")

    qrels_dir = os.path.join(eval_dir, "qrels")
    os.makedirs(qrels_dir, exist_ok=True)

    qrels_path = os.path.join(qrels_dir, "test.tsv")
    qrels_count = 0

    with open(qrels_path, "w", encoding="utf-8") as qrels_file:
        qrels_file.write("query-id\tcorpus-id\tscore\n")
        for query_id, file_identifier, segment_ids in query_mappings:
            for segment_id in segment_ids:
                key = (file_identifier, segment_id)
                text = chunk_mapping[key]
                if use_group_id_in_eval:
                    doc_id = corpus[text]
                else:
                    doc_id = text_to_beir_id[text]
                qrels_file.write(f"{query_id}\t{doc_id}\t1\n")
                qrels_count += 1

    id_type = "group_id" if use_group_id_in_eval else "_id"
    print(f"  Wrote {qrels_path} with {qrels_count} mappings (using {id_type})")


# ---------------------------------------------------------------------------
# Top-level conversion orchestrator
# ---------------------------------------------------------------------------


def run_conversion(
    input_path: str,
    corpus_id: str,
    output_dir: str | None = None,
    eval_only: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    quality_threshold: float = 7.0,
    max_pos_docs: int = 5,
    use_group_id_in_eval: bool = False,
    split_strategy: str = "random",
    groups_json: list[str] | None = None,
) -> None:
    """Run the full SDG-to-retriever-data conversion pipeline.

    Args:
        input_path: Path to a merged JSON file or directory of batch files.
        corpus_id: Corpus identifier.
        output_dir: Output directory (auto-derived if ``None``).
        eval_only: Generate only BEIR evaluation data.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        seed: Random seed.
        quality_threshold: Minimum quality score.
        max_pos_docs: Maximum positive docs per query.
        use_group_id_in_eval: Use hash-based group IDs in eval qrels.
        split_strategy: ``"random"``, ``"dedupped"``, or ``"cluster"``.
        groups_json: Paths to dedup group JSON files.
    """
    abs_input = os.path.abspath(input_path)
    if not os.path.exists(abs_input):
        raise ValueError(f"Input path does not exist: {abs_input}")

    if output_dir is None:
        suffix = "_eval" if eval_only else "_train_eval"
        if os.path.isfile(abs_input):
            input_basename = os.path.splitext(os.path.basename(abs_input))[0]
            output_dir = os.path.join(os.path.dirname(abs_input), f"{input_basename}{suffix}")
        else:
            output_dir = os.path.abspath(abs_input.rstrip("/") + suffix)
    else:
        output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    _print_conversion_header(
        abs_input,
        output_dir,
        corpus_id,
        eval_only,
        train_ratio,
        val_ratio,
        split_strategy,
        groups_json,
        seed,
        quality_threshold,
        max_pos_docs,
        use_group_id_in_eval,
    )

    generated_df = load_generated_json_files(abs_input)
    corpus, chunk_mapping = build_corpus_and_mappings(generated_df)
    filtered_qa_df, skipped_files = filter_qa_pairs_by_quality(generated_df, quality_threshold)

    if eval_only:
        generate_eval_set(
            corpus,
            chunk_mapping,
            filtered_qa_df,
            output_dir,
            max_pos_docs,
            eval_only=True,
            use_group_id_in_eval=use_group_id_in_eval,
        )
    else:
        train_df, val_df, test_df = _compute_split(
            filtered_qa_df,
            train_ratio,
            val_ratio,
            seed,
            split_strategy,
            groups_json,
        )
        generate_training_set(
            corpus,
            chunk_mapping,
            train_df,
            output_dir,
            corpus_id,
            max_pos_docs,
            output_filename="train.json",
            set_name="training",
        )
        generate_training_set(
            corpus,
            chunk_mapping,
            val_df,
            output_dir,
            corpus_id,
            max_pos_docs,
            output_filename="val.json",
            set_name="validation",
            write_corpus=False,
        )
        generate_eval_set(
            corpus,
            chunk_mapping,
            test_df,
            output_dir,
            max_pos_docs,
            eval_only=False,
            use_group_id_in_eval=use_group_id_in_eval,
        )

    _print_conversion_footer(output_dir, eval_only, skipped_files)


# ---------------------------------------------------------------------------
# Conversion internal helpers
# ---------------------------------------------------------------------------


def _compute_split(
    filtered_qa_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    split_strategy: str,
    groups_json: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Route to the correct split strategy."""
    if split_strategy == "random":
        return create_train_val_test_split(filtered_qa_df, train_ratio, val_ratio, seed)

    if not groups_json:
        raise ValueError(f"--groups-json is required when split_strategy={split_strategy}")

    groups = load_dedup_groups(groups_json)
    if split_strategy == "dedupped":
        groups = merge_groups_union_find(groups)

    qa_file_names: set[str] = set()
    for fnames in filtered_qa_df["file_name"]:
        if isinstance(fnames, list):
            qa_file_names.update(fnames)
        else:
            qa_file_names.add(fnames)

    ftg = build_file_to_group_mapping(groups, qa_file_names)
    return create_group_aware_split(filtered_qa_df, ftg, train_ratio, val_ratio, seed)


def _print_conversion_header(
    input_path: str,
    output_dir: str,
    corpus_id: str,
    eval_only: bool,
    train_ratio: float,
    val_ratio: float,
    split_strategy: str,
    groups_json: list[str] | None,
    seed: int,
    quality_threshold: float,
    max_pos_docs: int,
    use_group_id_in_eval: bool,
) -> None:
    """Print a banner with the conversion settings."""
    print("=" * 80)
    print("SDG to Retriever Data Converter")
    print("=" * 80)
    print(f"Input path: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Corpus ID: {corpus_id}")
    if eval_only:
        print("Mode: Evaluation only (BEIR format)")
    else:
        test_ratio = 1.0 - train_ratio - val_ratio
        print("Mode: Train/Val/Test split")
        print(f"Split strategy: {split_strategy}")
        print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio:.2f}")
        if groups_json:
            for gj in groups_json:
                print(f"  Groups JSON: {gj}")
    print(f"Random seed: {seed}")
    print(f"Quality threshold: {quality_threshold}")
    print(f"Max positive docs: {max_pos_docs}")
    print(f"Eval qrels ID type: {'group_id' if use_group_id_in_eval else '_id'}")
    print()


def _print_conversion_footer(output_dir: str, eval_only: bool, skipped_files: list[dict]) -> None:
    """Print completion summary."""
    print()
    print("=" * 80)
    print("Conversion complete!")
    print("=" * 80)
    print(f"Output location: {output_dir}")
    if eval_only:
        print("Generated (BEIR format):")
        print("  - corpus.jsonl")
        print("  - queries.jsonl")
        print("  - qrels/test.tsv")
    else:
        print("Generated:")
        print("  - train.json (retriever training format)")
        print("  - val.json (retriever validation format)")
        print("  - corpus/ (parquet + metadata)")
        print("  - eval_beir/ (BEIR test/evaluation format)")

    if skipped_files:
        print()
        print("=" * 80)
        print(f"Skipped Files ({len(skipped_files)} total)")
        print("=" * 80)
        for item in skipped_files:
            print(f"  - {item['file_name']}: {item['reason']}")
