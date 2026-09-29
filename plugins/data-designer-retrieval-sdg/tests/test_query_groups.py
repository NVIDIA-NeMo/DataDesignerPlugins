# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General query grouping without benchmark schemas."""

import pytest

from data_designer_retrieval_sdg.retrieval.query_groups import QueryGroupInput, resolve_query_groups


def test_optional_provenance_and_normalized_duplicates():
    rows = [
        QueryGroupInput(query_id="a", text="How does the warranty work?"),
        QueryGroupInput(query_id="b", text="HOW does the warranty work !"),
        QueryGroupInput(query_id="c", text="Where can I obtain replacement parts?"),
    ]
    groups = resolve_query_groups(rows)
    assert groups["a"] == groups["b"]
    assert groups["a"] != groups["c"]
    assert groups == resolve_query_groups(list(reversed(rows)))


def test_explicit_groups_and_external_seed_links_are_transitive():
    rows = [
        QueryGroupInput(query_id="a", text="What does the warranty cover?", query_group_id="g"),
        QueryGroupInput(query_id="b", text="Que couvre la garantie ?", query_group_id="g", seed_query_id="seed"),
        QueryGroupInput(query_id="c", text="Describe warranty exclusions.", parent_query_ids=["seed"]),
        QueryGroupInput(query_id="d", text="Explain coverage.", parent_query_ids=["a"]),
    ]
    assert len(set(resolve_query_groups(rows).values())) == 1


def test_near_duplicate_detection_is_optional():
    rows = [
        QueryGroupInput(query_id="a", text="Describe the terms and conditions of the warranty for all products"),
        QueryGroupInput(query_id="b", text="Describe the the terms and conditions of the warranty for all products"),
    ]
    assert len(set(resolve_query_groups(rows).values())) == 2
    assert len(set(resolve_query_groups(rows, group_near_duplicates=True).values())) == 1


@pytest.mark.parametrize("fields", [{"query_group_id": ""}, {"parent_query_ids": [""]}, {"seed_query_id": " "}])
def test_invalid_optional_provenance_fails(fields):
    with pytest.raises(ValueError):
        QueryGroupInput(query_id="a", text="A query", **fields)


def test_duplicate_ids_and_empty_normalized_queries_fail():
    row = QueryGroupInput(query_id="a", text="A query")
    with pytest.raises(ValueError, match="unique"):
        resolve_query_groups([row, row])
    with pytest.raises(ValueError, match="words"):
        resolve_query_groups([QueryGroupInput(query_id="a", text="?!")])
