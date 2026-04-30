# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-registration tests for both entry points."""

from data_designer.engine.testing.utils import assert_valid_plugin

from data_designer_retrieval_sdg.plugins import document_chunker_plugin, embedding_dedup_plugin


def test_embedding_dedup_plugin_valid() -> None:
    assert_valid_plugin(embedding_dedup_plugin)


def test_document_chunker_plugin_valid() -> None:
    assert_valid_plugin(document_chunker_plugin)
