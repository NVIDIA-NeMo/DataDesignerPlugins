# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.engine.testing.utils import assert_valid_plugin

from data_designer_retrieval_sdg.plugin import plugin


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)
