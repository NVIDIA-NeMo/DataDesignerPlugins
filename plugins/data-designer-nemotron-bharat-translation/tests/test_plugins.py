# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from data_designer.engine.testing.utils import assert_valid_plugin

from data_designer_nemotron_bharat_translation.plugins import document_translation_plugin

ALL_PLUGINS = [document_translation_plugin]


@pytest.mark.parametrize("plugin", ALL_PLUGINS, ids=[p.config_qualified_name for p in ALL_PLUGINS])
def test_valid_plugin(plugin) -> None:
    assert_valid_plugin(plugin)
