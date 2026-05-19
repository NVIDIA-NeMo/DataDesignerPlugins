# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.plugins.plugin import Plugin, PluginType

from data_designer_curator.plugin import curator_modify_plugin, curator_text_filter_plugin, exact_dedup_plugin, plugins
from data_designer_curator.processors.dedup import ExactDedupProcessor
from data_designer_curator.processors.filters import CuratorTextFilterProcessor
from data_designer_curator.processors.modifiers import CuratorModifyProcessor


@pytest.mark.parametrize("plugin", plugins)
def test_valid_plugin(plugin: Plugin) -> None:
    assert_valid_plugin(plugin)


@pytest.mark.parametrize(
    ("plugin", "plugin_type", "impl_cls"),
    [
        (exact_dedup_plugin, PluginType.PROCESSOR, ExactDedupProcessor),
        (curator_modify_plugin, PluginType.PROCESSOR, CuratorModifyProcessor),
        (curator_text_filter_plugin, PluginType.PROCESSOR, CuratorTextFilterProcessor),
    ],
)
def test_plugin_impl_types(plugin: Plugin, plugin_type: PluginType, impl_cls: type) -> None:
    assert plugin.plugin_type == plugin_type
    assert plugin.impl_cls is impl_cls
    assert issubclass(plugin.impl_cls, Processor)
