# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

from data_designer_curator.config import (
    CuratorDedupProcessorConfig,
    CuratorModifyProcessorConfig,
    CuratorTextFilterProcessorConfig,
)

_CONFIG_TYPES = (CuratorDedupProcessorConfig, CuratorModifyProcessorConfig, CuratorTextFilterProcessorConfig)

curator_dedup_plugin = Plugin(
    config_qualified_name="data_designer_curator.config.CuratorDedupProcessorConfig",
    impl_qualified_name="data_designer_curator.processors.dedup.CuratorDedupProcessor",
    plugin_type=PluginType.PROCESSOR,
)

curator_modify_plugin = Plugin(
    config_qualified_name="data_designer_curator.config.CuratorModifyProcessorConfig",
    impl_qualified_name="data_designer_curator.processors.modifiers.CuratorModifyProcessor",
    plugin_type=PluginType.PROCESSOR,
)

curator_text_filter_plugin = Plugin(
    config_qualified_name="data_designer_curator.config.CuratorTextFilterProcessorConfig",
    impl_qualified_name="data_designer_curator.processors.filters.CuratorTextFilterProcessor",
    plugin_type=PluginType.PROCESSOR,
)

plugins = [curator_dedup_plugin, curator_modify_plugin, curator_text_filter_plugin]
