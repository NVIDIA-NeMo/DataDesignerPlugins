# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

from data_designer_curator.config import (
    ExactDedupProcessorConfig,
    RemoteScoreColumnConfig,
    ScoreFilterProcessorConfig,
)

_CONFIG_TYPES = (ExactDedupProcessorConfig, ScoreFilterProcessorConfig, RemoteScoreColumnConfig)

exact_dedup_plugin = Plugin(
    config_qualified_name="data_designer_curator.config.ExactDedupProcessorConfig",
    impl_qualified_name="data_designer_curator.processors.dedup.ExactDedupProcessor",
    plugin_type=PluginType.PROCESSOR,
)

score_filter_plugin = Plugin(
    config_qualified_name="data_designer_curator.config.ScoreFilterProcessorConfig",
    impl_qualified_name="data_designer_curator.processors.filters.ScoreFilterProcessor",
    plugin_type=PluginType.PROCESSOR,
)

remote_score_plugin = Plugin(
    config_qualified_name="data_designer_curator.config.RemoteScoreColumnConfig",
    impl_qualified_name="data_designer_curator.columns.remote_score.RemoteScoreColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

plugins = [exact_dedup_plugin, score_filter_plugin, remote_score_plugin]
