# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="data_designer_gym.config.GymTaskProcessorConfig",
    impl_qualified_name="data_designer_gym.impl.GymTaskProcessor",
    plugin_type=PluginType.PROCESSOR,
)

resource_plugin = Plugin(
    config_qualified_name="data_designer_gym.config.GymResourceColumnConfig",
    impl_qualified_name="data_designer_gym.columns.GymResourceColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

harness_plugin = Plugin(
    config_qualified_name="data_designer_gym.config.GymHarnessColumnConfig",
    impl_qualified_name="data_designer_gym.columns.GymHarnessColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
