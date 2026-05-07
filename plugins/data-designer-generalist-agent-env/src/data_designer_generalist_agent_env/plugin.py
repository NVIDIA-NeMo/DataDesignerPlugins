# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

environment_plugin = Plugin(
    config_qualified_name="data_designer_generalist_agent_env.config.GeneralistAgentEnvironmentColumnConfig",
    impl_qualified_name="data_designer_generalist_agent_env.impl.GeneralistAgentEnvironmentColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

task_plugin = Plugin(
    config_qualified_name="data_designer_generalist_agent_env.config.GeneralistAgentTaskColumnConfig",
    impl_qualified_name="data_designer_generalist_agent_env.impl.GeneralistAgentTaskColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
