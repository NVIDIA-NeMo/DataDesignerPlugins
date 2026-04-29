# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer plugin registration for the retrieval-sdg-dedup column type."""

from data_designer.plugins.plugin import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="data_designer_retrieval_sdg.config.RetrievalSdgDedupColumnConfig",
    impl_qualified_name="data_designer_retrieval_sdg.dedup.RetrievalSdgDedupColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
