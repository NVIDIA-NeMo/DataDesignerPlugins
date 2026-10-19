# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer entry point for multimodal structured generation."""

from data_designer.plugins.plugin import Plugin, PluginType

retrieval_structured_plugin = Plugin(
    config_qualified_name="data_designer_retrieval_sdg.multimodal.config.RetrievalStructuredColumnConfig",
    impl_qualified_name="data_designer_retrieval_sdg.multimodal.structured.RetrievalStructuredCellGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
