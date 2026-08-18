# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.plugins.plugin import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="data_designer_docx.config.DocxProcessorConfig",
    impl_qualified_name="data_designer_docx.impl.DocxProcessor",
    plugin_type=PluginType.PROCESSOR,
)
