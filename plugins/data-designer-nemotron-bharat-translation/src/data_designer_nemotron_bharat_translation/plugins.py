# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer plugin registrations exported by this package.

One ``data_designer.plugins`` entry point is wired here:

- :data:`document_translation_plugin` -- direct translate + MQM evaluate +
  optional rewrite loop, optionally windowed for long documents and
  finished with a deterministic script remap
  (``column_type="document-translation"``).
"""

from data_designer.plugins.plugin import Plugin, PluginType

document_translation_plugin = Plugin(
    config_qualified_name="data_designer_nemotron_bharat_translation.config.DocumentTranslationConfig",
    impl_qualified_name="data_designer_nemotron_bharat_translation.document_translation.DocumentTranslationColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
