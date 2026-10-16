# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured column configuration for multimodal inference."""

from typing import Literal

from data_designer.config.column_configs import LLMStructuredColumnConfig


class RetrievalStructuredColumnConfig(LLMStructuredColumnConfig):
    """Generate schema-validated retrieval JSON, with or without a Markdown fence.

    All model, image, prompt, skip, and trace settings retain their native
    Data Designer meanings. Only response decoding uses the plugin parser.
    """

    column_type: Literal["retrieval-structured"] = "retrieval-structured"
