# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured column configuration for multimodal inference."""

from typing import Literal

from data_designer.config.column_configs import LLMStructuredColumnConfig


class RetrievalStructuredColumnConfig(LLMStructuredColumnConfig):
    """Generate schema-validated retrieval JSON, with or without a Markdown fence.

    All model, image, prompt, skip, and trace settings retain their native
    Data Designer meanings. The optional skip policy records exhausted model
    validation failures as tagged diagnostics instead of dropping the row.
    """

    column_type: Literal["retrieval-structured"] = "retrieval-structured"
    failed_record_policy: Literal["error", "skip"] = "error"
