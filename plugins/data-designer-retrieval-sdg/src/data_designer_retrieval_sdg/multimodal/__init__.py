# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic multimodal retrieval SDG EA public API."""

from typing import TYPE_CHECKING, Any

from data_designer_retrieval_sdg.multimodal.models import (
    GenerationContext,
    ModelSettings,
    MultimodalSDGConfig,
    QueryInstruction,
)

if TYPE_CHECKING:
    from data_designer_retrieval_sdg.multimodal.workflow import run_multimodal_sdg

__all__ = ["GenerationContext", "ModelSettings", "MultimodalSDGConfig", "QueryInstruction", "run_multimodal_sdg"]


def __getattr__(name: str) -> Any:
    """Load the workflow on demand so entry-point discovery cannot import it early.

    Args:
        name: Public attribute requested from the multimodal package.

    Returns:
        The workflow runner when requested.

    Raises:
        AttributeError: If the name is not a public lazy export.
    """
    if name == "run_multimodal_sdg":
        from data_designer_retrieval_sdg.multimodal.workflow import run_multimodal_sdg

        return run_multimodal_sdg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
