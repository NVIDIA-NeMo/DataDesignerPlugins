# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.errors import DataDesignerError


class DataDesignerCuratorError(DataDesignerError):
    """Base error for Data Designer Curator plugins."""


class CuratorDependencyError(DataDesignerCuratorError):
    """Raised when optional NeMo Curator dependencies are unavailable."""


class CuratorExecutionError(DataDesignerCuratorError):
    """Raised when NeMo Curator execution fails."""


class RemoteScoringError(DataDesignerCuratorError):
    """Raised when a remote scoring endpoint fails."""
