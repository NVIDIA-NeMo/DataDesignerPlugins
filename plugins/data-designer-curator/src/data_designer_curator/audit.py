# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def write_audit(df: pd.DataFrame, output_dir: Path, filename: str = "audit.parquet") -> None:
    """Write a processor audit artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / filename, index=False)
