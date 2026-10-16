# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded generic text/figure/table instructions adapted from the reference samplers."""

from __future__ import annotations

import json
import random
from pathlib import Path

from data_designer_retrieval_sdg.multimodal.models import QueryInstruction

FORMATS = ["question", "keyword", "instruction"]
FORBIDDEN = {"multi-hop": {"keyword"}, "enumerative": {"keyword"}, "boolean": {"keyword", "instruction"}}


def sample_instructions(seed: int, context_id: str) -> list[QueryInstruction]:
    """Sample one weighted instruction per evidence category using a local RNG.

    Args:
        seed: Run seed.
        context_id: Stable context identity; unrelated context order has no effect.

    Returns:
        Text, figure and table instructions, including requested type and answerability.
    """
    rng = random.Random(f"{seed}:{context_id}")
    pools = json.loads(Path(__file__).with_name("query_modules.json").read_text())
    result = []
    for kind in ("text", "figure", "table"):
        pool = pools[kind]
        modules = []
        for module in pool["modules"]:
            query_format = rng.choice([fmt for fmt in FORMATS if fmt not in FORBIDDEN.get(module["type"], set())])
            modules.append(
                QueryInstruction(
                    name=kind,
                    instruction=module["instruction"],
                    query_type=module["type"],
                    format=query_format,
                    modality="text" if kind == "text" else "image",
                    answerability=module["answerability"],
                )
            )
        result.extend(rng.choices(modules, weights=pool["weights"], k=1))
    return result
