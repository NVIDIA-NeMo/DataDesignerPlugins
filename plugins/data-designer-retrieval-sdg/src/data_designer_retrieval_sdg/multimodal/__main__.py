# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run generic multimodal retrieval SDG from an explicit YAML configuration."""

import argparse
from pathlib import Path

import yaml

from data_designer_retrieval_sdg.multimodal import MultimodalSDGConfig, run_multimodal_sdg


def main() -> None:
    """Load an operator-owned configuration and publish a portable handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = MultimodalSDGConfig.model_validate(yaml.safe_load(args.config.read_text()))
    print(run_multimodal_sdg(config))


if __name__ == "__main__":
    main()
