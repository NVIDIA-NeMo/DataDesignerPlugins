# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline defaults; readiness integration tests explicitly enable native checks."""

import pytest


@pytest.fixture(autouse=True)
def offline_model_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply the fixture deployment's placeholder without requiring user secrets."""
    monkeypatch.setenv("TEST_SDG_KEY", "offline-placeholder")
    monkeypatch.setenv("DATA_DESIGNER_SKIP_MODEL_HEALTH_CHECKS", "1")
