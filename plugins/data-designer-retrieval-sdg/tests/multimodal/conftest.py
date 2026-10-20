# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local credentials for the invented deployments used by offline tests."""

import pytest


@pytest.fixture(autouse=True)
def offline_model_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply the fixture deployment's placeholder without requiring user secrets."""
    monkeypatch.setenv("TEST_SDG_KEY", "offline-placeholder")
