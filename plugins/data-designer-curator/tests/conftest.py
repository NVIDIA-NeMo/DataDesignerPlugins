# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.storage.artifact_storage import ArtifactStorage


@pytest.fixture
def resource_provider(tmp_path: Path) -> ResourceProvider:
    return ResourceProvider(artifact_storage=ArtifactStorage(artifact_path=tmp_path))
