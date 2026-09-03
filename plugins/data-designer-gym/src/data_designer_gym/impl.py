# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.storage.artifact_storage import BatchStage

from data_designer_gym.config import GymTaskProcessorConfig
from data_designer_gym.conversion import gym_task_artifacts_from_dataframe

if TYPE_CHECKING:
    import pandas as pd


class GymTaskProcessor(Processor[GymTaskProcessorConfig]):
    """Write canonical scenarios and Gym-native processor artifacts."""

    def process_after_batch(
        self,
        data: pd.DataFrame,
        *,
        current_batch_number: int | None,
    ) -> pd.DataFrame:
        """Write Gym tasks while preserving the original Data Designer dataset."""
        tasks = gym_task_artifacts_from_dataframe(data, self.config)
        if tasks.empty:
            return data
        if current_batch_number is not None:
            self.artifact_storage.write_batch_to_parquet_file(
                batch_number=current_batch_number,
                dataframe=tasks,
                batch_stage=BatchStage.PROCESSORS_OUTPUTS,
                subfolder=self.config.name,
            )
        else:
            self.artifact_storage.write_parquet_file(
                parquet_file_name=f"{self.config.name}.parquet",
                dataframe=tasks,
                batch_stage=BatchStage.PROCESSORS_OUTPUTS,
            )
        return data
