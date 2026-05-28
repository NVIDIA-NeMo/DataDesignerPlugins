# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from data_designer.engine.resources.resource_provider import ResourceProvider

from data_designer_curator.config import CuratorModifierConfig, CuratorModifyProcessorConfig
from data_designer_curator.processors.modifiers import CuratorModifyProcessor


class FakeCuratorTextAdapter:
    calls: list[dict[str, object]] = []

    def modify(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(kwargs)
        data = kwargs["data"]
        assert isinstance(data, pd.DataFrame)
        output = data.copy()
        output["clean_text"] = output["text"].str.replace("http://example.test", "", regex=False).str.strip()
        return output


def test_curator_modify_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
    resource_provider: ResourceProvider,
) -> None:
    FakeCuratorTextAdapter.calls = []
    monkeypatch.setattr("data_designer_curator.processors.modifiers.CuratorTextAdapter", FakeCuratorTextAdapter)
    data = pd.DataFrame({"text": ["hello http://example.test"]})
    config = CuratorModifyProcessorConfig(
        name="clean",
        input_field="text",
        output_field="clean_text",
        modifiers=[CuratorModifierConfig(primitive="url_remover")],
    )

    output = CuratorModifyProcessor(config, resource_provider).process_after_generation(data)

    assert output["clean_text"].tolist() == ["hello"]
    assert FakeCuratorTextAdapter.calls[0]["input_field"] == "text"
    assert FakeCuratorTextAdapter.calls[0]["output_field"] == "clean_text"
    assert FakeCuratorTextAdapter.calls[0]["modifiers"] == [{"primitive": "url_remover", "params": {}}]


def test_curator_modify_raises_for_missing_input(resource_provider: ResourceProvider) -> None:
    data = pd.DataFrame({"other": ["hello"]})
    config = CuratorModifyProcessorConfig(
        name="clean",
        input_field="text",
        modifiers=[CuratorModifierConfig(primitive="url_remover")],
    )

    with pytest.raises(ValueError, match="Missing modifier input column"):
        CuratorModifyProcessor(config, resource_provider).process_after_generation(data)
