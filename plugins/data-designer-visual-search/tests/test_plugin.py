# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from data_designer.config.models import ModalityDataType
from data_designer.config.run_config import RunConfig
from data_designer.config.utils.image_helpers import ImageFormat
from data_designer.config.utils.trace_type import TraceType
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionResponse, ToolCall
from data_designer.engine.storage.artifact_storage import ArtifactStorage
from data_designer.engine.testing.utils import assert_valid_plugin
from PIL import Image, ImageDraw

from data_designer_visual_search.config import VisualSearchColumnConfig
from data_designer_visual_search.impl import VisualSearchColumnGenerator
from data_designer_visual_search.plugin import plugin
from data_designer_visual_search.tools import VisualImageWorkspace, VisualSearchToolExecutor


def test_valid_plugin() -> None:
    assert_valid_plugin(plugin)


class TestVisualSearchColumnConfig:
    def test_required_columns_include_image_and_template_references(self) -> None:
        config = VisualSearchColumnConfig(
            name="answer",
            image_column="image_path",
            prompt="Find {{ target }} in the image.",
            system_prompt="Prefer {{ style }} answers.",
            model_alias="vision",
        )

        assert config.required_columns == ["image_path", "target", "style"]

    def test_side_effect_columns_follow_options(self) -> None:
        config = VisualSearchColumnConfig(
            name="answer",
            image_column="image_path",
            prompt="Find the object.",
            model_alias="vision",
            with_trace=TraceType.LAST_MESSAGE,
            extract_reasoning_content=True,
        )

        assert config.side_effect_columns == [
            "answer__image_history",
            "answer__trace",
            "answer__reasoning_content",
        ]

    def test_base64_data_type_requires_image_format(self) -> None:
        with pytest.raises(ValueError, match="image_format is required"):
            VisualSearchColumnConfig(
                name="answer",
                image_column="image_base64",
                prompt="Find the object.",
                model_alias="vision",
                image_data_type=ModalityDataType.BASE64,
            )

    def test_base64_data_type_accepts_image_format(self) -> None:
        config = VisualSearchColumnConfig(
            name="answer",
            image_column="image_base64",
            prompt="Find the object.",
            model_alias="vision",
            image_data_type=ModalityDataType.BASE64,
            image_format=ImageFormat.PNG,
        )

        assert config.image_format == ImageFormat.PNG


class TestVisualImageWorkspace:
    def test_tools_create_branching_image_tree(self, tmp_path) -> None:
        image_path = tmp_path / "scene.png"
        _write_test_image(image_path)
        workspace = VisualImageWorkspace(source_value=str(image_path), base_path=tmp_path)

        root = workspace.open_image()
        crop = workspace.crop_image(root["image_id"], x=0, y=0, width=50, height=50, unit="percent")
        transform = workspace.transform_image(root["image_id"], flip_horizontal=True, resize_width=40)
        color_edit = workspace.edit_color(crop["image_id"], saturation=0.0, contrast=1.5)

        assert root["image_id"] == "img_0000"
        assert crop["image_id"] == "img_0001"
        assert crop["width"] == 50
        assert crop["height"] == 40
        assert transform["parent_image_id"] == root["image_id"]
        assert color_edit["parent_image_id"] == crop["image_id"]
        assert workspace.get_image_info(root["image_id"])["children_image_ids"] == ["img_0001", "img_0002"]
        assert workspace.image_context_block(color_edit["image_id"])["image_url"]["url"].startswith("data:image/png")

    def test_executor_returns_json_tool_results(self, tmp_path) -> None:
        image_path = tmp_path / "scene.png"
        _write_test_image(image_path)
        workspace = VisualImageWorkspace(source_value=str(image_path), base_path=tmp_path)
        executor = VisualSearchToolExecutor(workspace=workspace, allowed_tools=["open_image", "crop_image"])

        open_result = executor.execute("open_image", "{}")
        crop_result = executor.execute(
            "crop_image",
            json.dumps({"image_id": "img_0000", "x": 10, "y": 10, "width": 20, "height": 20}),
        )
        blocked_result = executor.execute("edit_color", json.dumps({"image_id": "img_0000"}))

        assert json.loads(open_result.content)["result"]["image_id"] == "img_0000"
        assert crop_result.image_ids == ["img_0001"]
        assert json.loads(blocked_result.content)["ok"] is False


class TestVisualSearchColumnGenerator:
    def test_generate_executes_tool_loop_and_attaches_resulting_image(self, tmp_path) -> None:
        image_path = tmp_path / "scene.png"
        _write_test_image(image_path)
        fake_model = FakeVisionModel()
        generator = _make_generator(
            VisualSearchColumnConfig(
                name="answer",
                image_column="image_path",
                prompt="Crop the red square and answer what color it is.",
                model_alias="vision",
                image_placeholder="<image>",
                with_trace=TraceType.LAST_MESSAGE,
            ),
            fake_model=fake_model,
            artifact_path=tmp_path,
        )

        result = generator.generate({"image_path": str(image_path)})

        assert result["answer"] == "The cropped object is red."
        assert [node["image_id"] for node in result["answer__image_history"]] == ["img_0000", "img_0001"]
        assert result["answer__trace"][0]["role"] == "assistant"
        assert len(fake_model.requests) == 2

        initial_request_messages = fake_model.requests[0]["messages"]
        assert initial_request_messages[1]["content"][0]["text"].startswith("<image>")
        second_request_messages = fake_model.requests[1]["messages"]
        attached_blocks = second_request_messages[-1]["content"]
        assert any(block["type"] == "image_url" for block in attached_blocks)
        assert attached_blocks[0]["text"].startswith("<image>")
        assert "tools" in fake_model.requests[0]["kwargs"]


class FakeVisionModel:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def completion(self, messages: list, **kwargs) -> ChatCompletionResponse:
        self.requests.append({"messages": [message.to_dict() for message in messages], "kwargs": kwargs})
        if len(self.requests) == 1:
            return ChatCompletionResponse(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="call_crop",
                            name="crop_image",
                            arguments_json=json.dumps(
                                {"image_id": "img_0000", "x": 0, "y": 0, "width": 50, "height": 50, "unit": "percent"}
                            ),
                        )
                    ]
                )
            )
        return ChatCompletionResponse(message=AssistantMessage(content="The cropped object is red."))


def _make_generator(
    config: VisualSearchColumnConfig,
    *,
    fake_model: FakeVisionModel,
    artifact_path,
) -> VisualSearchColumnGenerator:
    generator = VisualSearchColumnGenerator.__new__(VisualSearchColumnGenerator)
    generator._config = config
    generator._resource_provider = SimpleNamespace(
        artifact_storage=ArtifactStorage(artifact_path=artifact_path),
        run_config=RunConfig(),
    )
    generator.__dict__["model"] = fake_model
    return generator


def _write_test_image(path) -> None:
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 50, 40), fill="red")
    draw.rectangle((50, 40, 99, 79), fill="blue")
    image.save(path)
