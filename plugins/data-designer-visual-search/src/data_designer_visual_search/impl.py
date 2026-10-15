# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.config.utils.constants import REASONING_CONTENT_COLUMN_POSTFIX, TRACE_COLUMN_POSTFIX
from data_designer.config.utils.trace_type import TraceType
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel, GenerationStrategy
from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.utils import deserialize_json_values

from data_designer_visual_search.config import VisualSearchColumnConfig
from data_designer_visual_search.tools import VisualImageWorkspace, VisualSearchToolExecutor

if TYPE_CHECKING:
    from typing import Any

    from data_designer.engine.models.clients.types import ChatCompletionResponse, ToolCall

DEFAULT_VISUAL_SEARCH_SYSTEM_PROMPT = """\
You are a visual search agent working with an in-memory image tree.
Use the available image tools when cropping, transforming, or color-adjusting the image would help answer.
Every image has an image_id. Tool calls may operate on any previous image_id, so you can branch from earlier images.
After a tool creates an image, that image will be attached in the next user turn with its image_id.
When you have enough evidence, stop calling tools and answer the user's prompt directly.
"""

TOOL_BUDGET_EXHAUSTED_MESSAGE = (
    "Tool call budget exhausted. Use the images and tool results already shown, then provide the final answer."
)


class VisualSearchColumnGenerator(
    WithJinja2UserTemplateRendering,
    ColumnGeneratorWithModel[VisualSearchColumnConfig],
):
    """Run a vision model with built-in image-operation tools."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        """Generate a visual-search answer for one row."""
        deserialized_record = deserialize_json_values(data)
        workspace = self._create_workspace(deserialized_record)
        executor = VisualSearchToolExecutor(workspace=workspace, allowed_tools=self.config.allowed_tools)
        root = workspace.open_image()

        messages = self._build_initial_messages(deserialized_record, workspace, root["image_id"])
        final_text, trace = self._run_tool_loop(messages, workspace, executor)

        data[self.config.name] = final_text
        if self.config.include_image_history:
            data[f"{self.config.name}__image_history"] = workspace.history()
        if self.config.with_trace == TraceType.ALL_MESSAGES:
            data[f"{self.config.name}{TRACE_COLUMN_POSTFIX}"] = [message.to_dict() for message in trace]
        elif self.config.with_trace == TraceType.LAST_MESSAGE:
            last_assistant = next((message for message in reversed(trace) if message.role == "assistant"), None)
            data[f"{self.config.name}{TRACE_COLUMN_POSTFIX}"] = (
                [last_assistant.to_dict()] if last_assistant is not None else []
            )
        if self.config.extract_reasoning_content:
            data[f"{self.config.name}{REASONING_CONTENT_COLUMN_POSTFIX}"] = self._extract_reasoning_content(trace)
        return data

    def _create_workspace(self, record: dict[str, Any]) -> VisualImageWorkspace:
        return VisualImageWorkspace(
            source_value=record[self.config.image_column],
            base_path=self.base_dataset_path,
            image_data_type=self.config.image_data_type,
            image_format=self.config.image_format,
        )

    def _build_initial_messages(
        self,
        record: dict[str, Any],
        workspace: VisualImageWorkspace,
        root_image_id: str,
    ) -> list[ChatMessage]:
        prompt = self._render_template(self.config.prompt, record)
        prompt = (
            f"{prompt}\n\n"
            f"The source image is attached and is available in the tool workspace as image_id {root_image_id!r}. "
            "You may call open_image() to retrieve the same root image_id, or operate on this image_id directly."
        )
        if self.config.image_placeholder:
            prompt = f"{self.config.image_placeholder}\n{prompt}"

        messages: list[ChatMessage] = []
        system_prompt = self._build_system_prompt(record)
        if system_prompt:
            messages.append(ChatMessage.as_system(system_prompt))
        messages.append(
            ChatMessage.as_user([{"type": "text", "text": prompt}, workspace.image_context_block(root_image_id)])
        )
        return messages

    def _build_system_prompt(self, record: dict[str, Any]) -> str | None:
        system_parts: list[str] = []
        if self.config.use_default_system_prompt:
            system_parts.append(DEFAULT_VISUAL_SEARCH_SYSTEM_PROMPT)
        if self.config.system_prompt:
            system_parts.append(self._render_template(self.config.system_prompt, record))
        return "\n\n".join(part for part in system_parts if part).strip() or None

    def _render_template(self, template: str, record: dict[str, Any]) -> str:
        jinja_render_env = self._create_render_environment(dataset_variables=list(record.keys()))
        jinja_render_env.validate_template(template)
        return jinja_render_env.render_template(template, record, skip_template_validation=True)

    def _run_tool_loop(
        self,
        messages: list[ChatMessage],
        workspace: VisualImageWorkspace,
        executor: VisualSearchToolExecutor,
    ) -> tuple[str, list[ChatMessage]]:
        tool_call_turns = 0
        tools_enabled = True
        tool_schemas = executor.get_tool_schemas()

        while True:
            completion_response = self._complete(messages, tool_schemas if tools_enabled else None)
            tool_calls = completion_response.message.tool_calls
            if tool_calls and tools_enabled:
                tool_call_turns += 1
                messages.append(_assistant_tool_message(completion_response))

                if tool_call_turns > self.config.max_tool_call_turns:
                    messages.extend(ChatMessage.as_tool(TOOL_BUDGET_EXHAUSTED_MESSAGE, call.id) for call in tool_calls)
                    messages.append(ChatMessage.as_user(TOOL_BUDGET_EXHAUSTED_MESSAGE))
                    tools_enabled = False
                    continue

                image_ids = self._execute_tool_calls(messages, executor, tool_calls)
                if image_ids and self.config.attach_images_after_tool_calls:
                    messages.append(
                        ChatMessage.as_user(
                            _tool_image_context_blocks(
                                workspace,
                                image_ids,
                                image_placeholder=self.config.image_placeholder,
                            )
                        )
                    )
                continue

            response_text = (completion_response.message.content or "").strip()
            messages.append(
                ChatMessage.as_assistant(
                    content=response_text,
                    reasoning_content=completion_response.message.reasoning_content or None,
                )
            )
            return response_text, messages

    def _complete(
        self, messages: list[ChatMessage], tool_schemas: list[dict[str, Any]] | None
    ) -> ChatCompletionResponse:
        completion_kwargs = {"purpose": f"running visual search for column {self.config.name!r}"}
        if tool_schemas:
            completion_kwargs["tools"] = tool_schemas
        return self.model.completion(messages, **completion_kwargs)

    def _execute_tool_calls(
        self,
        messages: list[ChatMessage],
        executor: VisualSearchToolExecutor,
        tool_calls: list[ToolCall],
    ) -> list[str]:
        image_ids: list[str] = []
        for tool_call in tool_calls:
            result = executor.execute(tool_call.name, tool_call.arguments_json)
            messages.append(ChatMessage.as_tool(content=result.content, tool_call_id=tool_call.id))
            image_ids.extend(result.image_ids)
        return image_ids

    def _extract_reasoning_content(self, trace: list[ChatMessage]) -> str | None:
        reasoning_value: str | None = None
        for message in reversed(trace):
            if message.role == "assistant":
                reasoning_value = message.reasoning_content
                break
        return reasoning_value.strip() or None if reasoning_value is not None else None


def _assistant_tool_message(completion_response: ChatCompletionResponse) -> ChatMessage:
    tool_calls = [
        {
            "id": tool_call.id,
            "type": "function",
            "function": {"name": tool_call.name, "arguments": tool_call.arguments_json},
        }
        for tool_call in completion_response.message.tool_calls
    ]
    return ChatMessage.as_assistant(
        content=(completion_response.message.content or "").strip(),
        reasoning_content=completion_response.message.reasoning_content or None,
        tool_calls=tool_calls,
    )


def _tool_image_context_blocks(
    workspace: VisualImageWorkspace,
    image_ids: list[str],
    *,
    image_placeholder: str | None = None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for image_id in image_ids:
        info = workspace.get_image_info(image_id)
        text = (
            "Tool result image attached: "
            f"image_id={image_id}, parent_image_id={info['parent_image_id']}, "
            f"operation={info['operation']}, size={info['width']}x{info['height']}."
        )
        if image_placeholder:
            text = f"{image_placeholder}\n{text}"
        blocks.append(
            {
                "type": "text",
                "text": text,
            }
        )
        blocks.append(workspace.image_context_block(image_id))
    return blocks
