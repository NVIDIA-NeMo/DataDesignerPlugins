# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from data_designer.config.models import ModalityDataType
from data_designer.config.utils.image_helpers import (
    ImageFormat,
    decode_base64_image,
    detect_image_format,
    extract_base64_from_data_uri,
    is_base64_image,
    is_image_url,
)
from PIL import Image, ImageEnhance, ImageOps

DEFAULT_IMAGE_FORMAT = ImageFormat.PNG


@dataclass
class ImageNode:
    """Image plus lineage metadata stored in a visual-search workspace."""

    image_id: str
    image: Image.Image
    parent_image_id: str | None
    operation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, children_image_ids: list[str] | None = None) -> dict[str, Any]:
        """Return JSON-serializable metadata for this image node."""
        return {
            "image_id": self.image_id,
            "parent_image_id": self.parent_image_id,
            "children_image_ids": children_image_ids or [],
            "operation": self.operation,
            "width": self.image.width,
            "height": self.image.height,
            "metadata": self.metadata,
        }


@dataclass
class VisualToolExecution:
    """Result of executing a local visual tool call."""

    content: str
    image_ids: list[str] = field(default_factory=list)
    is_error: bool = False


class VisualImageWorkspace:
    """In-memory image tree for visual search tool calls.

    The workspace keeps every intermediate image addressable by ID. Tools can
    operate on any prior image ID, so a model can branch from an earlier crop or
    transform instead of being forced into a linear edit history.
    """

    def __init__(
        self,
        *,
        source_value: Any,
        base_path: Path | None = None,
        image_data_type: ModalityDataType | None = None,
        image_format: ImageFormat | None = None,
    ) -> None:
        self._source_value = source_value
        self._base_path = base_path
        self._image_data_type = image_data_type
        self._image_format = image_format
        self._nodes: dict[str, ImageNode] = {}
        self._next_image_index = 0
        self._root_image_id: str | None = None

    @property
    def root_image_id(self) -> str | None:
        """Return the root image ID after the source image has been opened."""
        return self._root_image_id

    def open_image(self, path: str | None = None) -> dict[str, Any]:
        """Open the configured source image or an explicitly supplied image reference.

        Args:
            path: Optional local path, URL, base64 string, or data URI. When omitted,
                the configured source image for this row is opened.

        Returns:
            Metadata for the opened image node.
        """
        if path is None and self._root_image_id is not None:
            return self.get_image_info(self._root_image_id)

        source = self._source_value if path is None else path
        image = self._load_image(source)
        node = self._create_node(
            image=image,
            parent_image_id=None,
            operation="open_image",
            metadata={"source": _summarize_source(source)},
        )
        if path is None:
            self._root_image_id = node.image_id
        return self.get_image_info(node.image_id)

    def get_image_info(self, image_id: str) -> dict[str, Any]:
        """Return metadata for an image ID."""
        node = self._get_node(image_id)
        return node.to_dict(children_image_ids=self._children_for(image_id))

    def list_images(self) -> dict[str, Any]:
        """Return the current image tree metadata."""
        return {
            "root_image_id": self._root_image_id,
            "images": [self.get_image_info(image_id) for image_id in self._nodes],
        }

    def crop_image(
        self,
        image_id: str,
        x: float,
        y: float,
        width: float,
        height: float,
        unit: str = "pixels",
    ) -> dict[str, Any]:
        """Crop an image by pixels or percentages and return the new image metadata."""
        node = self._get_node(image_id)
        left, top, right, bottom = _resolve_crop_box(node.image.size, x, y, width, height, unit)
        cropped = node.image.crop((left, top, right, bottom))
        child = self._create_node(
            image=cropped,
            parent_image_id=image_id,
            operation="crop_image",
            metadata={"box": {"left": left, "top": top, "right": right, "bottom": bottom}, "unit": "pixels"},
        )
        return self.get_image_info(child.image_id)

    def transform_image(
        self,
        image_id: str,
        rotate_degrees: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        resize_width: int | None = None,
        resize_height: int | None = None,
        preserve_aspect_ratio: bool = True,
    ) -> dict[str, Any]:
        """Rotate, flip, and resize an image, returning a new image ID."""
        node = self._get_node(image_id)
        image = node.image.copy()

        if rotate_degrees:
            image = image.rotate(-rotate_degrees, expand=True)
        if flip_horizontal:
            image = ImageOps.mirror(image)
        if flip_vertical:
            image = ImageOps.flip(image)
        if resize_width is not None or resize_height is not None:
            image = _resize_image(image, resize_width, resize_height, preserve_aspect_ratio)

        child = self._create_node(
            image=image,
            parent_image_id=image_id,
            operation="transform_image",
            metadata={
                "rotate_degrees": rotate_degrees,
                "flip_horizontal": flip_horizontal,
                "flip_vertical": flip_vertical,
                "resize_width": resize_width,
                "resize_height": resize_height,
                "preserve_aspect_ratio": preserve_aspect_ratio,
            },
        )
        return self.get_image_info(child.image_id)

    def edit_color(
        self,
        image_id: str,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0,
        grayscale: bool = False,
        invert: bool = False,
    ) -> dict[str, Any]:
        """Adjust color properties and return a new image ID."""
        node = self._get_node(image_id)
        image = node.image.copy()

        if grayscale:
            image = ImageOps.grayscale(image).convert("RGB")
        if invert:
            image = _invert_image(image)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        image = ImageEnhance.Sharpness(image).enhance(sharpness)

        child = self._create_node(
            image=image,
            parent_image_id=image_id,
            operation="edit_color",
            metadata={
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "sharpness": sharpness,
                "grayscale": grayscale,
                "invert": invert,
            },
        )
        return self.get_image_info(child.image_id)

    def image_context_block(self, image_id: str) -> dict[str, Any]:
        """Return an OpenAI-compatible image content block for an image ID."""
        data_uri = self.image_data_uri(image_id)
        return {"type": "image_url", "image_url": {"url": data_uri}}

    def image_data_uri(self, image_id: str) -> str:
        """Return an image as a PNG data URI."""
        base64_data = self.image_base64(image_id)
        return f"data:image/{DEFAULT_IMAGE_FORMAT.value};base64,{base64_data}"

    def image_base64(self, image_id: str) -> str:
        """Return an image encoded as base64 PNG."""
        node = self._get_node(image_id)
        buffer = io.BytesIO()
        image = _normalize_image_for_png(node.image)
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def history(self) -> list[dict[str, Any]]:
        """Return JSON-serializable operation history."""
        return [self.get_image_info(image_id) for image_id in self._nodes]

    def _create_node(
        self,
        *,
        image: Image.Image,
        parent_image_id: str | None,
        operation: str,
        metadata: dict[str, Any],
    ) -> ImageNode:
        image_id = f"img_{self._next_image_index:04d}"
        self._next_image_index += 1
        node = ImageNode(
            image_id=image_id,
            image=_normalize_loaded_image(image),
            parent_image_id=parent_image_id,
            operation=operation,
            metadata=metadata,
        )
        self._nodes[image_id] = node
        return node

    def _get_node(self, image_id: str) -> ImageNode:
        try:
            return self._nodes[image_id]
        except KeyError:
            known = ", ".join(self._nodes) or "(none)"
            raise ValueError(f"Unknown image_id {image_id!r}. Known image IDs: {known}")

    def _children_for(self, image_id: str) -> list[str]:
        return [node.image_id for node in self._nodes.values() if node.parent_image_id == image_id]

    def _load_image(self, source: Any) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.copy()
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source))
        if isinstance(source, str):
            return Image.open(io.BytesIO(self._load_image_bytes_from_string(source)))
        raise TypeError(f"Unsupported image source type: {type(source).__name__}")

    def _load_image_bytes_from_string(self, source: str) -> bytes:
        if self._image_data_type == ModalityDataType.URL or is_image_url(source):
            response = requests.get(source, timeout=60)
            response.raise_for_status()
            return response.content

        if (
            self._image_data_type == ModalityDataType.BASE64
            or source.startswith("data:image/")
            or is_base64_image(source)
        ):
            return decode_base64_image(source)

        path = Path(source)
        if not path.is_absolute() and self._base_path is not None:
            candidate = self._base_path / path
            if candidate.exists():
                path = candidate
        if not path.is_absolute() and not path.exists():
            path = Path.cwd() / source
        if path.exists():
            return path.read_bytes()

        try:
            return decode_base64_image(extract_base64_from_data_uri(source))
        except ValueError as exc:
            raise ValueError(f"Could not load image source {source!r} as a path, URL, or base64 image") from exc


class VisualSearchToolExecutor:
    """Executes the built-in visual-search tools for one row."""

    def __init__(
        self,
        *,
        workspace: VisualImageWorkspace,
        allowed_tools: list[str] | None = None,
        allow_external_open: bool = False,
    ) -> None:
        self._workspace = workspace
        self._allowed_tools = set(allowed_tools or TOOL_FUNCTIONS)
        self._allow_external_open = allow_external_open

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for the allowed tools."""
        return [schema for schema in VISUAL_SEARCH_TOOL_SCHEMAS if schema["function"]["name"] in self._allowed_tools]

    def execute(self, tool_name: str, arguments_json: str) -> VisualToolExecution:
        """Execute a tool call and return a tool-message-ready result."""
        if tool_name not in self._allowed_tools:
            return _error_result(tool_name, f"Tool {tool_name!r} is not allowed for this column.")
        if tool_name not in TOOL_FUNCTIONS:
            return _error_result(tool_name, f"Unknown visual-search tool {tool_name!r}.")

        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must decode to a JSON object.")
            if tool_name == "open_image" and arguments.get("path") and not self._allow_external_open:
                raise ValueError("open_image path is managed by the visual-search column; omit path for row input.")
            payload = TOOL_FUNCTIONS[tool_name](self._workspace, **arguments)
            return _success_result(tool_name, payload)
        except Exception as exc:
            return _error_result(tool_name, str(exc))


def _success_result(tool_name: str, payload: dict[str, Any]) -> VisualToolExecution:
    image_ids = [payload["image_id"]] if isinstance(payload.get("image_id"), str) else []
    return VisualToolExecution(
        content=json.dumps({"ok": True, "tool": tool_name, "result": payload}, sort_keys=True),
        image_ids=image_ids,
    )


def _error_result(tool_name: str, message: str) -> VisualToolExecution:
    return VisualToolExecution(
        content=json.dumps({"ok": False, "tool": tool_name, "error": message}, sort_keys=True),
        is_error=True,
    )


def _open_image(workspace: VisualImageWorkspace, path: str | None = None) -> dict[str, Any]:
    return workspace.open_image(path=path)


def _get_image_info(workspace: VisualImageWorkspace, image_id: str) -> dict[str, Any]:
    return workspace.get_image_info(image_id)


def _list_images(workspace: VisualImageWorkspace) -> dict[str, Any]:
    return workspace.list_images()


def _crop_image(
    workspace: VisualImageWorkspace,
    image_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    unit: str = "pixels",
) -> dict[str, Any]:
    return workspace.crop_image(image_id=image_id, x=x, y=y, width=width, height=height, unit=unit)


def _transform_image(
    workspace: VisualImageWorkspace,
    image_id: str,
    rotate_degrees: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    resize_width: int | None = None,
    resize_height: int | None = None,
    preserve_aspect_ratio: bool = True,
) -> dict[str, Any]:
    return workspace.transform_image(
        image_id=image_id,
        rotate_degrees=rotate_degrees,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        resize_width=resize_width,
        resize_height=resize_height,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )


def _edit_color(
    workspace: VisualImageWorkspace,
    image_id: str,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
    grayscale: bool = False,
    invert: bool = False,
) -> dict[str, Any]:
    return workspace.edit_color(
        image_id=image_id,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        sharpness=sharpness,
        grayscale=grayscale,
        invert=invert,
    )


TOOL_FUNCTIONS = {
    "open_image": _open_image,
    "get_image_info": _get_image_info,
    "list_images": _list_images,
    "crop_image": _crop_image,
    "transform_image": _transform_image,
    "edit_color": _edit_color,
}

VISUAL_SEARCH_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "open_image",
            "description": (
                "Open the configured source image for this row and return its image_id. "
                "If called repeatedly without a path, returns the existing root image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional image path, URL, or base64 data. Usually omit this.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_image_info",
            "description": "Return dimensions, parent, children, and operation metadata for an image_id.",
            "parameters": {
                "type": "object",
                "properties": {"image_id": {"type": "string", "description": "Image ID to inspect."}},
                "required": ["image_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_images",
            "description": "List all image IDs currently in memory with parent/child relationships.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crop_image",
            "description": (
                "Create a crop from any existing image_id. Use unit='percent' for approximate visual regions "
                "or unit='pixels' for exact coordinates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "x": {"type": "number", "description": "Left coordinate in pixels or percent."},
                    "y": {"type": "number", "description": "Top coordinate in pixels or percent."},
                    "width": {"type": "number", "description": "Crop width in pixels or percent."},
                    "height": {"type": "number", "description": "Crop height in pixels or percent."},
                    "unit": {"type": "string", "enum": ["pixels", "percent"], "default": "pixels"},
                },
                "required": ["image_id", "x", "y", "width", "height"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transform_image",
            "description": "Rotate, flip, and/or resize an existing image_id and return a new image_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "rotate_degrees": {"type": "number", "default": 0},
                    "flip_horizontal": {"type": "boolean", "default": False},
                    "flip_vertical": {"type": "boolean", "default": False},
                    "resize_width": {"type": "integer", "minimum": 1},
                    "resize_height": {"type": "integer", "minimum": 1},
                    "preserve_aspect_ratio": {"type": "boolean", "default": True},
                },
                "required": ["image_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_color",
            "description": (
                "Adjust brightness, contrast, saturation, sharpness, grayscale, or inversion for an image_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "brightness": {"type": "number", "default": 1.0, "minimum": 0},
                    "contrast": {"type": "number", "default": 1.0, "minimum": 0},
                    "saturation": {"type": "number", "default": 1.0, "minimum": 0},
                    "sharpness": {"type": "number", "default": 1.0, "minimum": 0},
                    "grayscale": {"type": "boolean", "default": False},
                    "invert": {"type": "boolean", "default": False},
                },
                "required": ["image_id"],
                "additionalProperties": False,
            },
        },
    },
]


def _resolve_crop_box(
    image_size: tuple[int, int],
    x: float,
    y: float,
    width: float,
    height: float,
    unit: str,
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Crop width and height must be positive.")
    if unit == "percent":
        left = round(image_width * (x / 100.0))
        top = round(image_height * (y / 100.0))
        right = round(image_width * ((x + width) / 100.0))
        bottom = round(image_height * ((y + height) / 100.0))
    elif unit == "pixels":
        left = round(x)
        top = round(y)
        right = round(x + width)
        bottom = round(y + height)
    else:
        raise ValueError("unit must be either 'pixels' or 'percent'.")

    left = max(0, min(image_width - 1, left))
    top = max(0, min(image_height - 1, top))
    right = max(left + 1, min(image_width, right))
    bottom = max(top + 1, min(image_height, bottom))
    return left, top, right, bottom


def _resize_image(
    image: Image.Image,
    resize_width: int | None,
    resize_height: int | None,
    preserve_aspect_ratio: bool,
) -> Image.Image:
    if resize_width is not None and resize_width < 1:
        raise ValueError("resize_width must be at least 1.")
    if resize_height is not None and resize_height < 1:
        raise ValueError("resize_height must be at least 1.")

    if resize_width is None and resize_height is None:
        return image
    if preserve_aspect_ratio:
        if resize_width is None:
            ratio = resize_height / image.height
            resize_width = max(1, round(image.width * ratio))
        elif resize_height is None:
            ratio = resize_width / image.width
            resize_height = max(1, round(image.height * ratio))
        resized = image.copy()
        resized.thumbnail((resize_width, resize_height), Image.Resampling.LANCZOS)
        return resized

    target_width = resize_width or image.width
    target_height = resize_height or image.height
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _invert_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        red, green, blue, alpha = image.split()
        inverted = ImageOps.invert(Image.merge("RGB", (red, green, blue)))
        inverted.putalpha(alpha)
        return inverted
    return ImageOps.invert(image.convert("RGB"))


def _normalize_loaded_image(image: Image.Image) -> Image.Image:
    image.load()
    if image.mode in {"RGBA", "RGB", "L"}:
        return image.copy()
    return image.convert("RGBA" if "A" in image.getbands() else "RGB")


def _normalize_image_for_png(image: Image.Image) -> Image.Image:
    if image.mode in {"RGB", "RGBA", "L"}:
        return image
    return image.convert("RGBA" if "A" in image.getbands() else "RGB")


def _summarize_source(source: Any) -> str:
    if not isinstance(source, str):
        return type(source).__name__
    if source.startswith("data:image/") or is_base64_image(source):
        try:
            image_format = detect_image_format(decode_base64_image(source))
            return f"{image_format.value} base64 image"
        except ValueError:
            return "base64 image"
    return source
