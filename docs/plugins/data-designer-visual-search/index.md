# data-designer-visual-search

`data-designer-visual-search` adds a `visual-search` column type for
image-grounded visual search workflows. It is intended for cases where a VLM
needs to inspect an image, crop into regions, transform the view, adjust color,
and then continue reasoning over the resulting image.

The plugin owns the extra plumbing that ordinary model tool calling does not
handle: each local image operation returns an `image_id`, the new image is held
in memory, and the generated image is attached back into the next model turn as
multimodal context.

## What It Provides

- A `VisualSearchColumnConfig` registered as column type `visual-search`.
- A row-scoped in-memory image workspace.
- Local tools for opening images, listing image IDs, inspecting image metadata,
  cropping, transforming, and editing color.
- Tree-shaped image history, so the model can branch from any previous
  `image_id` instead of following a single linear edit chain.
- A default side-effect column named `{column_name}__image_history` that records
  image IDs, parent IDs, child IDs, operations, dimensions, and operation
  metadata.
- Optional model trace and reasoning-content side-effect columns that match the
  conventions used by Data Designer LLM columns.

## Column Interface

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Output column name. |
| `column_type` | Yes | Must be `visual-search`. |
| `image_column` | Yes | Existing column containing a local image path, URL, base64 image, or image data URI. |
| `prompt` | Yes | Jinja2 prompt template for the visual search task. |
| `model_alias` | Yes | Alias of a vision-capable chat model in the Data Designer config. |
| `system_prompt` | No | Optional Jinja2 system prompt appended to the built-in visual search instructions. |
| `image_data_type` | No | Optional explicit image data type, such as `url` or `base64`. Leave unset for auto-detection. |
| `image_format` | Conditional | Required when `image_data_type` is explicitly `base64`. |
| `image_placeholder` | No | Optional text token to include next to every image attachment for endpoints that require one. |
| `max_tool_call_turns` | No | Maximum tool-calling turns per row. Defaults to `6`. |
| `allowed_tools` | No | Optional allowlist of built-in visual tools. Defaults to all tools. |
| `attach_images_after_tool_calls` | No | Whether to attach tool-created images into the next model turn. Defaults to `True`. |
| `include_image_history` | No | Whether to write `{name}__image_history`. Defaults to `True`. |
| `with_trace` | No | Optional trace capture mode. Defaults to `none`. |
| `extract_reasoning_content` | No | Whether to write `{name}__reasoning_content`. Defaults to `False`. |
| `use_default_system_prompt` | No | Whether to prepend built-in image-tool instructions. Defaults to `True`. |

## Built-In Tools

| Tool | Purpose |
| --- | --- |
| `open_image` | Opens the configured row image and returns the root `image_id`. |
| `get_image_info` | Returns dimensions, parent ID, children IDs, operation name, and metadata for an `image_id`. |
| `list_images` | Lists every image currently held in the row workspace. |
| `crop_image` | Crops an existing image by pixel or percent coordinates and returns a new `image_id`. |
| `transform_image` | Rotates, flips, or resizes an existing image and returns a new `image_id`. |
| `edit_color` | Adjusts brightness, contrast, saturation, sharpness, grayscale, or inversion and returns a new `image_id`. |

Tool results are ordinary tool messages containing JSON metadata. When a tool
creates an image, the plugin also attaches that image to the next user turn so
the model can inspect it visually.

## Image History

Every image node has stable metadata:

```json
{
  "image_id": "img_0001",
  "parent_image_id": "img_0000",
  "children_image_ids": [],
  "operation": "crop_image",
  "width": 512,
  "height": 384,
  "metadata": {
    "box": {"left": 0, "top": 0, "right": 512, "bottom": 384},
    "unit": "pixels"
  }
}
```

Because the model controls the `image_id` argument, it can crop from the root
image, transform that crop, rewind to the root, and crop a different region.
The workspace keeps the whole tree for the duration of that row.

## When To Use It

Use `visual-search` when the model needs iterative visual operations before it
can answer reliably. Good examples include reading small labels, comparing
regions, checking color after contrast adjustment, or zooming into a specific
part of a larger image.

For a single prompt over an image with no iterative image manipulation, a
standard Data Designer LLM column with multimodal context may be simpler.
