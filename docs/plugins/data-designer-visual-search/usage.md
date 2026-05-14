# Usage

This example starts with a dataframe column containing image paths and adds a
`visual-search` column. The model can call image tools while answering the
prompt, and the plugin will pass each resulting crop or edited image back to the
model automatically.

```python
import pandas as pd

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.interface.data_designer import DataDesigner

seed_df = pd.DataFrame(
    {
        "image_path": ["/path/to/store-shelf.png"],
        "target": ["the nutrition label on the cereal box"],
    }
)

provider = ModelProvider(
    name="nvidia",
    endpoint="https://integrate.api.nvidia.com/v1",
    api_key="NVIDIA_API_KEY",
    provider_type="openai",
)

vision_model = ModelConfig(
    alias="vision",
    model="qwen/qwen3.5-122b-a10b",
    provider="nvidia",
    inference_parameters=ChatCompletionInferenceParams(
        temperature=0,
        max_tokens=512,
        timeout=60,
    ),
)

builder = DataDesignerConfigBuilder(model_configs=[vision_model])
builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
builder.add_column(
    name="visual_answer",
    column_type="visual-search",
    image_column="image_path",
    prompt=(
        "Find {{ target }}. Use crop_image or edit_color if that helps. "
        "Return the text you can read and explain which image_id you used."
    ),
    model_alias="vision",
    max_tool_call_turns=4,
)

result = DataDesigner(
    artifact_path="artifacts",
    model_providers=[provider],
).preview(builder, num_records=1)
```

The generated dataset includes:

- `visual_answer`: the model's final answer.
- `visual_answer__image_history`: the image operation tree produced while
  answering the row.

## Restricting Tools

Use `allowed_tools` when you want the model to perform only a narrower set of
operations:

```python
builder.add_column(
    name="crop_only_answer",
    column_type="visual-search",
    image_column="image_path",
    prompt="Crop the upper-right quadrant and describe the dominant color.",
    model_alias="vision",
    allowed_tools=["open_image", "get_image_info", "crop_image"],
    max_tool_call_turns=2,
)
```

## Endpoint Image Tokens

Most OpenAI-compatible multimodal endpoints accept image content blocks directly.
Some model servers also require a model-specific image token in the text for
each attached image. Set `image_placeholder` for those endpoints:

```python
builder.add_column(
    name="answer",
    column_type="visual-search",
    image_column="image_path",
    prompt="Inspect the attached image and answer the question.",
    model_alias="vision",
    image_placeholder="<image>",
)
```

The plugin prepends the placeholder to the initial image turn and to every later
turn that attaches a tool-created image.

## Capturing Trace Output

The column supports the same trace side-effect pattern as other LLM-backed Data
Designer columns:

```python
from data_designer.config.utils.trace_type import TraceType

builder.add_column(
    name="answer_with_trace",
    column_type="visual-search",
    image_column="image_path",
    prompt="Zoom into the serial number and read it.",
    model_alias="vision",
    with_trace=TraceType.ALL_MESSAGES,
    extract_reasoning_content=True,
)
```

This adds `answer_with_trace__trace` and
`answer_with_trace__reasoning_content` when the selected model provides
reasoning content.
