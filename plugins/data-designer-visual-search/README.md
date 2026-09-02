# data-designer-visual-search

Data Designer plugin for VLM-driven visual search over image columns, with
local image crop, transform, and color-edit tools.

The `visual-search` column runs a vision-capable chat model with built-in
image-operation tools:

- `open_image`
- `get_image_info`
- `list_images`
- `crop_image`
- `transform_image`
- `edit_color`

Each operation returns an `image_id`. The column keeps intermediate images in
memory and re-attaches tool-produced images to the following model turn, so the
model can inspect a crop or transformed image before deciding what to do next.
Because IDs remain addressable, the model can branch from an earlier image
rather than being forced through a linear edit chain.

## Installation

```bash
pip install data-designer-visual-search
```

## Usage

Once installed, the `visual-search` column type is automatically discovered by
[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).

```python
import pandas as pd
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.interface.data_designer import DataDesigner

seed_df = pd.DataFrame({"image_path": ["/path/to/scene.png"]})

builder = DataDesignerConfigBuilder()
builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
builder.add_column(
    name="visual_answer",
    column_type="visual-search",
    image_column="image_path",
    prompt="Find the red object. Crop or transform the image if that helps.",
    model_alias="nvidia-vision",
    # Optional: set a model-specific image token here if your endpoint requires
    # one in the text for every attached image.
    # image_placeholder="<image>",
)

result = DataDesigner(artifact_path="artifacts").preview(builder, num_records=1)
```

The main output column contains the model's final answer. By default the plugin
also writes `{column_name}__image_history`, a compact tree of image IDs, parent
IDs, operations, and dimensions.

See `docs/` for the full interface reference and practical examples.
