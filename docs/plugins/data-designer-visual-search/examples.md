# Practical Examples

## Branch From an Earlier Crop

The image workspace is tree-shaped. A model can create one crop, inspect it,
then operate on the original image again:

1. `open_image()` returns `img_0000`.
2. `crop_image(image_id="img_0000", x=0, y=0, width=50, height=50, unit="percent")`
   returns `img_0001`.
3. `edit_color(image_id="img_0001", contrast=1.5)` returns `img_0002`.
4. `crop_image(image_id="img_0000", x=50, y=50, width=50, height=50, unit="percent")`
   returns `img_0003`.

The resulting history preserves both branches:

```text
img_0000 open_image
|-- img_0001 crop_image
|   `-- img_0002 edit_color
`-- img_0003 crop_image
```

This is useful when the model needs to compare multiple areas or recover from a
crop that turned out to be unhelpful.

## Read Small Text

```python
builder.add_column(
    name="label_text",
    column_type="visual-search",
    image_column="product_photo",
    prompt=(
        "Find the ingredients label. Crop tightly around it, increase contrast "
        "if needed, and return the text you can read."
    ),
    model_alias="vision",
    max_tool_call_turns=5,
)
```

Expected model behavior:

- Inspect the original image.
- Crop the label region.
- Optionally increase contrast or convert to grayscale.
- Answer using the attached edited crop.

## Compare Two Regions

```python
builder.add_column(
    name="comparison",
    column_type="visual-search",
    image_column="shelf_image",
    prompt=(
        "Compare the price tags on the left and right sides of the shelf. "
        "Use separate crops and report which price is lower."
    ),
    model_alias="vision",
    max_tool_call_turns=6,
)
```

The model can crop the left tag from `img_0000`, crop the right tag from
`img_0000`, inspect both resulting IDs, and answer from the evidence.

## Data URI Input

The `image_column` can contain base64 data or a full data URI instead of a file
path:

```python
builder.add_column(
    name="base64_answer",
    column_type="visual-search",
    image_column="image_data_uri",
    prompt="Crop the center of the image and describe what is visible.",
    model_alias="vision",
)
```

If values are raw base64 and the format cannot be detected reliably, set
`image_data_type="base64"` and `image_format="png"` or another supported image
format.
