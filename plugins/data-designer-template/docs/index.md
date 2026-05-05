# data-designer-template

The template plugin is the reference implementation for a simple Data Designer
column generator. It adds a `text-transform` column type that writes an output
column by transforming text from an existing source column.

## Installation

```bash
uv add data-designer data-designer-template
```

## Column type

Use the `text-transform` column type when a dataset needs a derived text column
using one of the supported string transforms.

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Output column name. |
| `source_column` | Yes | Existing text column to transform. |
| `transform` | No | One of `upper`, `lower`, or `title`; defaults to `upper`. |

## Implementation notes

The package keeps plugin registration in `plugin.py`, configuration in
`config.py`, and generation logic in `impl.py`. New plugins should follow the
same separation unless their behavior needs a different shape.
