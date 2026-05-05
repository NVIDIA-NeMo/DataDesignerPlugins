# Usage

The template plugin is intentionally small so plugin authors can inspect the
full package quickly. Its generator reads a source text column and writes a new
column using the configured transform.

```python
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource

builder = DataDesignerConfigBuilder()
builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
builder.add_column(
    name="name_upper",
    column_type="text-transform",
    source_column="name",
    transform="upper",
)
```

The package tests cover the public config object, generator behavior, and a
preview flow through Data Designer.
