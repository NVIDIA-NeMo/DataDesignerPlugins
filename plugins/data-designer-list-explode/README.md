# data-designer-list-explode

Data Designer plugin that explodes columns containing structured lists into one
row per element — similar to `pandas.DataFrame.explode()`.

## Installation

```bash
pip install data-designer-list-explode
```

## Usage

Once installed, the `list-explode` column type is automatically discovered by
[NeMo Data Designer](https://github.com/NVIDIA/NeMo-Data-Designer).

```yaml
columns:
  tag:
    column_type: list-explode
    source_column: tags
    drop_empty: false        # optional, default false
```

Given a DataFrame like:

| id | tags          |
|----|---------------|
| 1  | ["a", "b"]    |
| 2  | ["c"]         |

The `list-explode` column produces:

| id | tags | tag |
|----|------|-----|
| 1  | a    | a   |
| 1  | b    | b   |
| 2  | c    | c   |

For the full plugin authoring guide, see the
[main repository docs](https://gitlab-master.nvidia.com/etramel/data-designer-plugins/-/blob/main/docs/adding-a-plugin.md).
