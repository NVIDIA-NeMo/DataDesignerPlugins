# Group-consistent generation

Use `group-consistent` to select one correlated candidate record for each
logical group. The selection is deterministic for a given combination of group
values, role, seed, and candidate pool.

## Installation

```bash
uv add data-designer data-designer-group-consistent
```

## Column type

```python
from data_designer_group_consistent.config import GroupConsistentColumnConfig

builder.add_column(
    GroupConsistentColumnConfig(
        name="synthetic_first_name",
        group_by=["patient_id"],
        role="patient",
        seed=7,
        records=[
            {"first_name": "Amina", "last_name": "Diallo", "email": "amina@example.test"},
            {"first_name": "Carlos", "last_name": "Silva", "email": "carlos@example.test"},
        ],
        field_mapping={
            "synthetic_first_name": "first_name",
            "synthetic_last_name": "last_name",
            "synthetic_email": "email",
        },
    ),
)
```

All mapped fields come from the same candidate record. Rows with the same
`patient_id` therefore receive a coherent first name, last name, and email even
when those rows are generated in different batches.

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Primary output column. It must be a key in `field_mapping`. |
| `group_by` | Yes | Ordered list of upstream columns defining a logical group. |
| `records` | Yes | Non-empty candidate record pool. |
| `field_mapping` | Yes | Mapping from output column names to candidate record fields. |
| `role` | No | Namespace for independent entities in one group. Defaults to `default`. |
| `seed` | No | Deterministic selection seed. Defaults to `0`. |

## Implementation notes

The plugin uses SHA-256 rather than Python's process-dependent `hash()` function.
It does not persist a mapping table or call an LLM. Candidate order and pool size
are part of the effective configuration: changing them may change prior choices.

For the full plugin authoring guide, see the
[main repository docs](https://nvidia-nemo.github.io/DataDesignerPlugins/authoring/).
