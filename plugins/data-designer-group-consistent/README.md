# data-designer-group-consistent

Deterministic group-scoped record generation for Data Designer.

## Installation

```bash
uv add data-designer data-designer-group-consistent
```

## Usage

The `group-consistent` column type selects one candidate record for each logical
group and writes one or more correlated output columns. Selection is derived from
the group values, role, and seed, so it remains stable across row order, batches,
retries, and resumed runs with the same configuration.

```python
from data_designer.config import DataDesignerConfigBuilder
from data_designer_group_consistent.config import GroupConsistentColumnConfig

builder = DataDesignerConfigBuilder()
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

Every row with the same `patient_id` receives fields from the same candidate
record. Use a different `role` to create an independent identity, such as a
doctor or emergency contact, within the same group.

For the full plugin authoring guide, see the
[main repository docs](https://nvidia-nemo.github.io/DataDesignerPlugins/authoring/).

Plugin documentation for the repository site lives in this package's `docs/`
directory.
