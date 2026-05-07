# Usage

This example creates one Generalist agent environment tuple from a trip-planning
category. The same pattern works for other task categories where searching the
candidate space is harder than verifying a proposed answer.

```python
import pandas as pd
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.interface.data_designer import DataDesigner

seed_df = pd.DataFrame(
    {
        "category": ["planning a travel itinerary"],
        "constraints": ["family-friendly museums, moderate budget, reliable transport"],
    }
)

builder = DataDesignerConfigBuilder()
builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
builder.add_column(
    name="agent_env",
    column_type="generalist-agent-env",
    task_category_column="category",
    context_columns=["constraints"],
    difficulty="hard",
    required_tag="family",
)

result = DataDesigner(artifact_path="artifacts").preview(builder, num_records=1)
environment_tuple = result.dataset.loc[0, "agent_env"]
```

The generated row can be validated with the package helper:

```python
from data_designer_generalist_agent_env.validation import verify_environment_tuple

validation = verify_environment_tuple(environment_tuple)
assert validation.passed, validation.errors
assert validation.answer == environment_tuple["reference_answer"]
```

The output task is intentionally search-like: the solving agent must inspect,
filter, and rank records through the tool interface. The verifier remains
straightforward because it checks fixed constraints and a deterministic
tie-break order directly against the database.
