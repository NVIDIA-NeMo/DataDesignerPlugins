# data-designer-github

GitHub and local git repository seed reader for
[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).

## Installation

```bash
pip install data-designer-github
```

## Usage

This plugin provides a `github` seed source. Once installed, the seed reader is
automatically discovered by Data Designer.

```python
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.interface.data_designer import DataDesigner
from data_designer_github.config import GitHubSeedSource

builder = DataDesignerConfigBuilder()
builder.with_seed_dataset(
    GitHubSeedSource(
        repositories=["NVIDIA-NeMo/DataDesigner"],
        file_pattern="*.py",
        recursive=True,
    )
)

preview = DataDesigner().preview(builder, num_records=5)
print(preview.dataset[["repo_id", "relative_path", "code_lang", "content"]])
```

The reader can also scan local git repositories:

```python
builder.with_seed_dataset(
    GitHubSeedSource(
        path="/path/to/repos",
        repository_paths=["/path/to/one/repo"],
        file_pattern="*.py",
    )
)
```

Seed columns include repository metadata, file paths, language hints, file
content, and content SHA-256 hashes.

For the full plugin authoring guide, see the
[main repository docs](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/main/docs/adding-a-plugin.md).
