# Usage

This tutorial walks through the common patterns for turning repositories into
Data Designer seed rows. The examples use the Python builder API, but the same
configuration fields apply when a workflow is built from serialized config.

## Read a GitHub repository

Start with a small repository and a narrow file pattern. This keeps previews
fast and makes it clear which rows are entering the workflow.

```python
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.interface.data_designer import DataDesigner
from data_designer_github.config import GitHubSeedSource

builder = DataDesignerConfigBuilder()
builder.with_seed_dataset(
    GitHubSeedSource(
        repositories=["pallets/markupsafe"],
        file_pattern="*.py",
        recursive=True,
    )
)

builder.add_column(
    name="_row_id",
    column_type="sampler",
    sampler_type="uuid",
    params={},
)

preview = DataDesigner().preview(builder, num_records=5)
print(preview.dataset[["repo_id", "relative_path", "code_lang", "content"]])
```

The seed rows contain repository provenance and file text. Downstream columns can
then ask questions such as "summarize this file", "identify risky APIs", "write
a short module description", or "extract candidate test scenarios" using the
`content`, `relative_path`, `code_lang`, and `commit_sha` columns.

## Pin a branch, tag, or commit

Use `ref` when the dataset must be reproducible against a specific branch, tag,
or commit. Branches and tags are passed to `git clone --branch`; commit SHAs are
checked out after cloning.

```python
source = GitHubSeedSource(
    repositories=["NVIDIA-NeMo/DataDesigner"],
    ref="v0.5.7",
    clone_depth=1,
    file_pattern="*.py",
    recursive=True,
)
```

For arbitrary commit SHAs, set `clone_depth=None` if the commit may not be
reachable from the shallow default clone.

```python
source = GitHubSeedSource(
    repositories=["NVIDIA-NeMo/DataDesigner"],
    ref="0123456789abcdef0123456789abcdef01234567",
    clone_depth=None,
    file_pattern="*.py",
    recursive=True,
)
```

## Read local repositories

Local repositories are useful for private code, local experiments, or a checked
out monorepo that already exists on disk.

```python
source = GitHubSeedSource(
    repository_paths=[
        "/workspace/services/api",
        "/workspace/libraries/shared",
    ],
    file_pattern="*.py",
    recursive=True,
)
```

If `path` points at a git repository, that repository is read. If `path` points
at a directory whose immediate children are git repositories, each child
repository is discovered and read.

```python
source = GitHubSeedSource(
    path="/workspace/repos",
    file_pattern="*.ts",
    recursive=True,
)
```

## Control which files become rows

The reader first applies `file_pattern` and `recursive`, then filters by
extension, file name, exclude pattern, and file size.

```python
source = GitHubSeedSource(
    repositories=["NVIDIA-NeMo/DataDesigner"],
    file_pattern="*",
    recursive=True,
    include_extensions=["py", "toml", "md"],
    include_file_names=["Dockerfile", "Makefile"],
    exclude_patterns=[
        ".git/**",
        "**/__pycache__/**",
        "**/build/**",
        "**/dist/**",
        "docs/generated/**",
    ],
    max_file_size_bytes=250_000,
)
```

Use `include_extensions=None` for broad repository inventory tasks where the
glob and exclude patterns should decide the candidate set.

```python
source = GitHubSeedSource(
    repositories=["owner/repo"],
    file_pattern="LICENSE*",
    recursive=False,
    include_extensions=None,
)
```

## Typical workflows

`data-designer-github` works best as the seed layer for file-level code
workflows:

- Repository QA: score files for risky dependencies, missing license headers, or
  stale implementation notes.
- Documentation generation: turn source files into module summaries, migration
  notes, or API reference drafts.
- Test ideation: derive test scenarios from implementation files and route them
  to a code-generation column.
- Code search preparation: create embeddings or labels from stable file content
  and repository metadata.
- Dataset construction: sample representative code files from several projects
  while preserving `repo_id`, `relative_path`, and `commit_sha` provenance.

Because the reader emits full file content, prompts should account for file
length and language. A common pattern is to filter or sample seed rows first,
then generate focused columns that reference only the metadata and content each
task needs.

## Operational notes

The plugin requires `git` on `PATH`. GitHub repositories are cloned into a
temporary runtime directory for the reader attachment and local repositories are
read in place. Files that exceed `max_file_size_bytes` are skipped before
hydration. Files that cannot be decoded with `encoding` are skipped with a
warning rather than producing partial text.

The reader does not call the GitHub API, manage credentials, or expand GitHub
issues and pull requests. It is scoped to repository file content so workflows
can compose repository-aware seed data with the rest of Data Designer.
