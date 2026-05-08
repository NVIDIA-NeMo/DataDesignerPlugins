# data-designer-github

`data-designer-github` is a Data Designer seed reader for repository files. It
turns GitHub repositories or local git repositories into seed rows that carry
file content, path metadata, repository provenance, and commit identifiers.

Use it when a workflow needs code repository data as the starting point for
generation, review, transformation, or indexing tasks. The reader is intentionally
file-oriented: each matching text file becomes one seed row, and downstream Data
Designer columns decide how to summarize, critique, rewrite, label, or enrich
that row.

## Installation

```bash
uv add data-designer data-designer-github
```

The plugin is discovered through the `data_designer.plugins` entry point once it
is installed in the same environment as Data Designer.

## Seed source

Use the `github` seed source when the seed dataset should come from one or more
repositories.

| Field | Required | Description |
| --- | --- | --- |
| `path` | No | A local git repository path, or a directory whose immediate children are git repositories. |
| `repositories` | No | GitHub repositories to clone. Entries may be `owner/name`, `https://github.com/owner/name`, or `https://github.com/owner/name.git`. |
| `repository_paths` | No | Additional explicit local git repository paths to read. |
| `ref` | No | Branch, tag, or commit to check out for cloned GitHub repositories. |
| `clone_depth` | No | Shallow clone depth for GitHub repositories. Defaults to `1`; set to `None` for a full clone. |
| `clone_timeout_seconds` | No | Timeout for each clone or checkout operation. Defaults to `300`. |
| `file_pattern` | No | Inherited file glob from Data Designer's filesystem seed source. For example, `*.py`. |
| `recursive` | No | Whether `file_pattern` is applied recursively. |
| `include_extensions` | No | File extensions to include after the glob match. Defaults to common code and documentation extensions. Set to `None` to allow every extension. |
| `include_file_names` | No | Extensionless file names to include, such as `Dockerfile` and `Makefile`. |
| `exclude_patterns` | No | Relative path glob patterns to skip, including `.git`, cache, build, virtualenv, and dependency directories by default. |
| `max_file_size_bytes` | No | Maximum file size to hydrate into `content`. Defaults to `1_000_000`. |
| `encoding` | No | Text encoding used when reading file contents. Defaults to `utf-8`. |

At least one of `path`, `repositories`, or `repository_paths` is required.

## Output columns

| Column | Description |
| --- | --- |
| `repo_id` | Repository identifier. GitHub repositories use `owner/name`; local repositories use their GitHub remote when available, otherwise the directory name. |
| `repo_url` | Remote origin URL when available. |
| `commit_sha` | Checked-out commit SHA for the repository. |
| `source_kind` | `github` for cloned repositories, or `git_repository` for local repositories. |
| `repository_path` | Local path used by the reader. GitHub repositories are cloned into a temporary runtime directory. |
| `source_path` | Absolute path to the file that produced the seed row. |
| `relative_path` | File path relative to the repository root. |
| `file_name` | Basename of the file. |
| `file_extension` | Lowercase file extension. |
| `code_lang` | Language hint inferred from the file name or extension. |
| `size_bytes` | File size at manifest time. |
| `content_sha256` | SHA-256 hash of the hydrated file bytes. |
| `content` | Decoded text content. |

## Behavior

When the reader is attached, it resolves local repository roots, clones any
configured GitHub repositories, records the checked-out commit, and builds a
manifest of matching files. File content is read during row hydration, so Data
Designer can batch and sample repository content using the same seed reader
interfaces as other filesystem-backed datasets.

The plugin reads repository files only. It does not parse code into functions,
classes, symbols, dependency graphs, or AST nodes. If a workflow needs those
structures, use this reader to collect stable file-level inputs and add
downstream columns that perform the language-specific analysis.

The plugin shells out to `git` for repository operations and does not manage
GitHub API tokens. Public repositories work directly. Private repositories
require the execution environment's git credential configuration to already have
access.
