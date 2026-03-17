# Data Designer Plugins

# Creating a new plugin

If asked to create a new plugin, refer to README.md and docs/ for information on the development workflow for authoring.
Use tooling provided by this project (`data-designer-plugins`) wherever possible. Defer to tooling before running custom commands.
All plugins should be entirely self-contained and manage themselves.
Plugins should not depend on one another locally, however plugins may depend on publicly released pypi data-designer plugins. 

# Releasing a plugin

Never release or publish a plugin-version version as a tag (or to PyPI) without being asked or having express permission from the user. 

# Development Workflow

When creating or updating a plugin, make your edits within a worktree. 
Test your changes locally and ensure that you have a locally green CI by running the Makefile CI commands.
Upon completion, submit a merge request using the `glab` cli.

# Development Style

Tests should be written around public interfaces. 
Modern Python 3.11+ style type annotations should be used (e.g. builtins).
Utilize full Google-style docstrings for implemented functionality.
Don't use ducked, private, function-in-function definitions. 
Favor reusable, composable functions that can be composed within higher-level functions.
Keep function and method definitions short and legible, deferring to composition rather than nesting.

# References

- [DataDesigner GitHub](https://github.com/NVIDIA-NeMo/DataDesigner)
- [DataDesigner Latest Release Notes](https://github.com/NVIDIA-NeMo/DataDesigner/releases/latest)
- [DataDesigner Plugin Authoring Guide](https://nvidia-nemo.github.io/DataDesigner/latest/plugins/overview/)
- [data-designer-plugins authoring guide](docs/adding-a-plugin.md)
- [data-designer-plugins plugin catalog](docs/catalog.md)
