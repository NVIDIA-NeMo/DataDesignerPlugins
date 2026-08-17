# Contributing to NeMo Data Designer Plugins

Thank you for your interest in contributing to the NeMo Data Designer plugin catalog.

## Before You Open a Pull Request

External contributors must open a GitHub issue before opening a pull request. This gives maintainers an opportunity to confirm that the proposed change is in scope and to align on the approach before implementation begins.

1. Search the [existing issues](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/issues) for related work.
2. If no relevant issue exists, [open an issue](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/issues/new) describing the problem, proposed change, and affected plugin or repository component.
3. Wait for a maintainer to confirm that the contribution is ready to proceed.
4. Link the pull request to the issue with `Fixes #NNN` or `Closes #NNN`.

This issue-first requirement applies to external contributors. Repository collaborators may open pull requests directly for routine maintenance and already-planned work.

## Preparing Your Change

- Follow the repository structure and development guidance in the [README](README.md) and [AGENTS.md](AGENTS.md).
- Add or update tests for changed behavior.
- Run the relevant checks described in the README.
- Keep the pull request focused on the agreed issue scope.
- Complete the pull request template and describe how the change was validated.

Plugin changes should follow the reference implementation under `plugins/data-designer-template/` and remain self-contained within their plugin package.
