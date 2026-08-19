# Contributing to NeMo Data Designer Plugins

Thank you for your interest in contributing to the NeMo Data Designer plugin catalog.

## Before You Open a Pull Request

External contributors are expected to open a GitHub issue before opening a pull request. Maintainers apply the [`triaged`](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/labels/triaged) label when an open issue has been reviewed, approved, and is ready to be worked on.

1. Search the [existing issues](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/issues) for related work.
2. If no relevant issue exists, [open an issue](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/issues/new) describing the problem, proposed change, and affected plugin or repository component.
3. Wait for a maintainer to apply the `triaged` label before beginning implementation.
4. Link the pull request to the open, triaged issue with `Fixes #NNN` or `Closes #NNN`.

Pull requests from external contributors should link to an open, triaged issue. The linked-issue workflow reports when this policy is not met and rechecks when the selected issue gains or loses the `triaged` label, is closed, or is reopened; repository rules must separately require `Linked Issue Check / check` for that result to block merging. Repository collaborators may open pull requests directly for routine maintenance and already-planned work.

## Preparing Your Change

- Follow the repository structure and development guidance in the [README](README.md) and [AGENTS.md](AGENTS.md).
- Add or update tests for changed behavior.
- Run the relevant checks described in the README.
- Keep the pull request focused on the agreed issue scope.
- Complete the pull request template and describe how the change was validated.

Plugin changes should follow the reference implementation under `plugins/data-designer-template/` and remain self-contained within their plugin package.
