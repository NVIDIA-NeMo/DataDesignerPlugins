---
name: review-code
description: Review a Data Designer Plugins branch or GitHub pull request for correctness, plugin contracts, packaging, and test coverage. Use when asked for a code review.
---

# Review Code Changes

Review the current branch against `main`, or a pull request when the first argument is a PR number. Treat any remaining arguments as review focus. Follow the user's requested scope when it differs.

## Gather the changes

For a branch review, fetch the base branch when network access is available, then inspect the merge base, commits, full diff, changed-file list, and working-tree status. Review committed changes; mention relevant uncommitted changes separately. If there are no changes, say so.

For a PR review, read the PR description, base and head branches, state, commits, changed files, and diff with `gh pr view` and `gh pr diff`. Read existing inline comments and reviews before reporting a finding, so feedback is not repeated. Obtain the PR files locally in an isolated worktree when possible. If the PR is closed, merged, or draft, note its state and continue the review. If GitHub access is unavailable, explain the limit and review only changes available locally.

Read complete changed files and relevant callers, tests, and documentation. Use `AGENTS.md`, the root `README.md`, `docs/`, and `plugins/data-designer-template/` for this repository's conventions. Use Data Designer's implementation when a plugin API contract is unclear.

## Review passes

Make separate passes for correctness, design, and verification. Report only problems introduced by the change, with a precise file and line reference and a concrete consequence. Check claims against the code before reporting them.

### Correctness and contracts

- Trace inputs, outputs, errors, empty values, and resource cleanup through the changed paths.
- Check that each plugin's `data_designer.plugins` entry point resolves, `plugin.py` exposes the intended plugin, and config, implementation, and tests agree on the contract.
- Check user-visible schema changes for compatibility and for meaningful validation at the boundary.
- Check that plugins remain self-contained. A plugin may depend on a publicly released PyPI `data-designer-*` package, but should not import another plugin's local source.
- Check behavior across the supported Python 3.10+ baseline; flag syntax or dependencies that accidentally require a newer version.

### Design and maintainability

- Compare new plugin structure with `plugins/data-designer-template/` before proposing a new pattern.
- Inspect public names, entry points, configuration defaults, and error handling for consistency with neighboring plugins.
- Check whether changes to metadata or ownership require regenerated catalog, CODEOWNERS, or SPDX header output.
- Cross-check documentation when a change affects setup, configuration, or behavior described to users.

### Tests and repository checks

- Assess whether tests exercise public behavior, relevant edge cases, and plugin validation rather than merely mirroring implementation details.
- Run applicable checks when the checkout and dependencies allow it. Prefer the repository targets: `make lint`, `make test`, `make validate`, and `make check`; `make all` covers the full local CI set. For a focused plugin change, use `make test-plugin PLUGIN=<package-name>` when appropriate.
- Report any checks that could not run and why. Do not turn automatic formatting or lint findings into a long review list when the check already reports them clearly.

## Report

Lead with actionable findings, ordered by severity. For each, give a short title, file and line, the triggering condition, its impact, and a suggested fix. Distinguish confirmed defects from questions that need a maintainer decision. Then give a brief summary of the change and the checks run. If no findings are confirmed, say so and note any concrete remaining test gap.

Keep the review in the conversation for branch mode. In PR mode, prepare the review for the user; post it to GitHub only when the user has explicitly asked to post it. Do not approve or request changes on the PR unless the user asks for that action.
