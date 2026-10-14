# Agent skills

Contributor skills for this repository live in `.agents/skills/`. The `review-code` skill reviews a local branch or GitHub pull request against the plugin workspace's conventions.

Claude discovers the same skills through `.claude/skills`, which points to `.agents/skills`. Codex can read `.agents/skills/review-code/SKILL.md` directly.
