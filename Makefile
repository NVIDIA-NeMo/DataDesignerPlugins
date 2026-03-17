# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: sync lint format test validate catalog codeowners check-catalog check-codeowners check all

# ── Setup ────────────────────────────────────────────────────────────────

sync:
	uv sync --all-packages

# ── Lint ─────────────────────────────────────────────────────────────────

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

# ── Test ─────────────────────────────────────────────────────────────────
# Auto-discover plugins and test each in an isolated venv.
# Catches dependency leaks that workspace-level testing misses.

test:
	@failed=0; \
	for pyproject in plugins/*/pyproject.toml; do \
		plugin_dir="$$(dirname "$$pyproject")"; \
		plugin_name="$$(basename "$$plugin_dir")"; \
		if [ ! -d "$$plugin_dir/tests" ]; then \
			echo "⚠ $$plugin_name: no tests/ directory, skipping"; \
			continue; \
		fi; \
		echo "── Testing $$plugin_name (isolated) ──"; \
		uv venv ".venv-$$plugin_name"; \
		. ".venv-$$plugin_name/bin/activate"; \
		uv pip install -e "$$plugin_dir"; \
		uv pip install pytest; \
		pytest "$$plugin_dir/tests/" -v || failed=1; \
		deactivate; \
	done; \
	exit $$failed

# ── Validate ─────────────────────────────────────────────────────────────
# Assert every installed plugin passes assert_valid_plugin.

validate:
	uv run python tools/validate_plugins.py

# ── Catalog & CODEOWNERS ─────────────────────────────────────────────────

catalog:
	uv run python tools/generate_catalog.py > docs/catalog.md

codeowners:
	uv run python tools/aggregate_codeowners.py > CODEOWNERS

check-catalog:
	uv run python tools/generate_catalog.py > docs/catalog.md.new
	diff docs/catalog.md docs/catalog.md.new
	@rm -f docs/catalog.md.new

check-codeowners:
	uv run python tools/aggregate_codeowners.py > CODEOWNERS.new
	diff CODEOWNERS CODEOWNERS.new
	@rm -f CODEOWNERS.new

# ── Aggregate targets ────────────────────────────────────────────────────

check: check-catalog check-codeowners

all: lint test validate check
