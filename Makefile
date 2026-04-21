# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: sync lint format test validate catalog codeowners check-catalog check-codeowners check-license-headers update-license-headers check all bump release build-plugin validate-release test-plugin check-owner

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
	uv run ddp validate

# ── Catalog & CODEOWNERS ─────────────────────────────────────────────────

catalog:
	uv run ddp catalog > docs/catalog.md

codeowners:
	uv run ddp codeowners > .github/CODEOWNERS

check-catalog:
	uv run ddp catalog > docs/catalog.md.new
	diff docs/catalog.md docs/catalog.md.new
	@rm -f docs/catalog.md.new

check-codeowners:
	uv run ddp codeowners > .github/CODEOWNERS.new
	diff .github/CODEOWNERS .github/CODEOWNERS.new
	@rm -f .github/CODEOWNERS.new

check-license-headers:
	uv run ddp license-headers --check

update-license-headers:
	uv run ddp license-headers

# ── Aggregate targets ────────────────────────────────────────────────────

check: check-catalog check-codeowners check-license-headers

all: lint test validate check

# ── Release ─────────────────────────────────────────────────────────────
# Usage: make release PLUGIN=data-designer-template

PLUGIN ?=
PART ?= patch
PLUGIN_DIR = plugins/$(PLUGIN)

bump:
	@if [ -z "$(PLUGIN)" ]; then echo "ERROR: Set PLUGIN=<name>"; exit 1; fi
	uv run ddp bump $(PLUGIN) $(PART)

validate-release:
	@if [ -z "$(PLUGIN)" ]; then echo "ERROR: Set PLUGIN=<name>"; exit 1; fi
	@if [ ! -d "$(PLUGIN_DIR)" ]; then echo "ERROR: $(PLUGIN_DIR) not found"; exit 1; fi
	@PLUGIN_VERSION=$$(uv run python -c "import tomllib; print(tomllib.load(open('$(PLUGIN_DIR)/pyproject.toml','rb'))['project']['version'])"); \
	uv run ddp check-release "$(PLUGIN)" "$$PLUGIN_VERSION"

test-plugin:
	@if [ -z "$(PLUGIN)" ]; then echo "ERROR: Set PLUGIN=<name>"; exit 1; fi
	uv venv ".venv-$(PLUGIN)"
	. ".venv-$(PLUGIN)/bin/activate" && \
		uv pip install -e "$(PLUGIN_DIR)" && \
		uv pip install pytest && \
		pytest "$(PLUGIN_DIR)/tests/" -v && \
		deactivate

build-plugin: validate-release
	uv build "$(PLUGIN_DIR)" --out-dir dist/

check-owner:
	@if [ -z "$(PLUGIN)" ]; then echo "ERROR: Set PLUGIN=<name>"; exit 1; fi
	@USER_EMAIL=$$(git config user.email); \
	OWNERS=$$(grep -v '^\s*#' "$(PLUGIN_DIR)/CODEOWNERS" | grep -v '^\s*$$' | awk '{for(i=2;i<=NF;i++) print $$i}'); \
	MATCH=0; \
	for owner in $$OWNERS; do \
		case "$$owner" in *"$$USER_EMAIL"*) MATCH=1;; esac; \
	done; \
	if [ "$$MATCH" -eq 0 ]; then \
		echo "WARNING: $$USER_EMAIL is not listed in $(PLUGIN_DIR)/CODEOWNERS"; \
		echo "Listed owners: $$OWNERS"; \
		echo ""; \
		read -p "Continue anyway? (y/N) " confirm; \
		[ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ] || exit 1; \
	fi

release: check-owner test-plugin build-plugin
	@PLUGIN_VERSION=$$(uv run python -c "import tomllib; print(tomllib.load(open('$(PLUGIN_DIR)/pyproject.toml','rb'))['project']['version'])"); \
	echo "Creating tag: $(PLUGIN)/v$$PLUGIN_VERSION"; \
	git tag "$(PLUGIN)/v$$PLUGIN_VERSION"; \
	echo ""; \
	echo "Tag created. Push to trigger CI publish:"; \
	echo "  git push origin $(PLUGIN)/v$$PLUGIN_VERSION"
