# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: sync lint format test test-devtools test-plugins validate docs docs-server plugin-docs catalog package-index check-plugin-docs check-catalog check-package-index qa-package-index codeowners check-codeowners check-license-headers update-license-headers check all bump release build-plugin validate-release test-plugin check-release-state

# ── Setup ────────────────────────────────────────────────────────────────

sync:
	uv sync --all-packages

PACKAGE_LIST ?= .cache/package-index/packages.json
PACKAGES_URL ?= https://github.com/NVIDIA-NeMo/DataDesignerPlugins/releases/download/ddp-package-assets/
PACKAGE_INDEX_SITE ?= site
PACKAGE_INDEX_QA_DIR ?= /tmp/ddp-package-index-qa

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

test: test-devtools test-plugins

test-devtools:
	uv run pytest devtools/ddp/tests/ -v

test-plugins:
	@failed=0; \
	for pyproject in plugins/*/pyproject.toml; do \
		plugin_dir="$$(dirname "$$pyproject")"; \
		plugin_name="$$(basename "$$plugin_dir")"; \
		if [ ! -d "$$plugin_dir/tests" ]; then \
			echo "⚠ $$plugin_name: no tests/ directory, skipping"; \
			continue; \
		fi; \
		echo "── Testing $$plugin_name (isolated) ──"; \
		uv venv --clear ".venv-$$plugin_name"; \
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

# ── Documentation ───────────────────────────────────────────────────────

docs: plugin-docs
	uv run zensical build --clean --strict
	$(MAKE) package-index

DOCS_DEV_ADDR ?= localhost:8000

docs-server: plugin-docs
	uv run zensical serve --dev-addr $(DOCS_DEV_ADDR)

# ── Plugin docs, catalog & CODEOWNERS ─────────────────────────────────────

plugin-docs:
	uv run ddp plugin-docs

catalog:
	@if [ -z "$(PLUGIN)" ]; then echo "ERROR: Set PLUGIN=<name> to register a first-release catalog entry"; exit 1; fi
	uv run ddp catalog register "$(PLUGIN)"

package-index:
	uv run ddp package-index build --package-list "$(PACKAGE_LIST)" --packages-url "$(PACKAGES_URL)" --site-dir "$(PACKAGE_INDEX_SITE)"

qa-package-index:
	uv run ddp package-index qa --scratch-dir "$(PACKAGE_INDEX_QA_DIR)" --force

codeowners:
	uv run ddp codeowners > .github/CODEOWNERS

check-plugin-docs:
	uv run ddp plugin-docs --check

check-catalog:
	uv run ddp catalog check

check-package-index:
	uv run ddp package-index check --package-list "$(PACKAGE_LIST)" --packages-url "$(PACKAGES_URL)"

check-codeowners:
	uv run ddp codeowners > .github/CODEOWNERS.new
	diff .github/CODEOWNERS .github/CODEOWNERS.new
	@rm -f .github/CODEOWNERS.new

check-license-headers:
	uv run ddp license-headers --check

update-license-headers:
	uv run ddp license-headers

# ── Aggregate targets ────────────────────────────────────────────────────

check: check-plugin-docs check-catalog check-package-index check-codeowners check-license-headers

all: lint test validate check docs

# ── Release ─────────────────────────────────────────────────────────────
# Usage: make release PLUGIN=data-designer-template

PLUGIN ?=
PART ?= patch
PUBLISH ?= 0
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
	uv venv --clear ".venv-$(PLUGIN)"
	. ".venv-$(PLUGIN)/bin/activate" && \
		uv pip install -e "$(PLUGIN_DIR)" && \
		uv pip install pytest && \
		pytest "$(PLUGIN_DIR)/tests/" -v && \
		deactivate

build-plugin: validate-release
	uv build "$(PLUGIN_DIR)" --out-dir dist/

check-release-state:
	@test -z "$$(git status --porcelain)" || { echo "ERROR: release worktree must be clean"; git status --short; exit 1; }
	@git fetch origin main
	@[ "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/main)" ] || { echo "ERROR: release must run from the current origin/main tip"; exit 1; }

release: check-release-state test-plugin build-plugin
	@PLUGIN_VERSION=$$(uv run python -c "import tomllib; print(tomllib.load(open('$(PLUGIN_DIR)/pyproject.toml','rb'))['project']['version'])"); \
	RELEASE_TAG="$(PLUGIN)/v$$PLUGIN_VERSION"; \
	if git rev-parse -q --verify "refs/tags/$$RELEASE_TAG" >/dev/null; then \
		if [ "$$(git rev-list -n 1 "$$RELEASE_TAG")" != "$$(git rev-parse HEAD)" ]; then \
			echo "ERROR: tag $$RELEASE_TAG already exists on a different commit"; \
			exit 1; \
		fi; \
		echo "Tag already exists at HEAD: $$RELEASE_TAG"; \
	else \
		echo "Creating tag: $$RELEASE_TAG"; \
		git tag "$$RELEASE_TAG"; \
	fi; \
	echo ""; \
	if [ "$(PUBLISH)" = "1" ]; then \
		echo "Pushing tag and publishing GitHub Release: $$RELEASE_TAG"; \
		git push origin "$$RELEASE_TAG"; \
		gh release create "$$RELEASE_TAG" --title "$(PLUGIN) v$$PLUGIN_VERSION" --notes "Release $(PLUGIN) v$$PLUGIN_VERSION" --latest=false --verify-tag; \
	else \
		echo "Tag created. Push it, then publish a GitHub Release to trigger CI publish:"; \
		echo "  git push origin $$RELEASE_TAG"; \
		echo "  gh release create $$RELEASE_TAG --title \"$(PLUGIN) v$$PLUGIN_VERSION\" --notes \"Release $(PLUGIN) v$$PLUGIN_VERSION\" --latest=false --verify-tag"; \
		echo ""; \
		echo "Run with PUBLISH=1 to push the tag and create the GitHub Release automatically."; \
	fi
