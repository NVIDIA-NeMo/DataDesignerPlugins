# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

UV ?= uv
PYTHON ?= python3.14
VENV ?= .venv
PI_RUNTIME ?= .runtime
PI_COMMAND ?= $(abspath $(PI_RUNTIME)/node_modules/.bin/pi)

.PHONY: install install-pi test smoke

install:
	$(UV) venv --python $(PYTHON) $(VENV)
	$(UV) pip install --python $(VENV)/bin/python -e '.[test]'

install-pi:
	npm install --prefix $(PI_RUNTIME) --save-exact @earendil-works/pi-coding-agent@1.2.0

test: | .pytest-tmp
	$(VENV)/bin/python -m pytest tests/test_adapter.py --basetemp=.pytest-tmp/unit -q

smoke: | .pytest-tmp
	PI_COMMAND='$(PI_COMMAND)' $(VENV)/bin/python -m pytest tests/test_real_pi.py --basetemp=.pytest-tmp/smoke -q

.pytest-tmp:
	mkdir -p $@
