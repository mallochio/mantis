#!/usr/bin/env bash
set -euo pipefail
uv run pytest
uv run ruff check .
uv run python -m py_compile server.py pseudo_label.py eval_outcomes.py
bash -n llm-router.sh endpoint-profile.sh checks.sh _test_headless.sh tests/test_endpoint_profile.sh
bash tests/test_endpoint_profile.sh
uv run python pseudo_label.py --self-test
uv run python eval_outcomes.py --self-test
