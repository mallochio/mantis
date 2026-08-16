#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: OpenAI-compatible serving layer for the OpenFugu TRINITY + Conductor coordinators.
"""
serve.py — Fugu as a single OpenAI-compatible model endpoint.

A client POSTs to /v1/chat/completions as if calling one model; internally the
requested coordinator ("trinity" or "conductor") runs the full loop. The
model field in the request selects the coordinator.

stdlib http.server only — no FastAPI/uvicorn.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import httpx

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's orchestration runtime, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

from mini import (
    FuguRouter,
)

ROUTER: FuguRouter | None = None
_router_lock = threading.Lock()
MODEL_NAME = "mantis-trinity"
MODEL_MODES = {
    "mantis-trinity": "trinity",
    "mantis-ultra": "conductor",
}
MAX_TURNS = 5
DEFAULT_MAX_COMPLETION_TOKENS = 32768
# Compatibility name. Use upstream_output_token_cap() for request-time values.
MAX_UPSTREAM_OUTPUT_TOKENS = DEFAULT_MAX_COMPLETION_TOKENS
WORKER_TIMEOUT = float(os.environ.get("MANTIS_WORKER_TIMEOUT", "240"))

# These providers reject temperature != 1 when reasoning is enabled.
REASONING_MODELS = ("claude-", "gpt-5.6-", "glm-")
PROVIDERS = {
    "openrouter": (
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "OPENROUTER_API_KEY",
    ),
    "opencode-go": (
        os.environ.get("OPENCODE_GO_ENDPOINT_URL", "https://opencode.ai/zen/go/v1"),
        "OPENCODE_API_KEY",
    ),
}
_args: argparse.Namespace | None = None
_coordinators: dict[str, object] = {}
_coordinator_lock = threading.Lock()
_history_context = threading.local()
_provider_client = httpx.Client(timeout=WORKER_TIMEOUT)
_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}
_TRANSIENT_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError, httpx.StreamError)
_FAILOVER_DELAY = 0.5  # short, constant pause between attempts
_PRICES_URL = "https://openrouter.ai/api/v1/models"
_price_cache: dict[str, tuple[float, float]] | None = None
_cache_read_price_cache: dict[str, float] | None = None
_PUBLIC_TOOL_PREFIX = "call_m_"
_PUBLIC_RUN_TOKEN_LENGTH = 22
_INTERNAL_TOOL_ID = re.compile(r"c[0-9a-f]+")
RUN_TTL = float(os.environ.get("MANTIS_RUN_TTL", "600"))
MAX_TOOL_ROUNDS = int(os.environ.get("MANTIS_MAX_TOOL_ROUNDS_PER_STEP", "8"))
MAX_RUNS = int(os.environ.get("MANTIS_MAX_CONCURRENT_RUNS", "32"))
RUN_MAX_MSG_BYTES = 400_000
RUN_STORE = os.environ.get("MANTIS_RUN_STORE", "memory").lower()
if RUN_STORE not in {"memory", "redis", "file"}:
    raise ValueError("MANTIS_RUN_STORE must be memory, redis, or file")
REDIS_URL = os.environ.get("MANTIS_REDIS_URL", "")
_REDIS_PREFIX = os.environ.get("MANTIS_REDIS_PREFIX", "mantis:run:")
REDIS_LOCK_TIMEOUT = max(300, int(WORKER_TIMEOUT * MAX_TURNS + 60))
_redis_client: Any | None = None

_runs: dict[str, Any] = {}
_runs_lock = threading.Lock()
_runs_sweeper_started = False
_learning_lock = threading.Lock()
# Tool names whose payload may contain a test command. The prime-agent harness
# runs tests through `ipython` (code cells), not a bare `bash` tool.
_TEST_TOOL_NAMES = ("bash", "ipython", "exec", "python", "sh", "shell")
_TEST_COMMAND = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:python\d*\s+-m\s+(?:pytest|unittest)|pytest|npm\s+(?:run\s+)?test|"
    r"pnpm\s+(?:run\s+)?test|yarn\s+test|bun\s+test|cargo\s+test|go\s+test|dotnet\s+test|"
    r"mvn\s+test|gradle\s+test)(?:\s|$)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|hf)_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:AKIA[A-Z0-9]{16}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"(?i)\b(api[_ -]?key|token|password|secret|authorization)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bBearer\s+\S+"),
)

__all__ = [k for k in globals() if not k.startswith("__")]
