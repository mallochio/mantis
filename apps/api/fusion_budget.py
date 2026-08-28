"""mantis-fusion run-level budget guard."""

from __future__ import annotations

import time
from typing import Any


class FusionBudgetExceededError(RuntimeError):
    """Raised when a Fusion run exceeds its token, turn, or wall-time budget."""


class FusionBudgetGuard:
    """Track cumulative token, turn, and wall-time spend for one Fusion run."""

    def __init__(
        self,
        max_turns: int | None = None,
        max_tokens: int | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.timeout_ms = timeout_ms
        self._tokens = 0
        self._turns = 0
        self._started = time.monotonic()

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def turns(self) -> int:
        return self._turns

    def consume_turn(self) -> None:
        self._turns += 1
        if self.max_turns is not None and self._turns > self.max_turns:
            raise FusionBudgetExceededError(
                f"turn budget exceeded: {self._turns} > {self.max_turns}"
            )

    def consume_tokens(self, usage: dict[str, Any]) -> None:
        total = 0
        if isinstance(usage, dict):
            for key in ("total_tokens", "input_tokens", "prompt_tokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    total = max(total, value)
            if not total:
                total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if not isinstance(total, int):
            total = 0
        self._tokens += total
        if self.max_tokens is not None and self._tokens > self.max_tokens:
            raise FusionBudgetExceededError(
                f"token budget exceeded: {self._tokens} > {self.max_tokens}"
            )

    def check_timeout(self) -> None:
        if self.timeout_ms is None:
            return
        elapsed_ms = int((time.monotonic() - self._started) * 1000)
        if elapsed_ms > self.timeout_ms:
            raise FusionBudgetExceededError(
                f"wall-time budget exceeded: {elapsed_ms}ms > {self.timeout_ms}ms"
            )
