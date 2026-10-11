"""Session-scoped persistence for tool-delegation sidekick context."""

from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict
from typing import Any

import fusion

_MAX_SESSION_CONTEXTS = 128
_MAX_CONTEXT_MESSAGES = 256
_SESSION_CONTEXTS: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_SESSION_LOCK = threading.Lock()


def restore_sidekick(run: Any, session_id: str | None) -> bool:
    """Restore bounded sidekick history for a run only when its session is identified."""
    if not session_id or run.handoffs is None:
        return False
    key = _session_key(session_id)
    with _SESSION_LOCK:
        history = _SESSION_CONTEXTS.get(key)
        if history is None:
            return False
        _SESSION_CONTEXTS.move_to_end(key)
        restored = copy.deepcopy(history)
    system = run.sidekick_messages[:1]
    run.sidekick_messages = [*system, *restored]
    run.sidekick_context_cache["sidekick"] = copy.deepcopy(restored)
    return True


def save_sidekick(run: Any, session_id: str | None) -> None:
    """Store completed sidekick context through Fusion's bounded snapshot path."""
    if not session_id or run.handoffs is None or run.status != "completed":
        return
    run._snapshot_sidekick_contexts(fusion.FusionCoordinator())
    history = run.sidekick_context_cache.get("sidekick", [])
    if not history:
        return
    key = _session_key(session_id)
    with _SESSION_LOCK:
        _SESSION_CONTEXTS[key] = copy.deepcopy(history[-_MAX_CONTEXT_MESSAGES:])
        _SESSION_CONTEXTS.move_to_end(key)
        while len(_SESSION_CONTEXTS) > _MAX_SESSION_CONTEXTS:
            _SESSION_CONTEXTS.popitem(last=False)


def _session_key(session_id: str) -> str:
    """Hash session identity before using it as an in-memory store key."""
    return hashlib.sha256(session_id.encode()).hexdigest()
