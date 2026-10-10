"""Internal caller that routes requests to workers."""

_ROUTES = {
    "alpha": "worker-a",
    "beta": "worker-b",
    "gamma": "worker-c",
}


def handle(request: dict) -> dict:
    """Serve one request using the caller's own routing rules."""
    route = request.get("route")
    if route not in _ROUTES:
        return {"ok": False, "reason": "unknown route", "worker": None}
    worker = _ROUTES[route]
    priority = int(request.get("priority", 0))
    if priority < 0:
        return {"ok": False, "reason": "invalid priority", "worker": None}
    return {"ok": True, "reason": "", "worker": worker}
