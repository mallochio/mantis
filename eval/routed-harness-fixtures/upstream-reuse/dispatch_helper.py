"""Shared routed dispatch helper reused by internal callers."""

ROUTE_TABLE = {
    "alpha": "worker-a",
    "beta": "worker-b",
    "gamma": "worker-c",
}


def routed_dispatch(request: dict) -> dict:
    """Return the worker that should serve one request."""
    route = request.get("route")
    if route not in ROUTE_TABLE:
        return {"ok": False, "reason": "unknown route", "worker": None}
    worker = ROUTE_TABLE[route]
    priority = int(request.get("priority", 0))
    if priority < 0:
        return {"ok": False, "reason": "invalid priority", "worker": None}
    return {"ok": True, "reason": "", "worker": worker}
