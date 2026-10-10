"""Gateway layer: validate a request and forward it to orchestration."""

from mantis_router import orchestration


def dispatch(request: dict) -> dict:
    """Return the response envelope for one routed request."""
    if not isinstance(request, dict) or "op" not in request:
        return {"status": "error", "reason": "request must be a dict with an op"}
    result = orchestration.run_step(request)
    return {"status": "ok", "result": result}
