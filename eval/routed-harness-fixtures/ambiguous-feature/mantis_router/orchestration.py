"""Orchestration layer: execute one routed step."""

OPS = frozenset({"echo", "upper", "count"})


def run_step(step: dict) -> object:
    """Execute one step and return its result."""
    op = step["op"]
    if op not in OPS:
        raise ValueError(f"unsupported op {op!r}")
    if op == "echo":
        return step.get("value")
    if op == "upper":
        return str(step.get("value", "")).upper()
    return len(str(step.get("value", "")))
