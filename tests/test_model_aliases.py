"""Model id aliases clients may send to the chat endpoint."""

from __future__ import annotations

import api


def test_fusion_router_alias_resolves_to_fusion():
    """The prime-agent catalog id ``fusion-router`` must reach the fusion lane."""
    assert api._MODEL_ALIASES["fusion-router"] == "mantis/fusion"
    assert api._MODEL_ALIASES["mantis-fusion-router"] == "mantis/fusion"
    request = api.ChatRequest(
        model=api._MODEL_ALIASES["fusion-router"],
        messages=[api.Message(role="user", content="hi")],
    )
    assert api._experimental_gate(request) is None
