from server import _build_outgoing_body, _decide_cached, _normalize_messages_for_backend, _parse_supra_complexity


def demo():
    assert _decide_cached("same prompt") is _decide_cached("same prompt")
    assert _decide_cached("a") is not _decide_cached("b")

    assert _normalize_messages_for_backend([{"role": "developer", "content": "x"}]) == [
        {"role": "system", "content": "x"}
    ]
    unchanged = [{"role": "user", "content": "x"}]
    assert _normalize_messages_for_backend(unchanged) is unchanged

    assert _parse_supra_complexity("Domain: Programming | Complexity: 3 | Math: False | Code: True | Route: big model") == 3
    assert _parse_supra_complexity("Domain: Communication | Complexity: 2 | Math: False | Code: False | Route: small model") == 2
    assert _parse_supra_complexity("garbage with no complexity field") == 0
    assert _parse_supra_complexity("Domain: X | Complexity: 5 | Math: True") == 5

    # _build_outgoing_body tests
    # 1. Happy path: remap max_tokens, drop stop, inject effort, keep temperature for non-gpt
    orig_body = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "temperature": 0.7,
        "stop": ["\n"],
    }
    backend_deepseek = {"model": "deepseek-v4-pro", "effort": "xhigh", "max_tokens": 131072}
    out1 = _build_outgoing_body(orig_body, backend_deepseek)
    assert out1["model"] == "deepseek-v4-pro"
    assert out1["max_completion_tokens"] == 100
    assert "max_tokens" not in out1
    assert "stop" not in out1
    assert out1["reasoning_effort"] == "xhigh"
    assert out1["temperature"] == 0.7

    # 2. Clamping max_tokens when backend max_tokens is smaller
    backend_clamped = {"model": "deepseek-v4-pro", "effort": "", "max_tokens": 64}
    out2 = _build_outgoing_body(orig_body, backend_clamped)
    assert out2["max_completion_tokens"] == 64
    assert "reasoning_effort" not in out2

    # 3. gpt-5.6 temperature drop (temp != 1 or None drops temperature; temp == 1 keeps it)
    backend_gpt = {"model": "gpt-5.6-luna", "effort": "xhigh", "max_tokens": None}
    out3_drop = _build_outgoing_body({"temperature": 0.7}, backend_gpt)
    assert "temperature" not in out3_drop
    out3_keep = _build_outgoing_body({"temperature": 1}, backend_gpt)
    assert out3_keep["temperature"] == 1

    # 4. developer -> system role normalization
    dev_body = {"messages": [{"role": "developer", "content": "x"}, {"role": "user", "content": "y"}], "max_tokens": 5}
    out4 = _build_outgoing_body(dev_body, backend_deepseek)
    assert out4["messages"] == [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]

    # 5. Original body is untouched (immutability check)
    body_before = {"messages": [{"role": "developer", "content": "x"}], "max_tokens": 100, "stop": ["\n"]}
    body_copy = dict(body_before)
    _build_outgoing_body(body_before, backend_deepseek)
    assert body_before == body_copy


if __name__ == "__main__":
    demo()