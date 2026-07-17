from server import _normalize_messages_for_backend, _parse_supra_complexity


def demo():
    assert _normalize_messages_for_backend([{"role": "developer", "content": "x"}]) == [
        {"role": "system", "content": "x"}
    ]
    unchanged = [{"role": "user", "content": "x"}]
    assert _normalize_messages_for_backend(unchanged) is unchanged

    assert _parse_supra_complexity("Domain: Programming | Complexity: 3 | Math: False | Code: True | Route: big model") == 3
    assert _parse_supra_complexity("Domain: Communication | Complexity: 2 | Math: False | Code: False | Route: small model") == 2
    assert _parse_supra_complexity("garbage with no complexity field") == 0
    assert _parse_supra_complexity("Domain: X | Complexity: 5 | Math: True") == 5


if __name__ == "__main__":
    demo()