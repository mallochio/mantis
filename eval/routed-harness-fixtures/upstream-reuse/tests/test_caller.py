"""Behavior tests for the internal caller."""

import unittest

import caller


class CallerTests(unittest.TestCase):
    """Cover the caller's public routing behavior."""

    def test_known_route_routes_to_worker(self):
        self.assertEqual(caller.handle({"route": "alpha"})["worker"], "worker-a")

    def test_unknown_route_is_rejected(self):
        self.assertFalse(caller.handle({"route": "delta"})["ok"])

    def test_negative_priority_is_rejected(self):
        self.assertFalse(caller.handle({"route": "beta", "priority": -1})["ok"])

    def test_default_priority_is_accepted(self):
        self.assertTrue(caller.handle({"route": "gamma"})["ok"])


if __name__ == "__main__":
    unittest.main()
