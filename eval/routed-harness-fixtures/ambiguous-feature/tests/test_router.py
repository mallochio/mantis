"""Regression tests for the routed step-execution package."""

import unittest

from mantis_router import gateway


class GatewayTests(unittest.TestCase):
    """Cover dispatch envelopes for supported and unsupported requests."""

    def test_echo_returns_value(self):
        self.assertEqual(
            gateway.dispatch({"op": "echo", "value": "hello"}),
            {"status": "ok", "result": "hello"},
        )

    def test_upper_uppercases_value(self):
        self.assertEqual(
            gateway.dispatch({"op": "upper", "value": "hi"}),
            {"status": "ok", "result": "HI"},
        )

    def test_count_returns_length(self):
        self.assertEqual(
            gateway.dispatch({"op": "count", "value": "abc"}),
            {"status": "ok", "result": 3},
        )

    def test_invalid_request_is_rejected(self):
        self.assertEqual(gateway.dispatch({"value": "x"})["status"], "error")

    def test_unsupported_op_raises(self):
        with self.assertRaises(ValueError):
            gateway.dispatch({"op": "explode", "value": "x"})


if __name__ == "__main__":
    unittest.main()
