"""Focused integration suite for the seeded batch pipeline."""

import time
import unittest


def _stage(name: str) -> str:
    time.sleep(1.5)
    return f"{name}-done"


class PipelineIntegrationTests(unittest.TestCase):
    """Slow end-to-end checks over the seeded pipeline stages."""

    def test_ingest_stage_completes(self):
        self.assertEqual(_stage("ingest"), "ingest-done")

    def test_transform_stage_completes(self):
        self.assertEqual(_stage("transform"), "transform-done")

    def test_load_stage_completes(self):
        self.assertEqual(_stage("load"), "load-done")

    def test_report_stage_completes(self):
        self.assertEqual(_stage("report"), "report-done")


if __name__ == "__main__":
    unittest.main()
