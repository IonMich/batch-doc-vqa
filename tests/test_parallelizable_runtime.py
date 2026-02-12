#!/usr/bin/env python3
"""Tests for fully-parallelizable runtime aggregation behavior."""

import unittest

from batch_doc_vqa.benchmarks.table_generator import BenchmarkTableGenerator


class TestParallelizableRuntime(unittest.TestCase):
    def setUp(self):
        self.generator = BenchmarkTableGenerator(interactive=False)

    def test_calculate_fully_parallelizable_runtime_complete(self):
        raw_results = {
            "img-a": [{"_timing": {"elapsed_seconds": 2.5}}],
            "img-b": [{"_timing": {"elapsed_seconds": 1.2}}],
            "img-c": [{"_timing": {"elapsed_seconds": 3.1}}],
        }
        runtime_data = self.generator._calculate_fully_parallelizable_runtime(raw_results)
        self.assertTrue(runtime_data["fully_parallelizable_runtime_available"])
        self.assertAlmostEqual(runtime_data["fully_parallelizable_runtime_seconds"], 3.1)
        self.assertEqual(runtime_data["timed_images"], 3)
        self.assertEqual(runtime_data["total_images"], 3)

    def test_calculate_fully_parallelizable_runtime_incomplete(self):
        raw_results = {
            "img-a": [{"_timing": {"elapsed_seconds": 2.5}}],
            "img-b": [{"student_full_name": "No timing"}],
        }
        runtime_data = self.generator._calculate_fully_parallelizable_runtime(raw_results)
        self.assertFalse(runtime_data["fully_parallelizable_runtime_available"])
        self.assertEqual(runtime_data["timed_images"], 1)
        self.assertEqual(runtime_data["total_images"], 2)
        self.assertNotIn("fully_parallelizable_runtime_seconds", runtime_data)

    def test_format_runtime_cell_prefers_parallelizable_when_complete(self):
        data = {
            "stats": {
                "n_runs": 3,
                "runtime_seconds": 40.0,
                "fully_parallelizable_runtime_complete": True,
                "fully_parallelizable_runtime_seconds": 12.0,
            },
            "run_info": {"config": {"environment": {"runtime": "40 seconds"}}},
        }
        text = self.generator._format_runtime_cell(data, prefer_fully_parallelizable=True)
        self.assertEqual(text, "12 seconds (n=3)")

    def test_format_runtime_cell_returns_na_when_parallelizable_incomplete(self):
        data = {
            "stats": {
                "n_runs": 2,
                "runtime_seconds": 40.0,
                "fully_parallelizable_runtime_complete": False,
            },
            "run_info": {"config": {"environment": {"runtime": "40 seconds"}}},
        }
        text = self.generator._format_runtime_cell(data, prefer_fully_parallelizable=True)
        self.assertEqual(text, "N/A")


if __name__ == "__main__":
    unittest.main()
