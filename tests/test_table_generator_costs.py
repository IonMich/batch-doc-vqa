from __future__ import annotations

import unittest

from batch_doc_vqa.benchmarks.table_generator import BenchmarkTableGenerator


class TableGeneratorCostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = BenchmarkTableGenerator(
            metadata_file="/tmp/nonexistent-model-metadata.json",
            interactive=False,
        )

    def test_mixed_precise_and_failed_costs_use_pricing_fallback(self) -> None:
        run_info = {
            "config": {
                "additional": {
                    "model_pricing": {
                        "prompt": "0.001",
                        "completion": "0.01",
                    }
                }
            }
        }
        raw_results = {
            "precise.png": [
                {
                    "_token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "actual_cost": 0.25,
                    }
                }
            ],
            "estimated.png": [
                {
                    "_token_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                    },
                    "_cost_fetch": {
                        "status": "failed",
                        "status_code": 404,
                    },
                }
            ],
        }

        stats = self.generator._calculate_actual_costs(run_info, raw_results)

        self.assertAlmostEqual(stats["total_cost"], 0.293)
        self.assertAlmostEqual(stats["cost_per_image"], 0.1465)
        self.assertEqual(stats["cost_status"], "estimated")
        self.assertTrue(stats["cost_complete"])
        self.assertEqual(stats["total_requests"], 2)
        self.assertEqual(stats["precise_cost_requests"], 1)
        self.assertEqual(stats["estimated_cost_requests"], 1)
        self.assertEqual(stats["missing_cost_requests"], 0)

    def test_missing_cost_without_pricing_is_not_reported_as_complete_total(self) -> None:
        run_info = {"config": {"additional": {}}}
        raw_results = {
            "precise.png": [
                {
                    "_token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "actual_cost": 0.25,
                    }
                }
            ],
            "unknown.png": [
                {
                    "_token_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                    },
                    "_cost_fetch": {
                        "status": "failed",
                        "status_code": 404,
                    },
                }
            ],
        }

        stats = self.generator._calculate_actual_costs(run_info, raw_results)

        self.assertIsNone(stats["total_cost"])
        self.assertIsNone(stats["cost_per_image"])
        self.assertAlmostEqual(stats["observed_total_cost"], 0.25)
        self.assertEqual(stats["cost_status"], "partial")
        self.assertFalse(stats["cost_complete"])
        self.assertEqual(stats["total_requests"], 2)
        self.assertEqual(stats["costed_requests"], 1)
        self.assertEqual(stats["precise_cost_requests"], 1)
        self.assertEqual(stats["estimated_cost_requests"], 0)
        self.assertEqual(stats["missing_cost_requests"], 1)

    def test_all_precise_zero_costs_are_verified(self) -> None:
        stats = self.generator._calculate_actual_costs(
            {"config": {"additional": {}}},
            {
                "a.png": [{"_token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "actual_cost": 0.0}}],
                "b.png": [{"_token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "actual_cost": 0.0}}],
            },
        )
        self.assertEqual(stats["cost_status"], "verified-zero")
        self.assertTrue(stats["cost_complete"])
        self.assertEqual(stats["total_cost"], 0.0)
        self.assertEqual(stats["zero_cost_precise_requests"], 2)


if __name__ == "__main__":
    unittest.main()
