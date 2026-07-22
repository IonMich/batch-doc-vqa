#!/usr/bin/env python3
"""Tests for the standalone benchmark Pareto explorer."""

import tempfile
import unittest
from pathlib import Path

from batch_doc_vqa.benchmarks.interactive_pareto import (
    calculate_pareto_frontier,
    create_interactive_pareto_html,
)


class TestInteractivePareto(unittest.TestCase):
    def test_frontier_keeps_only_strict_accuracy_improvements(self):
        points = [
            {"cost": 0.01, "accuracy": 60.0},
            {"cost": 0.02, "accuracy": 60.0},
            {"cost": 0.03, "accuracy": 70.0},
        ]
        self.assertEqual(calculate_pareto_frontier(points), [points[0], points[2]])

    def test_generated_page_has_staged_inspection_and_empty_filter_frontier(self):
        run_stats = {
            "google/example": {
                "run_info": {"config": {"model": {"org": "google", "model": "example", "variant": None}}},
                "stats": {"total_cost": 0.01, "id_top1": 90.0},
            },
            "qwen/example": {
                "run_info": {"config": {"model": {"org": "qwen", "model": "example", "variant": None}}},
                "stats": {"total_cost": 0.02, "id_top1": 95.0},
            },
            "openai/example [r=xhigh]": {
                "run_info": {
                    "config": {
                        "model": {"org": "openai", "model": "example", "variant": None},
                        "additional": {"generation_params_effective": {"reasoning": {"effort": "xhigh"}}},
                    }
                },
                "stats": {"total_cost": 0.03, "id_top1": 92.0},
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "pareto.html"
            create_interactive_pareto_html(run_stats, output_path)
            html = output_path.read_text(encoding="utf-8")
        self.assertIn('id="inspect-org"', html)
        self.assertIn('id="inspect-model" disabled', html)
        self.assertIn("No organizations selected — showing the all-model Pareto frontier", html)
        self.assertIn("orgColors", html)
        self.assertIn('"model":"example (xhigh)"', html)


if __name__ == "__main__":
    unittest.main()
