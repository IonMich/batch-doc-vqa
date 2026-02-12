#!/usr/bin/env python3
"""Unit tests for summarize-model-runs helpers."""

import unittest

from batch_doc_vqa.benchmarks.summarize_model_runs import (
    filter_model_runs,
    model_key_from_config,
    resolve_window_hours,
    select_window_runs,
    summarize_values,
)


def _run(
    run_name: str,
    timestamp: str,
    *,
    org: str = "google",
    model: str = "gemma-3-4b-it",
    variant=None,
):
    return {
        "run_name": run_name,
        "config": {
            "run_info": {
                "run_name": run_name,
                "timestamp": timestamp,
            },
            "model": {
                "org": org,
                "model": model,
                "variant": variant,
            },
        },
    }


class TestSummarizeModelRunsHelpers(unittest.TestCase):
    def test_resolve_window_minutes_overrides_hours(self):
        self.assertAlmostEqual(resolve_window_hours(24.0, 90.0), 1.5)

    def test_resolve_window_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            resolve_window_hours(0.0, None)
        with self.assertRaises(ValueError):
            resolve_window_hours(24.0, 0.0)

    def test_model_key_from_config_with_variant(self):
        config = {"model": {"org": "qwen", "model": "qwen3-vl-8b-instruct", "variant": "exacto"}}
        self.assertEqual(model_key_from_config(config), "qwen/qwen3-vl-8b-instruct-exacto")

    def test_filter_model_runs_exact_case_insensitive(self):
        runs = [
            _run("a", "20260212_001000", org="google", model="gemma-3-4b-it"),
            _run("b", "20260212_001100", org="google", model="gemma-3-27b-it"),
        ]
        filtered = filter_model_runs(runs, "Google/Gemma-3-4B-it")
        self.assertEqual([r["run_name"] for r in filtered], ["a"])

    def test_select_window_runs_uses_latest_anchor_and_window(self):
        runs = [
            _run("latest", "20260212_010000"),
            _run("inside", "20260212_003500"),  # 25 min older
            _run("outside", "20260211_230000"),  # 2h older
        ]
        selected = select_window_runs(runs, window_hours=1.0)
        self.assertEqual(selected.anchor_run["run_name"], "latest")
        self.assertEqual([r["run_name"] for r in selected.runs_in_window], ["latest", "inside"])

    def test_summarize_values(self):
        summary = summarize_values([1.0, 2.0, 3.0])
        self.assertEqual(summary["n"], 3.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)
        self.assertEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["stdev"], 0.81649658, places=6)


if __name__ == "__main__":
    unittest.main()
