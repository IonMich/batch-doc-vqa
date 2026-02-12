#!/usr/bin/env python3
"""Unit tests for latest-cohort selection logic."""

import unittest
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from batch_doc_vqa.benchmarks.cohorts import select_latest_cohorts


def _run(
    run_name: str,
    timestamp: str,
    *,
    org: str = "google",
    model: str = "gemma-3-4b-it",
    git_commit: Optional[str] = "abc123def456",
    prompt_hash: Optional[str] = "prompt-hash-1",
) -> Dict[str, Any]:
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    run_info: Dict[str, Any] = {
        "run_name": run_name,
        "timestamp": timestamp,
        "timestamp_iso": dt.isoformat(),
    }
    if git_commit is not None or prompt_hash is not None:
        run_info["reproducibility"] = {
            "git_commit": git_commit,
            "prompt_hash": prompt_hash,
        }

    return {
        "run_name": run_name,
        "config": {
            "run_info": run_info,
            "model": {
                "org": org,
                "model": model,
                "variant": None,
            },
        },
    }


class TestLatestCohorts(unittest.TestCase):
    def test_selects_matching_runs_within_window(self):
        runs = [
            _run("m-new", "20260211_120000"),
            _run("m-old-1", "20260211_060000"),
            _run("m-old-2", "20260210_130100"),
        ]
        cohorts = select_latest_cohorts(runs, window_hours=24)
        cohort = cohorts["google/gemma-3-4b-it"]
        self.assertEqual([r["run_name"] for r in cohort.runs], ["m-new", "m-old-1", "m-old-2"])

    def test_excludes_runs_with_different_prompt_hash(self):
        runs = [
            _run("m-new", "20260211_120000", prompt_hash="prompt-a"),
            _run("m-old", "20260211_080000", prompt_hash="prompt-b"),
        ]
        cohorts = select_latest_cohorts(runs, window_hours=24)
        cohort = cohorts["google/gemma-3-4b-it"]
        self.assertEqual([r["run_name"] for r in cohort.runs], ["m-new"])

    def test_excludes_runs_with_different_git_commit(self):
        runs = [
            _run("m-new", "20260211_120000", git_commit="commit-a"),
            _run("m-old", "20260211_080000", git_commit="commit-b"),
        ]
        cohorts = select_latest_cohorts(runs, window_hours=24)
        cohort = cohorts["google/gemma-3-4b-it"]
        self.assertEqual([r["run_name"] for r in cohort.runs], ["m-new"])

    def test_excludes_runs_older_than_window(self):
        runs = [
            _run("m-new", "20260211_120000"),
            _run("m-too-old", "20260210_110000"),
        ]
        cohorts = select_latest_cohorts(runs, window_hours=24)
        cohort = cohorts["google/gemma-3-4b-it"]
        self.assertEqual([r["run_name"] for r in cohort.runs], ["m-new"])

    def test_falls_back_to_anchor_when_signature_missing_on_anchor(self):
        runs = [
            _run("m-new", "20260211_120000", git_commit=None, prompt_hash=None),
            _run("m-old", "20260211_080000"),
        ]
        cohorts = select_latest_cohorts(runs, window_hours=24)
        cohort = cohorts["google/gemma-3-4b-it"]
        self.assertEqual([r["run_name"] for r in cohort.runs], ["m-new"])

    def test_deduplicates_runs_by_run_name(self):
        duplicated = _run("m-new", "20260211_120000")
        runs = [
            duplicated,
            duplicated,
            _run("m-old", "20260211_080000"),
        ]
        cohorts = select_latest_cohorts(runs, window_hours=24)
        cohort = cohorts["google/gemma-3-4b-it"]
        self.assertEqual([r["run_name"] for r in cohort.runs], ["m-new", "m-old"])


if __name__ == "__main__":
    unittest.main()
