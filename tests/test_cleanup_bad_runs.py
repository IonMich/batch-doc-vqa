#!/usr/bin/env python3
"""Tests for bad-run cleanup CLI logic."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from batch_doc_vqa.benchmarks.cleanup_bad_runs import diagnose_run, main


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _base_config(*, runtime: str = "12 seconds") -> dict:
    return {
        "run_info": {
            "run_name": "google-gemma-3-4b-it_20260212_120000",
            "timestamp": "20260212_120000",
            "timestamp_iso": "2026-02-12T12:00:00+00:00",
            "reproducibility": {
                "git_commit": "abc123",
                "prompt_hash": "prompt-1",
            },
        },
        "model": {
            "org": "google",
            "model": "gemma-3-4b-it",
            "variant": None,
        },
        "environment": {
            "runtime": runtime,
        },
    }


class TestCleanupBadRuns(unittest.TestCase):
    def test_diagnose_marks_runtime_tbd_and_missing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "google-gemma-3-4b-it_20260212_120000"
            run_dir.mkdir(parents=True)
            _write_yaml(run_dir / "config.yaml", _base_config(runtime="TBD"))

            diagnosis = diagnose_run(
                run_dir,
                strict=False,
                strict_costs=False,
                strict_reproducibility=False,
            )

            self.assertTrue(diagnosis.is_bad)
            self.assertIn("run incomplete (runtime=TBD)", diagnosis.reasons)
            self.assertIn("missing results.json", diagnosis.reasons)

    def test_diagnose_strict_detects_inference_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "google-gemma-3-4b-it_20260212_120000"
            run_dir.mkdir(parents=True)
            _write_yaml(run_dir / "config.yaml", _base_config())
            _write_json(
                run_dir / "results.json",
                {
                    "imgs/q11/doc-31-page-3-0R29O2Y6.png": [
                        {
                            "student_full_name": "George Washington",
                            "university_id": "793896380",
                            "_schema_failed": True,
                        }
                    ]
                },
            )

            loose = diagnose_run(
                run_dir,
                strict=False,
                strict_costs=False,
                strict_reproducibility=False,
            )
            strict = diagnose_run(
                run_dir,
                strict=True,
                strict_costs=False,
                strict_reproducibility=False,
            )

            self.assertFalse(any("inference failures" in reason for reason in loose.reasons))
            self.assertTrue(any("inference failures" in reason for reason in strict.reasons))

    def test_diagnose_strict_costs_detects_missing_precise_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "google-gemma-3-4b-it_20260212_120000"
            run_dir.mkdir(parents=True)
            _write_yaml(run_dir / "config.yaml", _base_config())
            _write_json(
                run_dir / "results.json",
                {
                    "imgs/q11/doc-0-page-1-VHX7P2D0.png": [
                        {
                            "student_full_name": "Harry S. Truman",
                            "university_id": "11800",
                            "_token_usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 10,
                                "total_tokens": 110,
                                "generation_id": "gen-123",
                            },
                            "_cost_fetch": {
                                "status": "failed",
                                "attempts": 3,
                                "error": "timeout",
                            },
                        }
                    ]
                },
            )

            diagnosis = diagnose_run(
                run_dir,
                strict=True,
                strict_costs=True,
                strict_reproducibility=False,
            )

            self.assertTrue(any("cost fetch failed" in reason for reason in diagnosis.reasons))
            self.assertTrue(any("missing actual_cost" in reason for reason in diagnosis.reasons))
            self.assertTrue(any("generation_id still present" in reason for reason in diagnosis.reasons))

    def test_diagnose_strict_reproducibility_detects_git_dirty_relevant(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "google-gemma-3-4b-it_20260212_120000"
            run_dir.mkdir(parents=True)
            cfg = _base_config()
            cfg["run_info"]["reproducibility"].update(  # type: ignore[index]
                {
                    "git_dirty_relevant": True,
                    "git_dirty_relevant_count": 2,
                }
            )
            _write_yaml(run_dir / "config.yaml", cfg)
            _write_json(
                run_dir / "results.json",
                {
                    "imgs/q11/doc-0-page-1-VHX7P2D0.png": [
                        {
                            "student_full_name": "Harry S. Truman",
                            "university_id": "11800",
                            "section_number": "A",
                        }
                    ]
                },
            )

            loose = diagnose_run(
                run_dir,
                strict=False,
                strict_costs=False,
                strict_reproducibility=False,
            )
            strict_repro = diagnose_run(
                run_dir,
                strict=False,
                strict_costs=False,
                strict_reproducibility=True,
            )

            self.assertFalse(any("git_dirty_relevant" in reason for reason in loose.reasons))
            self.assertTrue(any("git_dirty_relevant=true" in reason for reason in strict_repro.reasons))

    def test_main_dry_run_keeps_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            run_dir = runs_dir / "google-gemma-3-4b-it_20260212_120000"
            run_dir.mkdir(parents=True)
            _write_yaml(run_dir / "config.yaml", _base_config(runtime="TBD"))

            with patch.object(sys, "argv", ["cleanup-bad-runs", "--runs-dir", str(runs_dir)]):
                rc = main()

            self.assertEqual(rc, 0)
            self.assertTrue(run_dir.exists())

    def test_main_apply_quarantine_moves_bad_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            quarantine_dir = Path(tmp) / "quarantine"
            run_dir = runs_dir / "google-gemma-3-4b-it_20260212_120000"
            run_dir.mkdir(parents=True)
            _write_yaml(run_dir / "config.yaml", _base_config(runtime="TBD"))

            with patch.object(
                sys,
                "argv",
                [
                    "cleanup-bad-runs",
                    "--runs-dir",
                    str(runs_dir),
                    "--apply",
                    "--quarantine-dir",
                    str(quarantine_dir),
                ],
            ):
                rc = main()

            self.assertEqual(rc, 0)
            self.assertFalse(run_dir.exists())
            self.assertTrue((quarantine_dir / run_dir.name).exists())


if __name__ == "__main__":
    unittest.main()
