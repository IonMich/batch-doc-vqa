#!/usr/bin/env python3
"""Tests for synthetic PDF task generation contracts (Phase 8 v1)."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from batch_doc_vqa.tools.generate_synthetic_pdf_task import (
    build_generation_plan,
    load_entities,
    load_task_config,
)


class TestSyntheticPdfTaskGenerator(unittest.TestCase):
    def test_load_task_config_defaults(self):
        cfg = load_task_config(None)
        self.assertEqual(cfg.task_type, "default_student")
        self.assertEqual(cfg.pages_per_doc, 4)
        self.assertEqual(cfg.target_pages, (1, 3))
        self.assertEqual(cfg.profile, "clean")

    def test_load_task_config_rejects_invalid_target_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "bad.yaml"
            cfg_path.write_text(
                "task_type: default_student\npages_per_doc: 2\ntarget_pages: [1, 3]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_task_config(cfg_path)

    def test_load_entities_normalizes_and_generates_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "entities.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["student_full_name", "student_id", "section_number"])
                writer.writerow(["Alice Example", "12345", "77"])
                writer.writerow(["Bob Example", "", ""])

            rows = load_entities(csv_path, seed=7)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].doc, 0)
            self.assertEqual(rows[0].student_id, "00012345")
            self.assertEqual(rows[0].section_number, "00077")
            self.assertEqual(rows[1].doc, 1)
            self.assertEqual(len(rows[1].student_id), 8)
            self.assertTrue(rows[1].student_id.isdigit())
            self.assertEqual(len(rows[1].section_number), 5)
            self.assertTrue(rows[1].section_number.isdigit())

    def test_generation_plan_is_deterministic_for_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "entities.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["student_full_name", "student_id", "section_number"])
                writer.writerow(["Alice Example", "12345678", "11900"])
                writer.writerow(["Bob Example", "87654321", "11900"])

            rows = load_entities(csv_path, seed=100)
            cfg = load_task_config(None, profile_override="tabular")
            plan_a = build_generation_plan(rows, cfg, seed=999)
            plan_b = build_generation_plan(rows, cfg, seed=999)
            self.assertEqual(plan_a, plan_b)

            # Different seed should alter at least some placement/noise values.
            plan_c = build_generation_plan(rows, cfg, seed=1000)
            self.assertNotEqual(plan_a, plan_c)

    def test_generation_plan_has_expected_structure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            entities_json = Path(tmp_dir) / "entities.json"
            entities_json.write_text(
                json.dumps(
                    [
                        {
                            "student_full_name": "Alice Example",
                            "student_id": "12345678",
                            "section_number": "11900",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            rows = load_entities(entities_json, seed=1)
            cfg = load_task_config(None, profile_override="noisy_mixed")
            plan = build_generation_plan(rows, cfg, seed=2)

            self.assertEqual(plan["task_type"], "default_student")
            self.assertEqual(plan["pages_per_doc"], 4)
            self.assertEqual(len(plan["docs"]), 1)
            self.assertEqual(len(plan["docs"][0]["pages"]), 4)
            self.assertIn("elements", plan["docs"][0]["pages"][0])


if __name__ == "__main__":
    unittest.main()

