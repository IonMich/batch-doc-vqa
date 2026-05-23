"""Tests for the versioned default_student synthetic benchmark suite."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from batch_doc_vqa.benchmarks.synthetic_suite import (
    REQUIRED_ARTIFACTS,
    generate_suite,
    load_suite_spec,
    task_config_for_variant,
)
from batch_doc_vqa.tools.generate_synthetic_pdf_task import (
    build_generation_plan,
    load_entities,
)


HAS_PYMUPDF = importlib.util.find_spec("fitz") is not None


class TestDefaultStudentSyntheticSuite(unittest.TestCase):
    def test_default_suite_spec_loads_expected_variants(self):
        spec = load_suite_spec()

        self.assertEqual(spec.suite_id, "default_student_synthetic_v1")
        self.assertEqual(spec.task_type, "default_student")
        self.assertEqual(spec.dpi, 150)
        self.assertEqual([variant.id for variant in spec.variants], ["clean", "tabular", "noisy_mixed"])

        expected = {
            "clean": ("clean", 4101),
            "tabular": ("tabular", 4201),
            "noisy_mixed": ("noisy_mixed", 4301),
        }
        for variant in spec.variants:
            self.assertEqual((variant.profile, variant.seed), expected[variant.id])
            self.assertEqual(variant.num_docs, 12)
            self.assertEqual(variant.pages_per_doc, 4)
            self.assertEqual(variant.target_pages, (1, 3))
            self.assertEqual(set(variant.expected_artifacts), REQUIRED_ARTIFACTS)

    def test_variant_generation_plans_are_deterministic(self):
        spec = load_suite_spec()
        fingerprints = {}

        for variant in spec.variants:
            rows = load_entities(spec.entities_file, seed=variant.seed, num_docs_limit=variant.num_docs)
            cfg = task_config_for_variant(variant)

            plan_a = build_generation_plan(rows, cfg, seed=variant.seed)
            plan_b = build_generation_plan(rows, cfg, seed=variant.seed)
            self.assertEqual(plan_a, plan_b)
            self.assertEqual(plan_a["task_type"], "default_student")
            self.assertEqual(plan_a["profile"], variant.profile)
            self.assertEqual(plan_a["pages_per_doc"], variant.pages_per_doc)
            self.assertEqual(plan_a["target_pages"], list(variant.target_pages))
            self.assertEqual(len(plan_a["docs"]), variant.num_docs)
            fingerprints[variant.id] = json.dumps(plan_a, sort_keys=True)

        self.assertEqual(len(set(fingerprints.values())), len(spec.variants))

    @unittest.skipUnless(HAS_PYMUPDF, "PyMuPDF is required for suite PDF/image integration tests")
    def test_generated_labels_and_manifest_match_default_student_schema(self):
        spec = load_suite_spec()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "suite"
            outputs = generate_suite(
                spec=spec,
                variant_ids=["clean"],
                output_root=output_root,
                overwrite=True,
                runs_dir=Path(tmp_dir) / "runs",
            )

            clean_outputs = outputs["clean"]
            labels_path = clean_outputs["labels"]
            doc_info_path = clean_outputs["doc_info"]
            metadata_path = output_root / "clean" / "metadata.json"
            plan_path = output_root / "clean" / "generation_plan.json"

            with open(labels_path, "r", encoding="utf-8", newline="") as f:
                labels = list(csv.DictReader(f))
            self.assertEqual(len(labels), 12)
            self.assertEqual(set(labels[0]), {"doc", "student_id", "student_full_name"})
            self.assertEqual([int(row["doc"]) for row in labels], list(range(12)))
            self.assertTrue(all(row["student_id"].isdigit() for row in labels))
            self.assertTrue(all(len(row["student_id"]) == 8 for row in labels))
            self.assertTrue(all(row["student_full_name"].strip() for row in labels))

            with open(doc_info_path, "r", encoding="utf-8", newline="") as f:
                manifest = list(csv.DictReader(f))
            self.assertEqual(len(manifest), 12 * 4)
            self.assertEqual(set(manifest[0]), {"doc", "page", "filename"})
            self.assertEqual(sorted({int(row["page"]) for row in manifest}), [1, 2, 3, 4])
            self.assertTrue(all(row["filename"].endswith("-clean.png") for row in manifest))

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["task_type"], "default_student")
            self.assertEqual(metadata["profile"], "clean")
            self.assertEqual(metadata["num_docs"], 12)
            self.assertEqual(metadata["pages_per_doc"], 4)
            self.assertEqual(metadata["target_pages"], [1, 3])

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(len(plan["docs"]), 12)
            self.assertIn("student_id", plan["docs"][0]["target"])
            self.assertIn("student_full_name", plan["docs"][0]["target"])
            self.assertIn("section_number", plan["docs"][0]["target"])

            for name in REQUIRED_ARTIFACTS:
                self.assertTrue((output_root / "clean" / spec.variant_by_id("clean").expected_artifacts[name]).exists())

    @unittest.skipUnless(HAS_PYMUPDF, "PyMuPDF is required for suite PDF/image integration tests")
    def test_generated_manifest_is_deterministic(self):
        spec = load_suite_spec()

        with tempfile.TemporaryDirectory() as tmp_dir:
            first = Path(tmp_dir) / "first"
            second = Path(tmp_dir) / "second"
            generate_suite(
                spec=spec,
                variant_ids=["tabular"],
                output_root=first,
                overwrite=True,
                runs_dir=Path(tmp_dir) / "runs",
            )
            generate_suite(
                spec=spec,
                variant_ids=["tabular"],
                output_root=second,
                overwrite=True,
                runs_dir=Path(tmp_dir) / "runs",
            )

            comparable_files = [
                "generation_plan.json",
                "test_ids.csv",
                "entities.normalized.json",
                "images/doc_info.csv",
            ]
            for relative in comparable_files:
                self.assertEqual(
                    (first / "tabular" / relative).read_text(encoding="utf-8"),
                    (second / "tabular" / relative).read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
