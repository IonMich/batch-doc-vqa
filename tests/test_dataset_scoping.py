from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from batch_doc_vqa.benchmarks.table_generator import BenchmarkTableGenerator


def _make_run_info(*, doc_info: str | None = None, images_dir: str | None = None) -> dict:
    additional = {}
    if doc_info is not None:
        additional["doc_info_file"] = doc_info
    if images_dir is not None:
        additional["images_dir"] = images_dir
    return {
        "run_name": "run",
        "config": {
            "additional": additional,
        },
    }


class DatasetScopingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = BenchmarkTableGenerator(interactive=False)

    def test_dataset_mismatch_excludes_run_with_different_doc_info(self) -> None:
        run_info = _make_run_info(doc_info="/tmp/other_dataset/doc_info.csv")
        ok, reason = self.generator._matches_requested_dataset(
            run_info,
            doc_info_file="imgs/q11/doc_info.csv",
            test_ids_file="tests/data/test_ids.csv",
        )
        self.assertFalse(ok)
        self.assertIsNotNone(reason)
        self.assertIn("dataset mismatch", str(reason))

    def test_legacy_run_allowed_for_default_q11_request(self) -> None:
        run_info = _make_run_info()
        ok, reason = self.generator._matches_requested_dataset(
            run_info,
            doc_info_file="imgs/q11/doc_info.csv",
            test_ids_file="tests/data/test_ids.csv",
        )
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_legacy_run_excluded_for_non_default_request(self) -> None:
        run_info = _make_run_info()
        ok, reason = self.generator._matches_requested_dataset(
            run_info,
            doc_info_file="/tmp/new_dataset/doc_info.csv",
            test_ids_file="/tmp/new_dataset/test_ids.csv",
        )
        self.assertFalse(ok)
        self.assertIsNotNone(reason)
        self.assertIn("legacy run", str(reason))

    def test_images_dir_match_when_doc_info_not_recorded(self) -> None:
        images_dir = str(Path("/tmp/new_dataset/images").resolve())
        run_info = _make_run_info(images_dir=images_dir)
        ok, reason = self.generator._matches_requested_dataset(
            run_info,
            doc_info_file="/tmp/new_dataset/images/doc_info.csv",
            test_ids_file="/tmp/new_dataset/test_ids.csv",
        )
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_dataset_fingerprint_depends_on_contents_not_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            first_dir = tmp_path / "first"
            second_dir = tmp_path / "second"
            first_dir.mkdir()
            second_dir.mkdir()

            first_doc_info = first_dir / "doc_info.csv"
            first_test_ids = first_dir / "test_ids.csv"
            second_doc_info = second_dir / "doc_info.csv"
            second_test_ids = second_dir / "test_ids.csv"
            for path, content in (
                (first_doc_info, "doc,page\n1,1\n"),
                (second_doc_info, "doc,page\n1,1\n"),
                (first_test_ids, "doc,id\n1,12345678\n"),
                (second_test_ids, "doc,id\n1,12345678\n"),
            ):
                path.write_text(content, encoding="utf-8")

            first_fingerprint = self.generator._dataset_fingerprint(
                str(first_doc_info),
                str(first_test_ids),
            )
            second_fingerprint = self.generator._dataset_fingerprint(
                str(second_doc_info),
                str(second_test_ids),
            )
            self.assertEqual(first_fingerprint, second_fingerprint)

            second_test_ids.write_text("doc,id\n1,87654321\n", encoding="utf-8")
            changed_fingerprint = self.generator._dataset_fingerprint(
                str(second_doc_info),
                str(second_test_ids),
            )
            self.assertNotEqual(first_fingerprint, changed_fingerprint)

    def test_non_default_dataset_table_hides_opencv_baseline_column(self) -> None:
        run_stats = {
            "google/gemma-3-27b-it": {
                "run_info": {
                    "config": {
                        "model": {
                            "org": "google",
                            "model": "gemma-3-27b-it",
                            "variant": None,
                            "model_size": "27B",
                            "open_weights": True,
                        },
                        "environment": {"runtime": "56 seconds"},
                    }
                },
                "stats": {
                    "digit_top1": 80.17,
                    "id_top1": 45.28,
                    "lastname_top1": 84.28,
                    "id_avg_lev": 1.1887,
                    "lastname_avg_lev": 0.6478,
                    "docs_detected": 94.64,
                    "docs_detected_count": 159,
                    "expected_docs_count": 168,
                    "cost_per_image": 0.000053,
                    "total_cost": 0.0179,
                },
            }
        }
        markdown = self.generator._generate_markdown_table(run_stats, include_baseline=False)
        self.assertNotIn("**OpenCV+CNN**", markdown)
        self.assertIn("**google**<br>gemma-3-27b-it", markdown)

    def test_default_dataset_table_keeps_opencv_baseline_column(self) -> None:
        run_stats = {
            "google/gemma-3-27b-it": {
                "run_info": {
                    "config": {
                        "model": {
                            "org": "google",
                            "model": "gemma-3-27b-it",
                            "variant": None,
                            "model_size": "27B",
                            "open_weights": True,
                        },
                        "environment": {"runtime": "56 seconds"},
                    }
                },
                "stats": {
                    "digit_top1": 80.17,
                    "id_top1": 45.28,
                    "lastname_top1": 84.28,
                    "id_avg_lev": 1.1887,
                    "lastname_avg_lev": 0.6478,
                    "docs_detected": 94.64,
                    "docs_detected_count": 159,
                    "expected_docs_count": 168,
                    "cost_per_image": 0.000053,
                    "total_cost": 0.0179,
                },
            }
        }
        markdown = self.generator._generate_markdown_table(run_stats, include_baseline=True)
        self.assertIn("**OpenCV+CNN**", markdown)


if __name__ == "__main__":
    unittest.main()
