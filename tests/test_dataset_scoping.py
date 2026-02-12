from __future__ import annotations

from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
