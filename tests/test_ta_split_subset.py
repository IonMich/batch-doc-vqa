from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from batch_doc_vqa.ta_benchmark.split_subset import main as split_subset_main


class TASplitSubsetTests(unittest.TestCase):
    def _write_doc_info(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["doc", "page", "filename"])
            writer.writeheader()
            writer.writerow({"doc": 0, "page": 1, "filename": "doc-0-page-1-A.png"})
            writer.writerow({"doc": 0, "page": 2, "filename": "doc-0-page-2-A.png"})
            writer.writerow({"doc": 1, "page": 1, "filename": "doc-1-page-1-B.png"})
            writer.writerow({"doc": 2, "page": 1, "filename": "doc-2-page-1-C.png"})

    def _write_test_ids(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "doc",
                    "student_id",
                    "student_full_name",
                    "section_number",
                    "assignment_id",
                    "submission_id",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "doc": 0,
                    "student_id": "11111111",
                    "student_full_name": "A",
                    "section_number": "12345",
                    "assignment_id": "1748",
                    "submission_id": "sub-0",
                }
            )
            writer.writerow(
                {
                    "doc": 1,
                    "student_id": "22222222",
                    "student_full_name": "B",
                    "section_number": "12345",
                    "assignment_id": "1748",
                    "submission_id": "sub-1",
                }
            )
            writer.writerow(
                {
                    "doc": 2,
                    "student_id": "33333333",
                    "student_full_name": "C",
                    "section_number": "12345",
                    "assignment_id": "1748",
                    "submission_id": "sub-2",
                }
            )

    def test_materialize_dev_docs_subset_from_json_compatible_ta_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            doc_info = root / "doc_info.csv"
            test_ids = root / "test_ids.csv"
            ta_dataset = root / "ta_dataset.yaml"
            out_dir = root / "subsets"

            self._write_doc_info(doc_info)
            self._write_test_ids(test_ids)
            ta_dataset_payload = {
                "doc_info_file": str(doc_info),
                "test_ids_file": str(test_ids),
                "splits": {
                    "dev_docs": [2, 0],
                },
            }
            ta_dataset.write_text(json.dumps(ta_dataset_payload, indent=2), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "ta-split-subset",
                    "--ta-dataset",
                    str(ta_dataset),
                    "--split",
                    "dev_docs",
                    "--output-dir",
                    str(out_dir),
                ],
            ):
                rc = split_subset_main()
            self.assertEqual(rc, 0)

            out_doc_info = out_dir / "doc_info_dev_docs.csv"
            out_test_ids = out_dir / "test_ids_dev_docs.csv"
            self.assertTrue(out_doc_info.exists())
            self.assertTrue(out_test_ids.exists())

            with open(out_doc_info, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({int(r["doc"]) for r in rows}, {0, 2})
            self.assertEqual(len(rows), 3)

            with open(out_test_ids, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({int(r["doc"]) for r in rows}, {0, 2})
            self.assertEqual(len(rows), 2)

    def test_skip_test_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            doc_info = root / "doc_info.csv"
            ta_dataset = root / "ta_dataset.yaml"
            out_dir = root / "subsets"

            self._write_doc_info(doc_info)
            ta_dataset_payload = {
                "doc_info_file": str(doc_info),
                "splits": {
                    "dev_docs": [1],
                },
            }
            ta_dataset.write_text(json.dumps(ta_dataset_payload, indent=2), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "ta-split-subset",
                    "--ta-dataset",
                    str(ta_dataset),
                    "--split",
                    "dev_docs",
                    "--output-dir",
                    str(out_dir),
                    "--skip-test-ids",
                ],
            ):
                rc = split_subset_main()
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "doc_info_dev_docs.csv").exists())
            self.assertFalse((out_dir / "test_ids_dev_docs.csv").exists())


if __name__ == "__main__":
    unittest.main()
