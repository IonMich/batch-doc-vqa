from __future__ import annotations

import csv
import tempfile
from pathlib import Path
import unittest

from batch_doc_vqa.datasets.ta_annotation_ui import (
    _build_default_label,
    _load_doc_index,
    _load_labels_map,
    save_label,
)
from batch_doc_vqa.ta_benchmark.schema import load_error_taxonomy, load_ta_schema


class TAAnnotationIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_ta_schema()
        self.known_error_tags = load_error_taxonomy()

    def _write_doc_files(self, root: Path) -> tuple[Path, Path, Path]:
        images_dir = root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "doc-0-page-1-A.png").write_bytes(b"")
        (images_dir / "doc-1-page-1-B.png").write_bytes(b"")

        doc_info = root / "doc_info.csv"
        with open(doc_info, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["doc", "page", "filename"])
            writer.writeheader()
            writer.writerow({"doc": 0, "page": 1, "filename": "doc-0-page-1-A.png"})
            writer.writerow({"doc": 1, "page": 1, "filename": "doc-1-page-1-B.png"})

        test_ids = root / "test_ids.csv"
        with open(test_ids, "w", encoding="utf-8", newline="") as f:
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
                    "student_id": "12345678",
                    "student_full_name": "Alice Example",
                    "section_number": "12345",
                    "assignment_id": "1748",
                    "submission_id": "sub0",
                }
            )
            writer.writerow(
                {
                    "doc": 1,
                    "student_id": "87654321",
                    "student_full_name": "Bob Example",
                    "section_number": "12345",
                    "assignment_id": "1748",
                    "submission_id": "sub1",
                }
            )
        return images_dir, doc_info, test_ids

    def test_doc_index_and_label_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir, doc_info, test_ids = self._write_doc_files(root)
            labels_dir = root / "annotations"
            labels_dir.mkdir(parents=True, exist_ok=True)

            docs = _load_doc_index(doc_info_csv=doc_info, test_ids_csv=test_ids, images_dir=images_dir)
            self.assertEqual(sorted(docs.keys()), [0, 1])
            self.assertEqual(docs[0]["assignment_id"], 1748)

            label = _build_default_label(
                doc_id=0,
                assignment_id=1748,
                submission_id="sub0",
            )
            label["template_version_id"] = "quiz3_v1"
            label["review"]["annotator_id"] = "tester"
            label["review"]["status"] = "verified"
            label["evidence_regions"] = [
                {
                    "evidence_id": "ev1",
                    "page": 1,
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                    "kind": "problem_description",
                }
            ]
            label["problems"] = [
                {
                    "problem_uid": "p1",
                    "problem_number": "1",
                    "description_text": "Find x",
                    "description_evidence_ids": ["ev1"],
                    "figure_evidence_ids": [],
                }
            ]

            ok, path, issues = save_label(
                labels_dir=labels_dir,
                payload=label,
                schema=self.schema,
                known_error_tags=self.known_error_tags,
            )
            self.assertTrue(ok)
            self.assertTrue(Path(path).exists())
            self.assertEqual(issues, [])

            loaded = _load_labels_map(labels_dir)
            self.assertIn(0, loaded)
            self.assertEqual(loaded[0]["template_version_id"], "quiz3_v1")

    def test_save_fails_when_feedback_missing_problem_uid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_dir = Path(tmpdir) / "annotations"
            labels_dir.mkdir(parents=True, exist_ok=True)

            label = _build_default_label(
                doc_id=0,
                assignment_id=1748,
                submission_id="sub0",
            )
            label["review"]["annotator_id"] = "tester"
            label["review"]["status"] = "verified"
            label["tier2"]["feedback"] = [
                {
                    "problem_uid": "",
                    "comment": "needs work",
                    "specificity": 0,
                    "actionability": 0,
                    "correctness": 1,
                    "evidence_ids": [],
                }
            ]

            ok, _path, issues = save_label(
                labels_dir=labels_dir,
                payload=label,
                schema=self.schema,
                known_error_tags=self.known_error_tags,
            )
            self.assertFalse(ok)
            self.assertTrue(any(issue["code"] in {"schema", "missing_feedback_problem_ref"} for issue in issues))


if __name__ == "__main__":
    unittest.main()
