from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import unittest

from batch_doc_vqa.ta_benchmark.proposals import build_proposals


class TAProposalsNormalizationTests(unittest.TestCase):
    def _write_doc_info(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["doc", "page", "filename"])
            writer.writeheader()
            writer.writerow({"doc": 0, "page": 1, "filename": "doc-0-page-1-AAA.png"})
            writer.writerow({"doc": 0, "page": 2, "filename": "doc-0-page-2-AAA.png"})

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
                    "student_id": "12345678",
                    "student_full_name": "Student Zero",
                    "section_number": "12345",
                    "assignment_id": "1748",
                    "submission_id": "sub-0",
                }
            )

    def test_build_proposals_normalizes_optional_structured_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            doc_info = root / "doc_info.csv"
            test_ids = root / "test_ids.csv"
            results = root / "results.json"
            out = root / "proposals.json"

            self._write_doc_info(doc_info)
            self._write_test_ids(test_ids)

            payload = {
                "doc-0-page-1-AAA.png": {
                    "student_full_name": "Student Zero",
                    "university_id": "12345678",
                    "section_number": "12345",
                    "evidence_regions": [
                        {
                            "bbox": [0.1, 0.1, 0.9, 0.2],
                            "kind": "problem_description",
                        }
                    ],
                    "problems": [
                        {
                            "problem_number": "1",
                            "description_text": "Compute force",
                            "description_evidence_ids": [],
                            "figure_evidence_ids": [],
                        }
                    ],
                },
                "doc-0-page-2-AAA.png": {
                    "evidence_regions": [
                        {
                            "evidence_id": "ev_shared",
                            "bbox": [0.2, 0.2, 0.8, 0.4],
                            "kind": "figure",
                        },
                        {
                            "evidence_id": "ev_shared",
                            "bbox": [0.2, 0.5, 0.8, 0.7],
                            "kind": "figure",
                        },
                    ],
                    "problems": [
                        {
                            "problem_uid": "",
                            "problem_number": "",
                            "description_text": "",
                            "description_evidence_ids": "bad",
                            "figure_evidence_ids": ["ev_shared"],
                        }
                    ],
                },
            }
            with open(results, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            proposals = build_proposals(
                results_json=results,
                doc_info_csv=doc_info,
                test_ids_csv=test_ids,
                output_path=out,
            )

            self.assertEqual(len(proposals["docs"]), 1)
            doc = proposals["docs"][0]
            evidence = doc["evidence_regions"]
            problems = doc["problems"]

            self.assertTrue(any(row.get("evidence_id") == "ev_p1_1" for row in evidence))
            self.assertTrue(any(int(row.get("page", 0)) == 1 for row in evidence))
            self.assertTrue(any(int(row.get("page", 0)) == 2 for row in evidence))

            shared_count = sum(1 for row in evidence if str(row.get("evidence_id", "")).startswith("ev_shared"))
            self.assertEqual(shared_count, 2)

            problem_uids = {str(row.get("problem_uid", "")) for row in problems}
            self.assertIn("p_1", problem_uids)
            self.assertTrue(any(uid.startswith("p_p2_") for uid in problem_uids))

            for row in problems:
                self.assertIsInstance(row.get("description_evidence_ids"), list)
                self.assertIsInstance(row.get("figure_evidence_ids"), list)


if __name__ == "__main__":
    unittest.main()
