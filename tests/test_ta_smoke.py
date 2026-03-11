from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from batch_doc_vqa.ta_benchmark.proposals import build_proposals
from batch_doc_vqa.ta_benchmark.report import main as ta_report_main
from batch_doc_vqa.ta_benchmark.score_runs import main as ta_score_main


class TASmokeTests(unittest.TestCase):
    def _write_doc_info(self, path: Path, n_docs: int) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["doc", "page", "filename"])
            writer.writeheader()
            for doc_id in range(n_docs):
                writer.writerow(
                    {
                        "doc": doc_id,
                        "page": 1,
                        "filename": f"doc-{doc_id}-page-1-X{doc_id}.png",
                    }
                )

    def _write_test_ids(self, path: Path, n_docs: int) -> None:
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
            for doc_id in range(n_docs):
                writer.writerow(
                    {
                        "doc": doc_id,
                        "student_id": f"{10000000 + doc_id}",
                        "student_full_name": f"Student {doc_id}",
                        "section_number": "12345",
                        "assignment_id": "1748",
                        "submission_id": f"sub-{doc_id}",
                    }
                )

    def _write_results(self, path: Path, n_docs: int) -> None:
        payload = {}
        for doc_id in range(n_docs):
            filename = f"doc-{doc_id}-page-1-X{doc_id}.png"
            payload[filename] = [
                {
                    "student_full_name": f"Student {doc_id}",
                    "university_id": f"{10000000 + doc_id}",
                    "section_number": "12345",
                }
            ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def test_end_to_end_smoke_20_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            n_docs = 20

            doc_info = root / "doc_info.csv"
            test_ids = root / "test_ids.csv"
            results_json = root / "results.json"
            proposals_json = root / "proposals.json"
            labels_dir = root / "annotations"
            labels_docs = labels_dir / "docs"
            scores_json = root / "scores.json"
            report_md = root / "report.md"
            report_summary = root / "report.summary.json"

            labels_docs.mkdir(parents=True, exist_ok=True)

            self._write_doc_info(doc_info, n_docs)
            self._write_test_ids(test_ids, n_docs)
            self._write_results(results_json, n_docs)

            proposals = build_proposals(
                results_json=results_json,
                doc_info_csv=doc_info,
                output_path=proposals_json,
                test_ids_csv=test_ids,
            )
            self.assertEqual(len(proposals["docs"]), n_docs)

            for row in proposals["docs"]:
                label = dict(row)
                label["review"] = {
                    "annotator_id": "smoke_tester",
                    "status": "verified",
                    "updated_at_utc": "2026-02-26T00:00:00+00:00",
                    "proposal_action": "accepted",
                }
                label["template_version_id"] = "quiz3_v1"
                doc_id = int(label["doc_id"])
                with open(labels_docs / f"doc-{doc_id}.json", "w", encoding="utf-8") as f:
                    json.dump(label, f, indent=2)

            with patch(
                "sys.argv",
                [
                    "ta-score-runs",
                    "--labels-dir",
                    str(labels_dir),
                    "--predictions",
                    str(proposals_json),
                    "--output",
                    str(scores_json),
                ],
            ):
                rc = ta_score_main()
            self.assertEqual(rc, 0)
            self.assertTrue(scores_json.exists())

            with patch(
                "sys.argv",
                [
                    "ta-generate-report",
                    "--scores-json",
                    str(scores_json),
                    "--output-md",
                    str(report_md),
                    "--output-json",
                    str(report_summary),
                ],
            ):
                rc = ta_report_main()
            self.assertEqual(rc, 0)
            self.assertTrue(report_md.exists())
            self.assertTrue(report_summary.exists())

            with open(scores_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["scores"]["doc_count"], n_docs)


if __name__ == "__main__":
    unittest.main()
