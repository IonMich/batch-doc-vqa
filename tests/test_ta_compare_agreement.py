from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from batch_doc_vqa.ta_benchmark.agreement import main as ta_agreement_main
from batch_doc_vqa.ta_benchmark.compare_runs import main as ta_compare_main


def _write_label(path: Path, *, doc_id: int, template: str) -> None:
    payload = {
        "schema_version": "ta_benchmark_v1",
        "doc_id": doc_id,
        "submission_id": f"sub-{doc_id}",
        "assignment_id": 1748,
        "template_version_id": template,
        "evidence_regions": [
            {
                "evidence_id": "ev1",
                "page": 1,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "kind": "problem_description",
            }
        ],
        "problems": [
            {
                "problem_uid": "p1",
                "problem_number": "1",
                "description_text": "Find x",
                "description_evidence_ids": ["ev1"],
                "figure_evidence_ids": [],
            }
        ],
        "tier2": {"rubric_scores": [], "feedback": []},
        "review": {
            "annotator_id": "tester",
            "status": "verified",
            "updated_at_utc": "2026-02-26T00:00:00+00:00",
            "proposal_action": "accepted",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class TACompareAgreementTests(unittest.TestCase):
    def test_agreement_cli_and_compare_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels_a = root / "labels_a" / "docs"
            labels_b = root / "labels_b" / "docs"
            labels_a.mkdir(parents=True, exist_ok=True)
            labels_b.mkdir(parents=True, exist_ok=True)

            _write_label(labels_a / "doc-0.json", doc_id=0, template="quiz3_v1")
            _write_label(labels_b / "doc-0.json", doc_id=0, template="quiz3_v1")

            agreement_out = root / "agreement.json"
            with patch(
                "sys.argv",
                [
                    "ta-check-agreement",
                    "--labels-a",
                    str(labels_a.parent),
                    "--labels-b",
                    str(labels_b.parent),
                    "--output",
                    str(agreement_out),
                ],
            ):
                rc = ta_agreement_main()
            self.assertEqual(rc, 0)
            self.assertTrue(agreement_out.exists())

            score_a = root / "score_a.json"
            score_b = root / "score_b.json"
            score_a.write_text(
                json.dumps(
                    {
                        "run_metadata": {"model": "m1", "source_run": "run1", "cost_per_image": 0.001},
                        "scores": {"rubric_scoring": {"qwk": 0.60}},
                    }
                ),
                encoding="utf-8",
            )
            score_b.write_text(
                json.dumps(
                    {
                        "run_metadata": {"model": "m2", "source_run": "run2", "cost_per_image": 0.002},
                        "scores": {"rubric_scoring": {"qwk": 0.70}},
                    }
                ),
                encoding="utf-8",
            )
            comparison_md = root / "comparison.md"
            with patch(
                "sys.argv",
                [
                    "ta-compare-runs",
                    "--scores",
                    f"{score_a},{score_b}",
                    "--output-md",
                    str(comparison_md),
                ],
            ):
                rc = ta_compare_main()
            self.assertEqual(rc, 0)
            self.assertTrue(comparison_md.exists())


if __name__ == "__main__":
    unittest.main()
