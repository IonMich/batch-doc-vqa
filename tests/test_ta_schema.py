from __future__ import annotations

from copy import deepcopy
import unittest

from batch_doc_vqa.ta_benchmark.schema import load_error_taxonomy, load_ta_schema, validate_label_payload


class TASchemaValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_ta_schema()
        self.known_error_tags = load_error_taxonomy()
        self.base_label = {
            "schema_version": "ta_benchmark_v1",
            "doc_id": 1,
            "submission_id": "sub-1",
            "assignment_id": 1748,
            "template_version_id": "quiz3_v1",
            "evidence_regions": [
                {
                    "evidence_id": "ev1",
                    "page": 1,
                    "bbox": [0.1, 0.1, 0.9, 0.2],
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
            "tier2": {
                "rubric_scores": [
                    {
                        "problem_uid": "p1",
                        "criterion_id": "setup",
                        "max_points": 3.0,
                        "awarded_points": 2.0,
                        "error_tags": ["missing_assumption"],
                        "evidence_ids": ["ev1"],
                    }
                ],
                "feedback": [
                    {
                        "problem_uid": "p1",
                        "comment": "Show the setup assumptions.",
                        "specificity": 1,
                        "actionability": 1,
                        "correctness": 1,
                        "evidence_ids": ["ev1"],
                    }
                ],
            },
            "review": {
                "annotator_id": "tester",
                "status": "verified",
                "updated_at_utc": "2026-02-26T00:00:00+00:00",
                "proposal_action": "edited",
            },
        }

    def test_valid_payload_passes(self) -> None:
        result = validate_label_payload(
            self.base_label,
            schema=self.schema,
            known_error_tags=self.known_error_tags,
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.issues, [])

    def test_deduction_without_evidence_fails(self) -> None:
        payload = deepcopy(self.base_label)
        payload["tier2"]["rubric_scores"][0]["evidence_ids"] = []
        result = validate_label_payload(
            payload,
            schema=self.schema,
            known_error_tags=self.known_error_tags,
        )
        self.assertFalse(result.is_valid)
        codes = {issue.code for issue in result.issues}
        self.assertIn("missing_deduction_evidence", codes)

    def test_bbox_geometry_invalid_fails(self) -> None:
        payload = deepcopy(self.base_label)
        payload["evidence_regions"][0]["bbox"] = [0.8, 0.2, 0.2, 0.1]
        result = validate_label_payload(
            payload,
            schema=self.schema,
            known_error_tags=self.known_error_tags,
        )
        self.assertFalse(result.is_valid)
        codes = {issue.code for issue in result.issues}
        self.assertIn("bbox_degenerate", codes)

    def test_unknown_error_tag_fails(self) -> None:
        payload = deepcopy(self.base_label)
        payload["tier2"]["rubric_scores"][0]["error_tags"] = ["unknown_tag"]
        result = validate_label_payload(
            payload,
            schema=self.schema,
            known_error_tags=self.known_error_tags,
        )
        self.assertFalse(result.is_valid)
        codes = {issue.code for issue in result.issues}
        self.assertIn("unknown_error_tag", codes)


if __name__ == "__main__":
    unittest.main()
