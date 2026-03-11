from __future__ import annotations

import unittest

from batch_doc_vqa.ta_benchmark.metrics import (
    levenshtein_distance,
    quadratic_weighted_kappa,
    score_documents,
)


class TAMetricsTests(unittest.TestCase):
    def test_levenshtein_distance_basic(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)

    def test_quadratic_weighted_kappa_perfect(self) -> None:
        y_true = [0.0, 1.0, 2.0, 3.0]
        y_pred = [0.0, 1.0, 2.0, 3.0]
        self.assertAlmostEqual(quadratic_weighted_kappa(y_true, y_pred), 1.0, places=8)

    def test_score_documents_core_metrics(self) -> None:
        labels = [
            {
                "doc_id": 0,
                "template_version_id": "quiz3_v1",
                "evidence_regions": [
                    {
                        "evidence_id": "ev1",
                        "page": 1,
                        "kind": "problem_description",
                        "bbox": [0.1, 0.1, 0.4, 0.3],
                    }
                ],
                "problems": [
                    {
                        "problem_uid": "p1",
                        "problem_number": "1",
                        "description_text": "Compute acceleration",
                        "description_evidence_ids": ["ev1"],
                        "figure_evidence_ids": ["ev2"],
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
                            "comment": "State assumptions explicitly.",
                            "specificity": 1,
                            "actionability": 1,
                            "correctness": 1,
                            "evidence_ids": ["ev1"],
                        }
                    ],
                },
            }
        ]
        predictions = [
            {
                "doc_id": 0,
                "template_version_id": "quiz3_v1",
                "evidence_regions": [
                    {
                        "evidence_id": "evX",
                        "page": 1,
                        "kind": "problem_description",
                        "bbox": [0.1, 0.1, 0.4, 0.3],
                    }
                ],
                "problems": [
                    {
                        "problem_uid": "p1",
                        "problem_number": "1",
                        "description_text": "Compute acceleration.",
                        "description_evidence_ids": ["evX"],
                        "figure_evidence_ids": ["ev2"],
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
                            "evidence_ids": ["evX"],
                        }
                    ],
                    "feedback": [
                        {
                            "problem_uid": "p1",
                            "comment": "State assumptions explicitly.",
                            "specificity": 1,
                            "actionability": 1,
                            "correctness": 1,
                            "evidence_ids": ["evX"],
                        }
                    ],
                },
            }
        ]

        scores = score_documents(labels, predictions, iou_threshold=0.5)
        self.assertEqual(scores["doc_count"], 1)
        self.assertAlmostEqual(scores["region_detection"]["f1"], 1.0, places=8)
        self.assertAlmostEqual(scores["template_matching"]["top1_accuracy"], 1.0, places=8)
        self.assertAlmostEqual(scores["figure_association"]["f1"], 1.0, places=8)
        self.assertAlmostEqual(scores["error_tagging"]["micro_f1"], 1.0, places=8)
        self.assertAlmostEqual(scores["rubric_scoring"]["mae"], 0.0, places=8)
        self.assertAlmostEqual(scores["rubric_scoring"]["exact_match_rate"], 1.0, places=8)
        self.assertAlmostEqual(scores["feedback_quality"]["overall_agreement"], 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
