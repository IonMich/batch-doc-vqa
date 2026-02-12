from __future__ import annotations

import unittest

import pandas as pd

from batch_doc_vqa.benchmarks.table_generator import BenchmarkTableGenerator
from batch_doc_vqa.utils.string_matching import get_relaxed_lastname_match, get_surname_candidates


class RelaxedSurnameMatchingTests(unittest.TestCase):
    def test_hyphenated_surname_matches_single_component(self) -> None:
        match = get_relaxed_lastname_match("Ana Barrios", "Ana Barrios-Ramos")
        self.assertEqual(match["distance"], 0)
        self.assertEqual(match["llm_surname"].casefold(), "barrios")
        self.assertEqual(match["gt_surname"].casefold(), "barrios")

    def test_compound_surname_matches_second_to_last_token(self) -> None:
        match = get_relaxed_lastname_match("Grecia Ocando", "Grecia Ocando Tiniacos")
        self.assertEqual(match["distance"], 0)
        self.assertEqual(match["llm_surname"].casefold(), "ocando")
        self.assertEqual(match["gt_surname"].casefold(), "ocando")

    def test_candidate_generation_includes_hyphen_variants(self) -> None:
        candidates = [item.casefold() for item in get_surname_candidates("Ana Barrios-Ramos")]
        self.assertIn("barrios-ramos", candidates)
        self.assertIn("barrios", candidates)
        self.assertIn("ramos", candidates)
        self.assertIn("barriosramos", candidates)


class DocsDetectedFormulaTests(unittest.TestCase):
    def test_docs_detected_uses_id_or_relaxed_lastname_condition(self) -> None:
        generator = BenchmarkTableGenerator(interactive=False)

        df_matching = pd.DataFrame(
            [
                {"doc": 1, "student_full_name": "A One", "student_id": "11111111", "filename": "doc-1-page-3.png", "id_distance": 2, "lastname_distance": 5},
                {"doc": 2, "student_full_name": "B Two", "student_id": "22222222", "filename": "doc-2-page-3.png", "id_distance": 4, "lastname_distance": 0},
                {"doc": 3, "student_full_name": "C Three", "student_id": "33333333", "filename": "doc-3-page-3.png", "id_distance": 4, "lastname_distance": 2},
            ]
        )
        df_test = pd.DataFrame(
            [
                {"doc": 1, "page": 3, "student_full_name": "A One", "student_id": "11111111", "llm_id": "11111111", "id_distance": 2, "lastname_distance": 5},
                {"doc": 2, "page": 3, "student_full_name": "B Two", "student_id": "22222222", "llm_id": "22222222", "id_distance": 4, "lastname_distance": 0},
                {"doc": 3, "page": 3, "student_full_name": "C Three", "student_id": "33333333", "llm_id": "33333333", "id_distance": 4, "lastname_distance": 2},
            ]
        )

        stats = generator._calculate_stats(
            df_matching,
            df_test,
            raw_results={},
            expected_docs_count=3,
        )

        self.assertEqual(stats["docs_detected_count"], 2)
        self.assertAlmostEqual(stats["docs_detected"], 66.6666666667, places=4)


if __name__ == "__main__":
    unittest.main()
