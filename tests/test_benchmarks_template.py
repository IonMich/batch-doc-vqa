from __future__ import annotations

from pathlib import Path
import unittest

from batch_doc_vqa.benchmarks.table_generator import _render_benchmarks_template


class BenchmarksTemplateRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        template_path = Path("src/batch_doc_vqa/templates/benchmarks.md")
        self.template = template_path.read_text(encoding="utf-8")

    def test_custom_dataset_render_uses_dataset_metadata_and_hides_baseline(self) -> None:
        rendered = _render_benchmarks_template(
            self.template,
            doc_info_file="/tmp/synth/images/doc_info.csv",
            test_ids_file="/tmp/synth/test_ids.csv",
            expected_docs=5,
            include_baseline=False,
        )
        self.assertIn("`/tmp/synth/images/doc_info.csv`", rendered)
        self.assertIn("`/tmp/synth/test_ids.csv`", rendered)
        self.assertIn("**5**", rendered)
        self.assertNotIn("## Baseline Comparison", rendered)
        self.assertNotIn("{{", rendered)
        self.assertNotIn("}}", rendered)

    def test_default_dataset_render_keeps_baseline_section(self) -> None:
        rendered = _render_benchmarks_template(
            self.template,
            doc_info_file="imgs/q11/doc_info.csv",
            test_ids_file="tests/data/test_ids.csv",
            expected_docs=32,
            include_baseline=True,
        )
        self.assertIn("## Baseline Comparison", rendered)
        self.assertIn("**32**", rendered)
        self.assertNotIn("{{", rendered)
        self.assertNotIn("}}", rendered)


if __name__ == "__main__":
    unittest.main()
