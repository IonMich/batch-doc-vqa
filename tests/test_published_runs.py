from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from batch_doc_vqa.benchmarks.published_runs import (
    build_archive_manifest,
    build_dataset_provenance,
    build_sanitized_request_evidence,
    is_complete_published_archive,
    load_published_summaries,
    make_published_run_summary,
    published_summary_path,
    summary_to_run_info,
    write_published_summary,
)
from batch_doc_vqa.benchmarks.table_generator import BenchmarkTableGenerator


class PublishedRunTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "run_info": {
                "run_name": "provider-model_20260721_000000",
                "timestamp": "20260721_000000",
                "timestamp_iso": "2026-07-21T00:00:00+00:00",
                "reproducibility": {
                    "git_commit": "abcdef0123456789",
                    "prompt_hash": "prompt-hash",
                    "inference_settings_hash": "settings-hash",
                    "parser_version": "v1",
                    "schema_version": "v1",
                    "git_dirty_relevant_paths_sample": ["/Users/example/private.py"],
                },
            },
            "model": {"org": "provider", "model": "model", "variant": None},
            "api": {"provider": "Router", "temperature": 0.0, "api_key": "must-not-publish"},
            "features": {"structured_output": True, "regex_patterns": False},
            "environment": {"runtime": "2 seconds"},
            "additional": {
                "prompt_template": "Sensitive raw prompt text",
                "doc_info_file": "/Users/example/doc_info.csv",
                "generation_params_effective": {"temperature": 0.0},
                "strict_schema": True,
                "pages": [1, 3],
            },
        }

    def _stats(self) -> dict:
        return {
            "digit_top1": 90.0,
            "id_top1": 80.0,
            "lastname_top1": 95.0,
            "total_cost": 0.2,
            "cost_per_image": 0.01,
            "observed_total_cost": 0.2,
            "cost_status": "precise",
            "cost_complete": True,
            "total_requests": 20,
            "costed_requests": 20,
            "precise_cost_requests": 20,
            "estimated_cost_requests": 0,
            "missing_cost_requests": 0,
        }

    def test_summary_is_sanitized_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_info = root / "doc_info.csv"
            test_ids = root / "test_ids.csv"
            doc_info.write_text("doc,page,filename\ndoc1,1,a.png\ndoc1,3,b.png\n", encoding="utf-8")
            test_ids.write_text("doc,student_id,student_full_name\ndoc1,11111111,Example Person\n", encoding="utf-8")
            dataset = build_dataset_provenance(doc_info, test_ids, dataset_id="fixture")
            summary = make_published_run_summary(
                {"run_name": "provider-model_20260721_000000", "config": self._config()},
                self._stats(),
                dataset,
            )
            serialized = json.dumps(summary)
            self.assertNotIn("must-not-publish", serialized)
            self.assertNotIn("Sensitive raw prompt text", serialized)
            self.assertNotIn("/Users/example", serialized)
            self.assertIn("aggregation_fingerprint", summary)

            runs_dir = root / "published" / "runs"
            path = published_summary_path(runs_dir, summary["run_name"])
            write_published_summary(path, summary)
            loaded = load_published_summaries(runs_dir)
            self.assertEqual(loaded, [summary])
            run_info = summary_to_run_info(loaded[0])
            self.assertEqual(run_info["dataset_content_hash"], dataset["content_hash"])
            self.assertEqual(run_info["config"]["run_info"]["reproducibility"]["aggregation_fingerprint"], summary["aggregation_fingerprint"])

    def test_finalized_manifest_marks_archive_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_info = root / "doc_info.csv"
            test_ids = root / "test_ids.csv"
            doc_info.write_text("doc,page,filename\ndoc1,1,a.png\n", encoding="utf-8")
            test_ids.write_text("doc,student_id,student_full_name\ndoc1,11111111,Example Person\n", encoding="utf-8")
            dataset = build_dataset_provenance(doc_info, test_ids, dataset_id="fixture")
            summary = make_published_run_summary(
                {"run_name": "provider-model_20260721_000000", "config": self._config()},
                self._stats(),
                dataset,
            )
            runs_dir = root / "published" / "runs"
            write_published_summary(published_summary_path(runs_dir, summary["run_name"]), summary)
            manifest = build_archive_manifest([summary], dataset)
            manifest_path = runs_dir.parent / "archive.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertTrue(is_complete_published_archive(runs_dir, manifest_path))

    def test_request_evidence_keeps_operational_data_without_predictions(self) -> None:
        evidence = build_sanitized_request_evidence(
            {
                "private/source.png": [
                    {
                        "student_full_name": "Sensitive Name",
                        "university_id": "12345678",
                        "section_number": "12345",
                        "_token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "actual_cost": 0.12},
                        "_timing": {"elapsed_seconds": 1.5},
                        "_generation_meta": {"model": "provider/model", "finish_reason": "stop"},
                    }
                ]
            }
        )
        serialized = json.dumps(evidence)
        self.assertNotIn("Sensitive Name", serialized)
        self.assertNotIn("12345678", serialized)
        self.assertNotIn("private/source.png", serialized)
        record = evidence["records"][0]
        self.assertTrue(record["response_fields"]["student_full_name"]["present"])
        self.assertEqual(record["token_usage"]["prompt_tokens"], 10)
        self.assertEqual(record["cost"]["provenance"], "precise")

    def test_table_generator_can_consume_published_summary_without_raw_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_info = root / "doc_info.csv"
            test_ids = root / "test_ids.csv"
            doc_info.write_text("doc,page,filename\ndoc1,1,a.png\n", encoding="utf-8")
            test_ids.write_text("doc,student_id,student_full_name\ndoc1,11111111,Example Person\n", encoding="utf-8")
            dataset = build_dataset_provenance(doc_info, test_ids, dataset_id="fixture")
            summary = make_published_run_summary(
                {"run_name": "provider-model_20260721_000000", "config": self._config()},
                self._stats(),
                dataset,
            )
            runs_dir = root / "published" / "runs"
            write_published_summary(published_summary_path(runs_dir, summary["run_name"]), summary)
            generator = BenchmarkTableGenerator(
                runs_base_dir=str(root / "raw-runs-do-not-exist"),
                metadata_file=str(root / "metadata.json"),
                interactive=False,
                source="published",
                published_runs_dir=str(runs_dir),
                cache_results=False,
                write_metadata=False,
            )
            run_stats = generator.build_run_stats(doc_info_file=str(doc_info), test_ids_file=str(test_ids))
            self.assertIn("provider/model", run_stats)
            self.assertEqual(run_stats["provider/model"]["stats"]["cost_status"], "precise")
            self.assertEqual(run_stats["provider/model"]["stats"]["total_cost"], 0.2)


if __name__ == "__main__":
    unittest.main()
