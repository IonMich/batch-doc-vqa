from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from batch_doc_vqa.benchmarks.published_runs import (
    build_archive_manifest,
    build_dataset_provenance,
    build_sanitized_request_evidence,
    is_complete_published_archive,
    load_complete_archive_manifest,
    load_published_summaries,
    make_published_run_summary,
    migrate_published_archive,
    published_summary_path,
    summary_to_run_info,
    validate_published_summary,
    write_published_summary,
)
from batch_doc_vqa.benchmarks.table_generator import BenchmarkTableGenerator


class PublishedRunTests(unittest.TestCase):
    def _config(self, *, pages: list[int] | None = None) -> dict:
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
                "images_dir": "/Users/example/private-images",
                "generation_params_effective": {"temperature": 0.0},
                "strict_schema": True,
                "pages": pages if pages is not None else [1, 3],
            },
        }

    def _stats(self, *, request_count: int = 2) -> dict:
        return {
            "digit_top1": 90.0,
            "id_top1": 80.0,
            "lastname_top1": 95.0,
            "docs_detected_count": 1,
            "expected_docs_count": 1,
            "total_cost": 0.2,
            "cost_per_image": 0.2 / request_count,
            "observed_total_cost": 0.2,
            "cost_status": "precise",
            "cost_complete": True,
            "total_requests": request_count,
            "costed_requests": request_count,
            "precise_cost_requests": request_count,
            "estimated_cost_requests": 0,
            "missing_cost_requests": 0,
            "zero_cost_precise_requests": 0,
            "timed_images": request_count,
            "total_images": request_count,
            "fully_parallelizable_runtime_available": True,
            "fully_parallelizable_runtime_seconds": 1.5,
        }

    def _raw_results(self) -> dict:
        return {
            "private/a.png": [
                {
                    "student_full_name": "Sensitive Name",
                    "university_id": "12345678",
                    "section_number": "12345",
                    "_token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "actual_cost": 0.1,
                    },
                    "_timing": {"elapsed_seconds": 1.5},
                    "_generation_meta": {"model": "provider/model", "finish_reason": "stop"},
                }
            ],
            "private/b.png": [
                {
                    "student_full_name": "Another Sensitive Name",
                    "university_id": "87654321",
                    "section_number": "12345",
                    "_token_usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 6,
                        "total_tokens": 18,
                        "actual_cost": 0.1,
                    },
                    "_timing": {"elapsed_seconds": 1.0},
                    "_generation_meta": {"model": "provider/model", "finish_reason": "stop"},
                }
            ],
        }

    def _dataset_files(self, root: Path) -> tuple[Path, Path, Path]:
        doc_info = root / "doc_info.csv"
        test_ids = root / "test_ids.csv"
        source = root / "source.pdf"
        doc_info.write_text(
            "doc,page,filename\ndoc1,1,a-random.png\ndoc1,3,b-random.png\n",
            encoding="utf-8",
        )
        test_ids.write_text(
            "doc,student_id,student_full_name\ndoc1,11111111,Example Person\n",
            encoding="utf-8",
        )
        source.write_bytes(b"stable source fixture")
        return doc_info, test_ids, source

    def _summary(self, root: Path) -> tuple[dict, dict]:
        doc_info, test_ids, source = self._dataset_files(root)
        dataset = build_dataset_provenance(
            doc_info,
            test_ids,
            dataset_id="fixture",
            dataset_source_file=source,
        )
        summary = make_published_run_summary(
            {"run_name": "provider-model_20260721_000000", "config": self._config()},
            self._stats(),
            dataset,
            request_evidence=build_sanitized_request_evidence(self._raw_results()),
        )
        return summary, dataset

    def test_logical_dataset_identity_ignores_randomized_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            test_ids = root / "test_ids.csv"
            source = root / "source.pdf"
            first.write_text("doc,page,filename\n0,1,a-ABC.png\n0,3,b-ABC.png\n", encoding="utf-8")
            second.write_text("doc,page,filename\n0,1,a-XYZ.png\n0,3,b-XYZ.png\n", encoding="utf-8")
            test_ids.write_text("doc,id\n0,12345678\n", encoding="utf-8")
            source.write_bytes(b"same source")

            first_dataset = build_dataset_provenance(first, test_ids, dataset_source_file=source)
            second_dataset = build_dataset_provenance(second, test_ids, dataset_source_file=source)

            self.assertEqual(first_dataset, second_dataset)
            self.assertNotIn("doc_info_sha256", first_dataset)
            self.assertIn("manifest_content_sha256", first_dataset)

    def test_summary_is_sanitized_validated_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, dataset = self._summary(root)
            serialized = json.dumps(summary)
            self.assertNotIn("must-not-publish", serialized)
            self.assertNotIn("Sensitive raw prompt text", serialized)
            self.assertNotIn("Sensitive Name", serialized)
            self.assertNotIn("87654321", serialized)
            self.assertNotIn("/Users/example", serialized)
            self.assertNotIn("images_dir", serialized)
            self.assertNotIn("document_ref", serialized)
            self.assertNotIn("records", summary["request_evidence"])
            self.assertEqual(summary["request_scope"]["pages"], [1, 3])

            runs_dir = root / "published" / "runs"
            path = published_summary_path(runs_dir, summary["run_name"])
            write_published_summary(path, summary)
            loaded = load_published_summaries(runs_dir)
            self.assertEqual(loaded, [summary])
            run_info = summary_to_run_info(loaded[0])
            self.assertEqual(run_info["dataset_content_hash"], dataset["content_hash"])
            self.assertEqual(run_info["request_scope"]["pages"], [1, 3])

    def test_validator_recomputes_fingerprints_and_cost_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, _ = self._summary(Path(tmp))
            bad_fingerprint = copy.deepcopy(summary)
            bad_fingerprint["aggregation_fingerprint"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_published_summary(bad_fingerprint)

            bad_cost = copy.deepcopy(summary)
            bad_cost["stats"]["missing_cost_requests"] = 1
            with self.assertRaisesRegex(ValueError, "cost request counts"):
                validate_published_summary(bad_cost)

            bad_scope = copy.deepcopy(summary)
            bad_scope["request_scope"]["observed_requests"] = 1
            with self.assertRaisesRegex(ValueError, "exactly the expected"):
                validate_published_summary(bad_scope)

    def test_validator_rejects_broad_absolute_paths_and_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, _ = self._summary(Path(tmp))
            for unsafe in ("/tmp/private.json", r"C:\\Users\\name\\private.json", "sk-proj-abcdefghijk"):
                tampered = copy.deepcopy(summary)
                tampered["stats"]["unsafe"] = unsafe
                with self.subTest(unsafe=unsafe), self.assertRaisesRegex(ValueError, "secret-like|absolute"):
                    validate_published_summary(tampered)

    def test_finalized_manifest_marks_archive_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, dataset = self._summary(root)
            runs_dir = root / "published" / "runs"
            write_published_summary(published_summary_path(runs_dir, summary["run_name"]), summary)
            manifest = build_archive_manifest([summary], dataset)
            manifest_path = runs_dir.parent / "archive.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertTrue(is_complete_published_archive(runs_dir, manifest_path))
            self.assertEqual(load_complete_archive_manifest(runs_dir, manifest_path), manifest)

            stale = json.loads(manifest_path.read_text(encoding="utf-8"))
            stale["run_count"] = 2
            manifest_path.write_text(json.dumps(stale), encoding="utf-8")
            self.assertFalse(is_complete_published_archive(runs_dir, manifest_path))

    def test_request_evidence_is_aggregate_and_prediction_free(self) -> None:
        evidence = build_sanitized_request_evidence(self._raw_results())
        serialized = json.dumps(evidence)
        self.assertNotIn("Sensitive Name", serialized)
        self.assertNotIn("12345678", serialized)
        self.assertNotIn("private/a.png", serialized)
        self.assertNotIn("provider/model", serialized)
        self.assertEqual(evidence["request_count"], 2)
        self.assertEqual(evidence["response_field_presence"]["student_full_name"], 2)
        self.assertEqual(evidence["token_usage"]["prompt_tokens_total"], 22)
        self.assertEqual(evidence["cost"]["precise_requests"], 2)

    def test_table_generator_consumes_finalized_archive_without_scoring_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, dataset = self._summary(root)
            doc_info = root / "doc_info.csv"
            test_ids = root / "test_ids.csv"
            runs_dir = root / "published" / "runs"
            write_published_summary(published_summary_path(runs_dir, summary["run_name"]), summary)
            manifest = build_archive_manifest([summary], dataset)
            (runs_dir.parent / "archive.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            doc_info.unlink()
            test_ids.unlink()

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
            self.assertEqual(generator._get_expected_docs_count(str(test_ids)), 1)

    def test_schema_one_archive_can_be_migrated_without_raw_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_info, test_ids, source = self._dataset_files(root)
            raw_results = self._raw_results()
            legacy_records = []
            for index, result in enumerate(raw_results.values()):
                entry = result[0]
                legacy_records.append(
                    {
                        "request_index": index,
                        "document_ref": "a" * 64,
                        "response_fields": {
                            field: {"present": field in entry, "character_count": None}
                            for field in ("student_full_name", "university_id", "ufid", "section_number")
                        },
                        "failure_markers": [],
                        "timing": entry["_timing"],
                        "token_usage": entry["_token_usage"],
                        "cost": {"actual_cost": entry["_token_usage"]["actual_cost"], "provenance": "precise"},
                        "generation": {},
                    }
                )
            legacy = {
                "schema_version": 1,
                "run_name": "provider-model_20260721_000000",
                "dataset": {"content_hash": "legacy"},
                "config": self._config(pages=[]),
                "aggregation_fingerprint": "legacy",
                "aggregation_inputs": {},
                "stats": self._stats(),
                "request_evidence": {"schema_version": 1, "request_count": 2, "records": legacy_records},
            }
            runs_dir = root / "published" / "runs"
            runs_dir.mkdir(parents=True)
            path = published_summary_path(runs_dir, legacy["run_name"])
            path.write_text(json.dumps(legacy), encoding="utf-8")

            migrate_published_archive(
                published_runs_dir=runs_dir,
                doc_info_file=doc_info,
                test_ids_file=test_ids,
                dataset_id="fixture",
                dataset_source_file=source,
                default_request_pages=[1, 3],
            )

            migrated = load_published_summaries(runs_dir)[0]
            self.assertEqual(migrated["schema_version"], 2)
            self.assertNotIn("document_ref", json.dumps(migrated))
            self.assertTrue(is_complete_published_archive(runs_dir, runs_dir.parent / "archive.json"))


if __name__ == "__main__":
    unittest.main()
