#!/usr/bin/env python3
"""Tests for strict vs non-strict schema behavior in OpenRouter inference."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from batch_doc_vqa.openrouter import inference


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._payload


class _DummyProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *_args, **_kwargs):
        return None


class TestOpenRouterSchemaValidation(unittest.TestCase):
    def _run_inference_with_custom_schema(self, *, strict_schema: bool | None) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_path = tmp_path / "prompt.md"
            schema_path = tmp_path / "schema.json"
            prompt_path.write_text("Extract one field as structured JSON.", encoding="utf-8")
            schema_path.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "required": ["entity"],
                        "properties": {"entity": {"type": "string"}},
                        "additionalProperties": False,
                    }
                ),
                encoding="utf-8",
            )

            captured: Dict[str, Any] = {}

            class _CapturingRunManager:
                latest_instance = None

                def __init__(self):
                    type(self).latest_instance = self
                    self.base_output_dir = tmp_path / "runs"
                    self.base_output_dir.mkdir(parents=True, exist_ok=True)
                    self.saved_results: Dict[str, Any] = {}

                def create_run_directory(self, config):
                    run_dir = self.base_output_dir / config.run_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    return run_dir

                def save_run_config(self, run_name, config):
                    _ = run_name
                    captured["config"] = config
                    return self.base_output_dir / "config.yaml"

                def save_results(self, run_name, results):
                    _ = run_name
                    self.saved_results = results
                    captured["results"] = results
                    return self.base_output_dir / "results.json"

            def _fake_create_completion(*_args, **_kwargs):
                return _FakeResponse(
                    {
                        "id": "gen-test-1",
                        "provider": "mock-provider",
                        "choices": [
                            {
                                "message": {
                                    "content": '{"entity": 123}',
                                },
                                "finish_reason": "stop",
                                "native_finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 7,
                            "total_tokens": 18,
                        },
                    }
                )

            with (
                patch("batch_doc_vqa.openrouter.inference.fetch_openrouter_models", return_value=[]),
                patch("batch_doc_vqa.openrouter.inference.create_completion", side_effect=_fake_create_completion),
                patch(
                    "batch_doc_vqa.openrouter.inference.parse_response_content",
                    side_effect=lambda content, _fmt: json.loads(content),
                ),
                patch(
                    "batch_doc_vqa.openrouter.inference.get_imagepaths_from_doc_info",
                    return_value=[str(tmp_path / "imgs" / "doc-0-page-1-TEST.png")],
                ),
                patch("batch_doc_vqa.openrouter.inference.create_inference_progress", return_value=_DummyProgress()),
                patch("batch_doc_vqa.openrouter.inference.add_inference_task", return_value=1),
                patch(
                    "batch_doc_vqa.openrouter.inference.batch_update_generation_costs",
                    side_effect=lambda results, max_workers=1: results,
                ),
                patch("batch_doc_vqa.openrouter.inference.build_git_dirty_warning_lines", return_value=[]),
                patch("batch_doc_vqa.openrouter.inference.time.sleep", return_value=None),
                patch("batch_doc_vqa.openrouter.inference.RunManager", _CapturingRunManager),
                patch.dict(
                    inference.MODEL_CONFIG_OVERRIDES,
                    {"unit/test-model": {"schema_retry_max": 0}},
                    clear=False,
                ),
            ):
                run_name = inference.run_openrouter_inference(
                    model_name="unit/test-model",
                    prompt_file=str(prompt_path),
                    schema_file=str(schema_path),
                    strict_schema=strict_schema,
                    images_dir=str(tmp_path / "imgs"),
                    dataset_manifest_file=str(tmp_path / "imgs" / "doc_info.csv"),
                    pages=[1],
                    concurrency=1,
                    retry_max=0,
                    skip_reproducibility_checks=True,
                    provider_zdr=False,
                )

            self.assertTrue(run_name)
            self.assertIn("results", captured)
            first_result_list = next(iter(captured["results"].values()))
            self.assertIsInstance(first_result_list, list)
            self.assertGreaterEqual(len(first_result_list), 1)
            entry = first_result_list[0]
            self.assertIsInstance(entry, dict)
            return entry

    def test_custom_schema_non_strict_enables_passthrough(self):
        entry = self._run_inference_with_custom_schema(strict_schema=False)
        self.assertTrue(entry.get("_schema_failed"))
        self.assertTrue(entry.get("_schema_passthrough"))
        self.assertEqual(entry.get("entity"), 123)

    def test_custom_schema_strict_marks_result_failed(self):
        entry = self._run_inference_with_custom_schema(strict_schema=True)
        self.assertTrue(entry.get("_schema_failed"))
        self.assertNotIn("_schema_passthrough", entry)
        self.assertEqual(entry.get("entity"), 123)

    def test_custom_schema_default_is_strict_when_flag_omitted(self):
        entry = self._run_inference_with_custom_schema(strict_schema=None)
        self.assertTrue(entry.get("_schema_failed"))
        self.assertNotIn("_schema_passthrough", entry)


if __name__ == "__main__":
    unittest.main()
