#!/usr/bin/env python3
"""Regression tests for empty-choice recovery in OpenRouter inference."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable
from unittest.mock import patch

from batch_doc_vqa.openrouter import inference


class _FakeResponse:
    def __init__(
        self,
        payload: Dict[str, Any],
        *,
        status_code: int = 200,
        headers: Dict[str, str] | None = None,
    ):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> Dict[str, Any]:
        return self._payload


class _DummyProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *_args, **_kwargs):
        return None


class TestOpenRouterMissingChoices(unittest.TestCase):
    def _run_inference(self, responses: Iterable[_FakeResponse]) -> tuple[Dict[str, Any], list[float]]:
        response_iter = iter(responses)
        sleep_calls: list[float] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            captured: Dict[str, Any] = {}

            class _CapturingRunManager:
                def __init__(self):
                    self.base_output_dir = tmp_path / "runs"
                    self.base_output_dir.mkdir(parents=True, exist_ok=True)

                def create_run_directory(self, config):
                    run_dir = self.base_output_dir / config.run_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    return run_dir

                def save_run_config(self, _run_name, config):
                    captured["config"] = config
                    return self.base_output_dir / "config.yaml"

                def save_results(self, _run_name, results):
                    captured["results"] = results
                    return self.base_output_dir / "results.json"

            def _fake_create_completion(*_args, **_kwargs):
                try:
                    return next(response_iter)
                except StopIteration as exc:
                    raise AssertionError("create_completion called more times than expected") from exc

            def _fake_sleep(seconds: float):
                sleep_calls.append(seconds)

            with (
                patch("batch_doc_vqa.openrouter.inference.fetch_openrouter_models", return_value=[]),
                patch("batch_doc_vqa.openrouter.inference.create_completion", side_effect=_fake_create_completion),
                patch(
                    "batch_doc_vqa.openrouter.inference.parse_response_content",
                    side_effect=lambda content, _fmt: json.loads(content),
                ),
                patch(
                    "batch_doc_vqa.openrouter.inference.get_imagepaths",
                    return_value=[str(tmp_path / "doc-0-page-1-TEST.png")],
                ),
                patch("batch_doc_vqa.openrouter.inference.create_inference_progress", return_value=_DummyProgress()),
                patch("batch_doc_vqa.openrouter.inference.add_inference_task", return_value=1),
                patch(
                    "batch_doc_vqa.openrouter.inference.batch_update_generation_costs",
                    side_effect=lambda results, max_workers=1: results,
                ),
                patch("batch_doc_vqa.openrouter.inference.build_git_dirty_warning_lines", return_value=[]),
                patch("batch_doc_vqa.openrouter.inference.time.sleep", side_effect=_fake_sleep),
                patch("batch_doc_vqa.openrouter.inference.RunManager", _CapturingRunManager),
            ):
                run_name = inference.run_openrouter_inference(
                    model_name="unit/test-model",
                    images_dir=str(tmp_path),
                    pages=[1],
                    concurrency=1,
                    retry_max=0,
                    skip_reproducibility_checks=True,
                    provider_zdr=False,
                )

        self.assertTrue(run_name)
        self.assertIn("results", captured)
        return captured["results"], sleep_calls

    def test_missing_choices_embedded_error_is_saved_as_api_error(self):
        error_payload = {
            "error": {
                "code": 503,
                "message": "provider warming up",
            },
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 0,
                "total_tokens": 12,
            },
        }
        results, _sleep_calls = self._run_inference(
            [
                _FakeResponse(error_payload),
                _FakeResponse(error_payload),
                _FakeResponse(error_payload),
            ]
        )

        first_result_list = next(iter(results.values()))
        self.assertEqual(len(first_result_list), 1)
        entry = first_result_list[0]
        self.assertEqual(entry.get("_api_error"), 503)
        self.assertEqual(entry.get("_api_error_message"), "provider warming up")
        self.assertNotIn("_no_response", entry)

    def test_missing_choices_429_uses_retry_after_header(self):
        success_payload = {
            "provider": "mock-provider",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "student_full_name": "Ada Lovelace",
                                "ufid": "12345678",
                                "section_number": "11900",
                            }
                        )
                    },
                    "finish_reason": "stop",
                    "native_finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 20,
                "total_tokens": 35,
            },
        }
        results, sleep_calls = self._run_inference(
            [
                _FakeResponse({"usage": {"prompt_tokens": 15, "completion_tokens": 0, "total_tokens": 15}}),
                _FakeResponse(
                    {"error": {"code": 429, "message": "rate limited"}},
                    status_code=429,
                    headers={"Retry-After": "17"},
                ),
                _FakeResponse(success_payload),
            ]
        )

        self.assertIn(17.0, sleep_calls)
        first_result_list = next(iter(results.values()))
        self.assertEqual(len(first_result_list), 1)
        entry = first_result_list[0]
        self.assertEqual(entry.get("student_full_name"), "Ada Lovelace")
        self.assertNotIn("_api_error", entry)


if __name__ == "__main__":
    unittest.main()
