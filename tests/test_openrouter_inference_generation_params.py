#!/usr/bin/env python3
"""Tests for OpenRouter generation parameter resolution and precedence."""

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


class TestOpenRouterGenerationParams(unittest.TestCase):
    def _run_and_capture_config(self, *, model_name: str = "qwen/qwen3.5-plus-02-15", **run_kwargs: Any) -> Dict[str, Any]:
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

                def save_results(self, _run_name, _results):
                    return self.base_output_dir / "results.json"

            def _fake_create_completion(*_args, **_kwargs):
                return _FakeResponse(
                    {
                        "provider": "mock-provider",
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "student_full_name": "A Student",
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
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
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
                patch("batch_doc_vqa.openrouter.inference.time.sleep", return_value=None),
                patch("batch_doc_vqa.openrouter.inference.RunManager", _CapturingRunManager),
            ):
                run_name = inference.run_openrouter_inference(
                    model_name=model_name,
                    images_dir=str(tmp_path),
                    pages=[1],
                    concurrency=1,
                    retry_max=0,
                    skip_reproducibility_checks=True,
                    provider_zdr=False,
                    **run_kwargs,
                )

        self.assertTrue(run_name)
        self.assertIn("config", captured)
        return captured["config"]

    def test_qwen_profile_is_used_when_cli_overrides_are_omitted(self):
        config = self._run_and_capture_config()
        api = config["api"]
        additional = config["additional"]
        sources = additional["generation_param_sources"]
        effective = additional["generation_params_effective"]

        self.assertEqual(api["temperature"], 0.6)
        self.assertEqual(api["top_p"], 0.95)
        self.assertEqual(api["top_k"], 20)
        self.assertEqual(api["min_p"], 0.0)
        self.assertEqual(api["presence_penalty"], 0.0)
        self.assertEqual(api["repetition_penalty"], 1.0)
        self.assertEqual(sources["temperature"], "model_override")
        self.assertEqual(sources["top_p"], "model_override")
        self.assertEqual(sources["top_k"], "model_override")
        self.assertEqual(sources["min_p"], "model_override")
        self.assertEqual(sources["presence_penalty"], "model_override")
        self.assertEqual(sources["repetition_penalty"], "model_override")
        self.assertIsNone(effective.get("reasoning"))
        self.assertNotIn("include_reasoning", effective)

    def test_qwen_3_vl_instruct_profile_is_used_when_cli_overrides_are_omitted(self):
        config = self._run_and_capture_config(model_name="qwen/qwen3-vl-8b-instruct")
        api = config["api"]
        sources = config["additional"]["generation_param_sources"]

        self.assertEqual(api["temperature"], 0.7)
        self.assertEqual(api["top_p"], 0.8)
        self.assertEqual(api["top_k"], 20)
        self.assertEqual(api["repetition_penalty"], 1.0)
        self.assertIsNone(api["presence_penalty"])
        self.assertEqual(sources["temperature"], "model_override")
        self.assertEqual(sources["top_p"], "model_override")
        self.assertEqual(sources["top_k"], "model_override")
        self.assertEqual(sources["repetition_penalty"], "model_override")
        self.assertEqual(sources["presence_penalty"], "global_default")

    def test_qwen_3_vl_thinking_profile_is_used_when_cli_overrides_are_omitted(self):
        config = self._run_and_capture_config(model_name="qwen/qwen3-vl-30b-a3b-thinking")
        api = config["api"]
        sources = config["additional"]["generation_param_sources"]

        self.assertEqual(api["temperature"], 0.8)
        self.assertEqual(api["top_p"], 0.95)
        self.assertEqual(api["top_k"], 20)
        self.assertEqual(api["repetition_penalty"], 1.0)
        self.assertIsNone(api["presence_penalty"])
        self.assertEqual(sources["temperature"], "model_override")
        self.assertEqual(sources["top_p"], "model_override")
        self.assertEqual(sources["top_k"], "model_override")
        self.assertEqual(sources["repetition_penalty"], "model_override")
        self.assertEqual(sources["presence_penalty"], "global_default")

    def test_google_frontier_profile_uses_global_temp_and_model_sampling_overrides(self):
        config = self._run_and_capture_config(model_name="google/gemini-2.5-flash-lite")
        api = config["api"]
        sources = config["additional"]["generation_param_sources"]

        self.assertEqual(api["temperature"], 1.0)
        self.assertEqual(api["top_p"], 0.95)
        self.assertEqual(api["top_k"], 64)
        self.assertEqual(sources["temperature"], "global_default")
        self.assertEqual(sources["top_p"], "model_override")
        self.assertEqual(sources["top_k"], "model_override")

    def test_nova_lite_profile_is_used_when_cli_overrides_are_omitted(self):
        config = self._run_and_capture_config(model_name="amazon/nova-lite-v1")
        api = config["api"]
        sources = config["additional"]["generation_param_sources"]

        self.assertEqual(api["temperature"], 0.7)
        self.assertEqual(api["top_p"], 0.9)
        self.assertEqual(api["top_k"], 50)
        self.assertEqual(sources["temperature"], "model_override")
        self.assertEqual(sources["top_p"], "model_override")
        self.assertEqual(sources["top_k"], "model_override")

    def test_kimi_k2_5_profile_is_used_when_cli_overrides_are_omitted(self):
        config = self._run_and_capture_config(model_name="moonshotai/kimi-k2.5")
        api = config["api"]
        sources = config["additional"]["generation_param_sources"]

        self.assertEqual(api["temperature"], 0.6)
        self.assertEqual(api["top_p"], 0.95)
        self.assertIsNone(api["top_k"])
        self.assertEqual(sources["temperature"], "model_override")
        self.assertEqual(sources["top_p"], "model_override")
        self.assertEqual(sources["top_k"], "global_default")

    def test_cli_overrides_take_precedence_over_model_profile(self):
        config = self._run_and_capture_config(
            temperature=0.2,
            max_tokens=2048,
            top_p=0.8,
            top_k=15,
            min_p=0.1,
            presence_penalty=0.4,
            repetition_penalty=1.2,
        )
        api = config["api"]
        sources = config["additional"]["generation_param_sources"]

        self.assertEqual(api["temperature"], 0.2)
        self.assertEqual(api["max_tokens"], 2048)
        self.assertEqual(api["top_p"], 0.8)
        self.assertEqual(api["top_k"], 15)
        self.assertEqual(api["min_p"], 0.1)
        self.assertEqual(api["presence_penalty"], 0.4)
        self.assertEqual(api["repetition_penalty"], 1.2)
        self.assertEqual(sources["temperature"], "cli")
        self.assertEqual(sources["max_tokens"], "cli")
        self.assertEqual(sources["top_p"], "cli")
        self.assertEqual(sources["top_k"], "cli")
        self.assertEqual(sources["min_p"], "cli")
        self.assertEqual(sources["presence_penalty"], "cli")
        self.assertEqual(sources["repetition_penalty"], "cli")


if __name__ == "__main__":
    unittest.main()
