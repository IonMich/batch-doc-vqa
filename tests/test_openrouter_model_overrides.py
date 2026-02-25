#!/usr/bin/env python3
"""Tests for OpenRouter model-specific override resolution."""

import unittest
from unittest.mock import patch

from batch_doc_vqa.openrouter.api import resolve_model_config_overrides


class TestOpenRouterModelOverrideResolution(unittest.TestCase):
    def test_qwen_3_5_series_includes_recommended_defaults(self):
        resolved = resolve_model_config_overrides("qwen/qwen3.5-27b")
        self.assertEqual(resolved.get("temperature"), 0.6)
        self.assertEqual(resolved.get("top_p"), 0.95)
        self.assertEqual(resolved.get("top_k"), 20)
        self.assertEqual(resolved.get("min_p"), 0.0)
        self.assertEqual(resolved.get("presence_penalty"), 0.0)
        self.assertEqual(resolved.get("repetition_penalty"), 1.0)

    def test_qwen_3_vl_instruct_series_includes_recommended_defaults(self):
        expected = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
        }
        for model_name in (
            "qwen/qwen3-vl-8b-instruct",
            "qwen/qwen3-vl-30b-a3b-instruct",
            "qwen/qwen3-vl-32b-instruct",
            "qwen/qwen3-vl-235b-a22b-instruct",
        ):
            with self.subTest(model_name=model_name):
                resolved = resolve_model_config_overrides(model_name)
                self.assertEqual(resolved, expected)

    def test_qwen_3_vl_thinking_series_includes_recommended_defaults(self):
        expected_by_model = {
            "qwen/qwen3-vl-8b-thinking": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.0,
            },
            "qwen/qwen3-vl-30b-a3b-thinking": {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.0,
            },
            "qwen/qwen3-vl-235b-a22b-thinking": {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.0,
            },
        }
        for model_name, expected in expected_by_model.items():
            with self.subTest(model_name=model_name):
                resolved = resolve_model_config_overrides(model_name)
                self.assertEqual(resolved, expected)

    def test_google_frontier_models_include_recommended_sampling_defaults(self):
        expected = {
            "top_p": 0.95,
            "top_k": 64,
        }
        for model_name in (
            "google/gemma-3-4b-it",
            "google/gemma-3-27b-it",
            "google/gemini-2.5-flash-lite",
            "google/gemini-2.5-flash-lite-preview-09-2025",
        ):
            with self.subTest(model_name=model_name):
                resolved = resolve_model_config_overrides(model_name)
                self.assertEqual(resolved, expected)

    def test_nova_lite_includes_recommended_defaults(self):
        resolved = resolve_model_config_overrides("amazon/nova-lite-v1")
        self.assertEqual(
            resolved,
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
            },
        )

    def test_kimi_k2_5_includes_recommended_defaults(self):
        resolved = resolve_model_config_overrides("moonshotai/kimi-k2.5")
        self.assertEqual(
            resolved,
            {
                "temperature": 0.6,
                "top_p": 0.95,
            },
        )

    def test_resolves_exact_model_override(self):
        with patch.dict(
            "batch_doc_vqa.openrouter.api.MODEL_CONFIG_OVERRIDES",
            {"unit/test-model": {"temperature": 0.6}},
            clear=False,
        ):
            resolved = resolve_model_config_overrides("unit/test-model")
        self.assertEqual(resolved.get("temperature"), 0.6)

    def test_resolves_base_model_when_variant_suffix_present(self):
        with patch.dict(
            "batch_doc_vqa.openrouter.api.MODEL_CONFIG_OVERRIDES",
            {"unit/test-model": {"top_p": 0.95}},
            clear=False,
        ):
            resolved = resolve_model_config_overrides("unit/test-model:free")
        self.assertEqual(resolved.get("top_p"), 0.95)

    def test_returns_copy_not_original_mapping(self):
        with patch.dict(
            "batch_doc_vqa.openrouter.api.MODEL_CONFIG_OVERRIDES",
            {"unit/test-model": {"top_k": 20}},
            clear=False,
        ):
            resolved = resolve_model_config_overrides("unit/test-model")
            resolved["top_k"] = 999
            fresh = resolve_model_config_overrides("unit/test-model")
        self.assertEqual(fresh.get("top_k"), 20)


if __name__ == "__main__":
    unittest.main()
