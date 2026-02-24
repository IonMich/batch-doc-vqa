#!/usr/bin/env python3
"""Tests for OpenRouter model-specific override resolution."""

import unittest
from unittest.mock import patch

from batch_doc_vqa.openrouter.api import resolve_model_config_overrides


class TestOpenRouterModelOverrideResolution(unittest.TestCase):
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
