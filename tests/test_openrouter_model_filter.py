#!/usr/bin/env python3
"""Tests for OpenRouter vision-model filtering helpers."""

import unittest

from batch_doc_vqa.openrouter.api import filter_vision_models, model_supports_image_input


class TestOpenRouterModelFilter(unittest.TestCase):
    def test_rejects_text_only_even_if_description_mentions_multimodal(self):
        model = {
            "id": "google/gemma-3n-e4b-it",
            "description": "Supports multimodal inputs including visual and audio data",
            "architecture": {
                "modality": "text->text",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
        }
        self.assertFalse(model_supports_image_input(model))

    def test_accepts_image_in_input_modalities(self):
        model = {
            "id": "qwen/qwen2.5-vl-7b",
            "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
        }
        self.assertTrue(model_supports_image_input(model))

    def test_accepts_image_in_modality_string_fallback(self):
        model = {
            "id": "some/provider-vision-model",
            "architecture": {
                "modality": "image+text->text",
            },
        }
        self.assertTrue(model_supports_image_input(model))

    def test_filter_vision_models_uses_architecture_not_description(self):
        text_only_marketing = {
            "id": "text-only",
            "description": "Great multimodal visual assistant",
            "architecture": {
                "modality": "text->text",
                "input_modalities": ["text"],
            },
        }
        true_vision = {
            "id": "vision-model",
            "description": "Vision model",
            "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
            },
        }

        filtered = filter_vision_models([text_only_marketing, true_vision])
        ids = [m.get("id") for m in filtered]
        self.assertEqual(ids, ["vision-model"])


if __name__ == "__main__":
    unittest.main()
