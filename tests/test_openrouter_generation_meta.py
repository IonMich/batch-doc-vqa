#!/usr/bin/env python3
"""Tests for OpenRouter generation metadata enrichment and ID handling."""

import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from batch_doc_vqa.openrouter.api import (
    _extract_generation_meta,
    _fetch_generation_stats_with_retries,
    batch_update_generation_costs,
)


class TestGenerationMetaExtraction(unittest.TestCase):
    def test_extract_generation_meta_keeps_safe_subset(self):
        payload = {
            "provider": {
                "name": "DeepInfra",
                "slug": "deepinfra",
                "id": "provider-internal-id",
            },
            "model": "google/gemma-3-27b-it",
            "status": "succeeded",
            "finish_reason": "stop",
            "native_finish_reason": "stop",
            "latency_ms": 1432,
            "processing_time_ms": 1200,
            "created_at": "2026-02-12T00:01:23Z",
            "prompt": "sensitive prompt text",
            "response_text": "sensitive output text",
            "total_cost": 0.000071,
        }

        meta = _extract_generation_meta(payload)
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta.get("provider"), "deepinfra")
        self.assertEqual(meta.get("provider_name"), "DeepInfra")
        self.assertEqual(meta.get("provider_slug"), "deepinfra")
        self.assertEqual(meta.get("model"), "google/gemma-3-27b-it")
        self.assertEqual(meta.get("finish_reason"), "stop")
        self.assertEqual(meta.get("native_finish_reason"), "stop")
        self.assertEqual(meta.get("latency_ms"), 1432)
        self.assertEqual(meta.get("processing_time_ms"), 1200)
        self.assertEqual(meta.get("created_at"), "2026-02-12T00:01:23Z")
        self.assertNotIn("id", meta)
        self.assertNotIn("prompt", meta)
        self.assertNotIn("response_text", meta)
        self.assertNotIn("generation_id", meta)

    def test_extract_generation_meta_optionally_keeps_generation_id(self):
        meta = _extract_generation_meta(
            {"provider": "deepinfra"},
            generation_id="gen-abc123",
            include_generation_id=True,
        )
        self.assertEqual(meta.get("generation_id"), "gen-abc123")


class TestBatchUpdateGenerationCosts(unittest.TestCase):
    def _base_results(self):
        return {
            "imgs/q11/doc-0-page-1-VHX7P2D0.png": [
                {
                    "student_full_name": "Harry S. Truman",
                    "university_id": "11800",
                    "_token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                        "generation_id": "gen-xyz",
                    },
                }
            ]
        }

    @patch("batch_doc_vqa.openrouter.api.time.sleep", return_value=None)
    @patch("batch_doc_vqa.openrouter.api._fetch_generation_stats_with_retries")
    def test_success_default_strips_generation_id_and_adds_meta(self, mock_fetch, _mock_sleep):
        mock_fetch.return_value = {
            "success": True,
            "attempts": 1,
            "status_code": 200,
            "retryable": False,
            "error": None,
            "data": {
                "provider": "deepinfra",
                "model": "google/gemma-3-4b-it",
                "finish_reason": "stop",
                "native_finish_reason": "stop",
                "native_tokens_prompt": 222,
                "native_tokens_completion": 12,
                "total_cost": 0.000021,
            },
        }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            updated = batch_update_generation_costs(deepcopy(self._base_results()), max_workers=1)

        entry = updated["imgs/q11/doc-0-page-1-VHX7P2D0.png"][0]
        token_usage = entry["_token_usage"]
        self.assertNotIn("generation_id", token_usage)
        self.assertEqual(token_usage.get("actual_cost"), 0.000021)
        self.assertEqual(token_usage.get("prompt_tokens"), 222)
        self.assertEqual(token_usage.get("completion_tokens"), 12)
        self.assertIn("_generation_meta", entry)
        self.assertEqual(entry["_generation_meta"].get("provider"), "deepinfra")
        self.assertNotIn("generation_id", entry["_generation_meta"])

    @patch("batch_doc_vqa.openrouter.api.time.sleep", return_value=None)
    @patch("batch_doc_vqa.openrouter.api._fetch_generation_stats_with_retries")
    def test_failure_default_strips_generation_id(self, mock_fetch, _mock_sleep):
        mock_fetch.return_value = {
            "success": False,
            "attempts": 3,
            "status_code": 429,
            "retryable": True,
            "error": "HTTP 429",
            "data": None,
        }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            updated = batch_update_generation_costs(deepcopy(self._base_results()), max_workers=1)

        entry = updated["imgs/q11/doc-0-page-1-VHX7P2D0.png"][0]
        self.assertNotIn("generation_id", entry["_token_usage"])
        self.assertIn("_cost_fetch", entry)
        self.assertNotIn("generation_id", entry["_cost_fetch"])

    @patch("batch_doc_vqa.openrouter.api.time.sleep", return_value=None)
    @patch("batch_doc_vqa.openrouter.api._fetch_generation_stats_with_retries")
    def test_keep_generation_id_opt_in(self, mock_fetch, _mock_sleep):
        mock_fetch.return_value = {
            "success": False,
            "attempts": 2,
            "status_code": 500,
            "retryable": True,
            "error": "HTTP 500",
            "data": None,
        }

        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_KEEP_GENERATION_ID": "1"},
            clear=False,
        ):
            updated = batch_update_generation_costs(deepcopy(self._base_results()), max_workers=1)

        entry = updated["imgs/q11/doc-0-page-1-VHX7P2D0.png"][0]
        self.assertEqual(entry["_token_usage"].get("generation_id"), "gen-xyz")
        self.assertEqual(entry["_cost_fetch"].get("generation_id"), "gen-xyz")


class _MockGenerationResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class TestFetchGenerationStatsWithRetries(unittest.TestCase):
    @patch("batch_doc_vqa.openrouter.api.time.sleep", return_value=None)
    @patch("batch_doc_vqa.openrouter.api.requests.get")
    def test_retries_until_required_fields_are_available(self, mock_get, _mock_sleep):
        mock_get.side_effect = [
            _MockGenerationResponse(
                200,
                {"data": {"model": "qwen/qwen3.5-27b", "native_tokens_prompt": 100}},
            ),
            _MockGenerationResponse(
                200,
                {
                    "data": {
                        "model": "qwen/qwen3.5-27b",
                        "native_tokens_prompt": 100,
                        "native_tokens_completion": 20,
                        "total_cost": 0.00042,
                    }
                },
            ),
        ]

        outcome = _fetch_generation_stats_with_retries(
            "gen-test",
            "test-key",
            max_retries=2,
            base_delay_seconds=0.01,
            request_timeout_seconds=1.0,
        )

        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["attempts"], 2)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(outcome["data"].get("total_cost"), 0.00042)

    @patch("batch_doc_vqa.openrouter.api.time.sleep", return_value=None)
    @patch("batch_doc_vqa.openrouter.api.requests.get")
    def test_fails_when_required_fields_never_appear(self, mock_get, _mock_sleep):
        mock_get.side_effect = [
            _MockGenerationResponse(200, {"data": {"model": "qwen/qwen3.5-27b"}}),
            _MockGenerationResponse(200, {"data": {"model": "qwen/qwen3.5-27b"}}),
            _MockGenerationResponse(200, {"data": {"model": "qwen/qwen3.5-27b"}}),
        ]

        outcome = _fetch_generation_stats_with_retries(
            "gen-test",
            "test-key",
            max_retries=2,
            base_delay_seconds=0.01,
            request_timeout_seconds=1.0,
        )

        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["attempts"], 3)
        self.assertTrue(outcome["retryable"])
        self.assertIn("missing required fields", str(outcome.get("error")))


if __name__ == "__main__":
    unittest.main()
