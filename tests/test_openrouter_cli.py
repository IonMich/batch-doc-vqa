#!/usr/bin/env python3
"""Tests for OpenRouter CLI argument parsing and wiring."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from batch_doc_vqa.openrouter import cli


class TestOpenRouterCliHelpers(unittest.TestCase):
    def test_parse_provider_order_normalizes_and_deduplicates(self):
        parsed = cli.parse_provider_order(" DeepInfra,crusoe,deepinfra , ,Crusoe ")
        self.assertEqual(parsed, ["deepinfra", "crusoe"])

    def test_parse_pages_uses_defaults_when_not_provided(self):
        self.assertEqual(cli.parse_pages(None, default_selection=[1, 3]), [1, 3])

    def test_parse_pages_rejects_non_numeric_values(self):
        with self.assertRaises(ValueError):
            cli.parse_pages("1,two,3", default_selection=[1, 3])

    def test_parse_pages_rejects_all_invalid_values(self):
        with self.assertRaises(ValueError):
            cli.parse_pages("0,-1,0", default_selection=[1, 3])


class TestOpenRouterCliMain(unittest.TestCase):
    @patch("batch_doc_vqa.openrouter.cli.setup_api_key", return_value=None)
    @patch("batch_doc_vqa.openrouter.cli.resolve_preset_definition")
    @patch("batch_doc_vqa.openrouter.inference.run_openrouter_inference", return_value="run_123")
    def test_main_wires_doc_info_alias_and_routing_flags(
        self,
        mock_run_inference,
        mock_resolve_preset,
        _mock_setup_api_key,
    ):
        mock_resolve_preset.return_value = SimpleNamespace(
            preset_id="default_student",
            default_pages=(1, 3),
        )
        argv = [
            "openrouter-inference",
            "--model",
            "qwen/qwen3-vl-8b-instruct",
            "--doc-info",
            "/tmp/my/doc_info.csv",
            "--images-dir",
            "/tmp/my/images",
            "--provider-order",
            "DeepInfra,deepinfra,crusoe",
            "--no-fallbacks",
            "--no-provider-zdr",
            "--pages",
            "1,3",
        ]

        with patch("sys.argv", argv):
            cli.main()

        _, kwargs = mock_run_inference.call_args
        self.assertEqual(kwargs["model_name"], "qwen/qwen3-vl-8b-instruct")
        self.assertEqual(kwargs["preset_id"], "default_student")
        self.assertEqual(kwargs["images_dir"], "/tmp/my/images")
        self.assertEqual(kwargs["dataset_manifest_file"], "/tmp/my/doc_info.csv")
        self.assertEqual(kwargs["provider_order"], ["deepinfra", "crusoe"])
        self.assertIs(kwargs["provider_allow_fallbacks"], False)
        self.assertIs(kwargs["provider_zdr"], False)
        self.assertEqual(kwargs["pages"], [1, 3])

    @patch("batch_doc_vqa.openrouter.cli.setup_api_key", return_value=None)
    @patch("batch_doc_vqa.openrouter.cli.resolve_preset_definition", side_effect=ValueError("Unknown extraction preset"))
    @patch("batch_doc_vqa.openrouter.inference.run_openrouter_inference")
    def test_main_exits_on_invalid_preset(
        self,
        mock_run_inference,
        _mock_resolve_preset,
        _mock_setup_api_key,
    ):
        argv = [
            "openrouter-inference",
            "--model",
            "qwen/qwen3-vl-8b-instruct",
            "--preset",
            "does_not_exist",
        ]

        with patch("sys.argv", argv):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()

        self.assertEqual(ctx.exception.code, 2)
        mock_run_inference.assert_not_called()

    @patch("batch_doc_vqa.openrouter.cli.setup_api_key", return_value=None)
    @patch("batch_doc_vqa.openrouter.cli.resolve_preset_definition")
    @patch("batch_doc_vqa.openrouter.ui.interactive_provider_model_selection", return_value="google/gemini-2.5-flash")
    @patch("batch_doc_vqa.openrouter.inference.run_openrouter_inference", return_value="run_456")
    def test_main_uses_interactive_model_when_model_is_omitted(
        self,
        mock_run_inference,
        _mock_interactive_select,
        mock_resolve_preset,
        _mock_setup_api_key,
    ):
        mock_resolve_preset.return_value = SimpleNamespace(
            preset_id="default_student",
            default_pages=(1, 3),
        )
        argv = ["openrouter-inference"]

        with patch("sys.argv", argv):
            cli.main()

        _, kwargs = mock_run_inference.call_args
        self.assertEqual(kwargs["model_name"], "google/gemini-2.5-flash")
        self.assertIs(kwargs["confirm_reproducibility_warnings"], True)


if __name__ == "__main__":
    unittest.main()
