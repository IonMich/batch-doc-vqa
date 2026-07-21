#!/usr/bin/env python3
"""Regression tests for run-manager reproducibility metadata."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from batch_doc_vqa.core.run_manager import RunConfig


def _fake_git(args):
    if args == ["rev-parse", "HEAD"]:
        return "deadbeef"
    if args == ["status", "--porcelain"]:
        return ""
    return ""


class TestRunManagerReproducibility(unittest.TestCase):
    @patch("batch_doc_vqa.core.run_manager._safe_run_git_command", side_effect=_fake_git)
    def test_provider_routing_config_is_derived_at_serialize_time(self, _mock_git):
        config = RunConfig(
            org="google",
            model="gemma-3-4b-it",
            additional_config={
                "provider_routing_effective": {"zdr": True, "data_collection": "deny"},
                "actual_model_providers": set(),
            },
        )

        # Simulate providers discovered during inference after config init.
        config.additional_config["actual_model_providers"].add("DeepInfra")

        serialized = config.to_dict()
        repro = serialized["run_info"]["reproducibility"]["provider_routing_config"]

        self.assertEqual(repro.get("actual_model_providers"), ["DeepInfra"])
        self.assertEqual(
            repro.get("provider_routing_effective"),
            {"zdr": True, "data_collection": "deny"},
        )

    @patch("batch_doc_vqa.core.run_manager._safe_run_git_command", side_effect=_fake_git)
    def test_records_dataset_manifest_content_hash_without_replacing_path(self, _mock_git):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "doc_info.csv"
            manifest.write_text("doc,page,filename\ndoc1,1,a.png\n", encoding="utf-8")
            config = RunConfig(
                org="google",
                model="gemma-3-4b-it",
                additional_config={"dataset_manifest_file": str(manifest)},
            )
            serialized = config.to_dict()
            additional = serialized["additional"]
            self.assertEqual(additional["dataset_manifest_file"], str(manifest))
            self.assertEqual(len(additional["dataset_manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
