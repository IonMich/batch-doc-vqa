import json
import tempfile
import unittest
from pathlib import Path

from batch_doc_vqa.openrouter.spec import load_extraction_spec


class TestOpenRouterSpec(unittest.TestCase):
    def test_default_spec_uses_default_preset(self):
        spec = load_extraction_spec()
        self.assertEqual(spec.mode, "default_student")
        self.assertEqual(spec.preset_id, "default_student")
        self.assertEqual(spec.default_pages, (1, 3))
        self.assertIsNone(spec.prompt_source)
        self.assertIsNone(spec.schema_source)
        self.assertFalse(spec.strict_schema_default)
        self.assertIn("properties", spec.schema)

    def test_prompt_file_switches_to_custom_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            prompt_path.write_text("Extract custom entities as JSON.", encoding="utf-8")

            spec = load_extraction_spec(prompt_file=str(prompt_path))
            self.assertEqual(spec.mode, "custom")
            self.assertEqual(spec.preset_id, "default_student")
            self.assertEqual(spec.prompt_source, str(prompt_path.resolve(strict=False)))
            self.assertFalse(spec.strict_schema_default)

    def test_explicit_default_preset_id(self):
        spec = load_extraction_spec(preset_id="default_student")
        self.assertEqual(spec.preset_id, "default_student")
        self.assertEqual(spec.mode, "default_student")
        self.assertEqual(spec.default_pages, (1, 3))

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            load_extraction_spec(preset_id="unknown_preset")

    def test_schema_file_must_be_top_level_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            schema_path = Path(tmp_dir) / "schema.json"
            schema_path.write_text(json.dumps({"type": "array"}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_extraction_spec(schema_file=str(schema_path))

    def test_custom_schema_loads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            schema_path = Path(tmp_dir) / "schema.json"
            schema_path.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "required": ["entities"],
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "additionalProperties": False,
                    }
                ),
                encoding="utf-8",
            )

            spec = load_extraction_spec(schema_file=str(schema_path))
            self.assertEqual(spec.mode, "custom")
            self.assertEqual(spec.schema_source, str(schema_path.resolve(strict=False)))
            self.assertTrue(spec.strict_schema_default)
            self.assertIn("entities", spec.schema.get("properties", {}))


if __name__ == "__main__":
    unittest.main()
