import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from batch_doc_vqa.openrouter.extraction_adapter import build_extraction_adapter
from batch_doc_vqa.openrouter.spec import load_extraction_spec


class TestDefaultStudentExtractionAdapter(unittest.TestCase):
    def setUp(self) -> None:
        spec = load_extraction_spec()
        self.adapter = build_extraction_adapter(spec=spec)

    def test_normalize_maps_ufid_and_enforces_digit_rules(self):
        normalized, errors = self.adapter.normalize_output(
            {
                "student_full_name": 123,
                "ufid": "12A34",
                "section_number": "123456",
            }
        )

        self.assertIsInstance(normalized, dict)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["student_full_name"], "123")
        self.assertEqual(normalized["university_id"], "12A34")
        self.assertEqual(normalized["section_number"], "123456")
        self.assertEqual(len(errors), 2)
        self.assertIn("university_id must contain only digits", errors[0])
        self.assertIn("section_number must contain only digits", errors[1])

    def test_coerce_invalid_output_clears_non_matching_numeric_fields(self):
        coerced, corrections = self.adapter.coerce_invalid_output(
            {
                "student_full_name": "Jane Student",
                "university_id": "123456789",
                "section_number": "A100",
            }
        )

        self.assertIsInstance(coerced, dict)
        self.assertIsNotNone(coerced)
        self.assertEqual(coerced["student_full_name"], "Jane Student")
        self.assertEqual(coerced["university_id"], "")
        self.assertEqual(coerced["section_number"], "")
        self.assertEqual(len(corrections), 2)

    def test_format_success_status_matches_default_student_output(self):
        status_with_id = self.adapter.format_success_status(
            {"student_full_name": "Jane Student", "university_id": "12345678"}
        )
        status_without_id = self.adapter.format_success_status(
            {"student_full_name": "Jane Student", "university_id": ""}
        )
        status_coerced = self.adapter.format_success_status(
            {"student_full_name": "Jane Student"},
            schema_coerced=True,
        )

        self.assertEqual(status_with_id, "✓ Jane Student (ID: 12345678)")
        self.assertEqual(status_without_id, "✓ Jane Student")
        self.assertEqual(status_coerced, "✓ Jane Student (schema coerced)")


class TestGenericSchemaExtractionAdapter(unittest.TestCase):
    def test_generic_adapter_uses_schema_validator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            schema_path = Path(tmp_dir) / "schema.json"
            prompt_path.write_text("Extract entities as JSON.", encoding="utf-8")
            schema = {
                "type": "object",
                "required": ["entities"],
                "properties": {
                    "entities": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            }
            schema_path.write_text(json.dumps(schema), encoding="utf-8")

            spec = load_extraction_spec(
                prompt_file=str(prompt_path),
                schema_file=str(schema_path),
            )
            validator = Draft202012Validator(spec.schema)
            adapter = build_extraction_adapter(spec=spec, schema_validator=validator)

            ok_payload, ok_errors = adapter.normalize_output({"entities": ["a", "b"]})
            bad_payload, bad_errors = adapter.normalize_output({"entities": "not-an-array"})

            self.assertEqual(ok_payload, {"entities": ["a", "b"]})
            self.assertEqual(ok_errors, [])
            self.assertEqual(bad_payload, {"entities": "not-an-array"})
            self.assertGreaterEqual(len(bad_errors), 1)
            self.assertIn("entities", bad_errors[0])


if __name__ == "__main__":
    unittest.main()
