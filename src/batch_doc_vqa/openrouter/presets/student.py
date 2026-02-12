#!/usr/bin/env python3
"""
Built-in preset for the default student benchmark extraction task.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from ...core.prompts import STUDENT_EXTRACTION_PROMPT

if TYPE_CHECKING:
    from ..spec import ExtractionSpec


PRESET_ID = "default_student"
MODE = "default_student"
DEFAULT_PAGES: tuple[int, ...] = (1, 3)
PROMPT_TEXT = STUDENT_EXTRACTION_PROMPT
SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["student_full_name", "university_id", "section_number"],
    "additionalProperties": True,
    "properties": {
        "student_full_name": {"type": "string"},
        "university_id": {"type": "string", "pattern": r"^$|^\d{1,8}$"},
        "section_number": {"type": "string", "pattern": r"^$|^\d{1,5}$"},
    },
}


@dataclass(frozen=True)
class StudentExtractionAdapter:
    spec: "ExtractionSpec"

    mode: str = MODE

    def base_result_entry(self) -> Dict[str, Any]:
        return {
            "student_full_name": "",
            "university_id": "",
            "section_number": "",
        }

    def normalize_output(self, parsed_obj: Any) -> tuple[Optional[Dict[str, Any]], list[str]]:
        if not isinstance(parsed_obj, dict):
            return None, ["Top-level JSON must be an object."]

        normalized: Dict[str, Any] = dict(parsed_obj)
        if "university_id" not in normalized and "ufid" in normalized:
            normalized["university_id"] = normalized.pop("ufid")

        for field in ("student_full_name", "university_id", "section_number"):
            value = normalized.get(field, "")
            if value is None:
                value = ""
            elif not isinstance(value, str):
                value = str(value)
            normalized[field] = value.strip()

        schema_errors: list[str] = []
        university_id = normalized.get("university_id", "")
        section_number = normalized.get("section_number", "")

        if university_id and not re.fullmatch(r"\d{1,8}", university_id):
            schema_errors.append(
                f'university_id must contain only digits (1-8 chars) or be empty; got "{university_id}"'
            )

        if section_number and not re.fullmatch(r"\d{1,5}", section_number):
            schema_errors.append(
                f'section_number must contain only digits (1-5 chars) or be empty; got "{section_number}"'
            )

        return normalized, schema_errors

    def coerce_invalid_output(self, parsed_obj: Optional[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], list[str]]:
        if not isinstance(parsed_obj, dict):
            return None, []

        coerced: Dict[str, Any] = dict(parsed_obj)
        corrections: list[str] = []

        for field in ("student_full_name", "university_id", "section_number"):
            value = coerced.get(field, "")
            if value is None:
                value = ""
            elif not isinstance(value, str):
                value = str(value)
            coerced[field] = value.strip()

        university_id = coerced.get("university_id", "")
        if university_id and not re.fullmatch(r"\d{1,8}", university_id):
            corrections.append(f'university_id "{university_id}" -> ""')
            coerced["university_id"] = ""

        section_number = coerced.get("section_number", "")
        if section_number and not re.fullmatch(r"\d{1,5}", section_number):
            corrections.append(f'section_number "{section_number}" -> ""')
            coerced["section_number"] = ""

        return coerced, corrections

    def build_schema_retry_prompt(
        self,
        previous_output: Any,
        schema_errors: list[str],
        *,
        attempt_number: int,
    ) -> str:
        try:
            previous_output_json = json.dumps(previous_output, ensure_ascii=False)
        except TypeError:
            previous_output_json = str(previous_output)

        issues = "\n".join(f"- {error}" for error in schema_errors) or "- Output did not satisfy schema."
        correction_hints: list[str] = []

        if isinstance(previous_output, dict):
            ufid_value = previous_output.get("ufid", previous_output.get("university_id", ""))
            if isinstance(ufid_value, str) and ufid_value:
                if ufid_value.isdigit() and len(ufid_value) > 8:
                    correction_hints.append(
                        f'- Your previous ufid "{ufid_value}" has {len(ufid_value)} digits; maximum allowed is 8.'
                    )
                elif not ufid_value.isdigit():
                    correction_hints.append(
                        f'- Your previous ufid "{ufid_value}" contains non-digit characters; use digits only.'
                    )

            section_value = previous_output.get("section_number", "")
            if isinstance(section_value, str) and section_value:
                if section_value.isdigit() and len(section_value) > 5:
                    correction_hints.append(
                        f'- Your previous section_number "{section_value}" has {len(section_value)} digits; maximum allowed is 5.'
                    )
                elif not section_value.isdigit():
                    correction_hints.append(
                        f'- Your previous section_number "{section_value}" contains non-digit characters; use digits only.'
                    )

        hint_block = ""
        if correction_hints:
            hint_block = "Specific corrections:\n" + "\n".join(correction_hints) + "\n\n"

        escalation_block = ""
        if attempt_number >= 2:
            escalation_block = (
                f"This is correction retry attempt #{attempt_number}. "
                "You repeated an invalid schema previously.\n"
                "Do not repeat the same invalid value.\n\n"
            )

        return (
            "You previously returned invalid structured output for this same image.\n\n"
            f"{escalation_block}"
            f"Previous invalid output:\n{previous_output_json}\n\n"
            f"Schema issues:\n{issues}\n\n"
            f"{hint_block}"
            "Return ONLY valid JSON in this exact format:\n"
            "{\n"
            '  "student_full_name": "Full name of the student",\n'
            '  "ufid": "8-digit UFID number if present, empty string if missing",\n'
            '  "section_number": "5-digit section number"\n'
            "}\n\n"
            "Rules:\n"
            "- ufid must be digits only (0-9) and 1-8 characters, or empty string.\n"
            "- section_number must be digits only (0-9) and 1-5 characters, or empty string.\n"
            "- If ufid is longer than 8 digits or uncertain, return empty string instead of invalid value.\n"
            "- If section_number is longer than 5 digits or uncertain, return empty string instead of invalid value.\n"
            "- Do not include markdown, code fences, or explanations."
        )

    def format_success_status(self, parsed_obj: Dict[str, Any], *, schema_coerced: bool = False) -> str:
        name = parsed_obj.get("student_full_name", "N/A")
        ufid = parsed_obj.get("university_id", "")
        if schema_coerced:
            return f"✓ {name} (schema coerced)"
        return f"✓ {name} (ID: {ufid})" if ufid else f"✓ {name}"


def build_adapter(
    *,
    spec: "ExtractionSpec",
    schema_validator: Optional[Any] = None,
) -> StudentExtractionAdapter:
    _ = schema_validator
    return StudentExtractionAdapter(spec=spec)
