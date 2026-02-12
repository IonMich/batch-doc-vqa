#!/usr/bin/env python3
"""
Extraction adapters isolate schema/prompt specifics from inference orchestration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from .spec import ExtractionSpec
from .presets import build_preset_adapter


class ExtractionAdapter(Protocol):
    mode: str

    def base_result_entry(self) -> Dict[str, Any]:
        ...

    def normalize_output(self, parsed_obj: Any) -> tuple[Optional[Dict[str, Any]], list[str]]:
        ...

    def coerce_invalid_output(self, parsed_obj: Optional[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], list[str]]:
        ...

    def build_schema_retry_prompt(
        self,
        previous_output: Any,
        schema_errors: list[str],
        *,
        attempt_number: int,
    ) -> str:
        ...

    def format_success_status(self, parsed_obj: Dict[str, Any], *, schema_coerced: bool = False) -> str:
        ...


@dataclass(frozen=True)
class GenericSchemaExtractionAdapter:
    spec: ExtractionSpec
    schema_validator: Optional[Any]

    @property
    def mode(self) -> str:
        return self.spec.mode

    def base_result_entry(self) -> Dict[str, Any]:
        return {}

    def normalize_output(self, parsed_obj: Any) -> tuple[Optional[Dict[str, Any]], list[str]]:
        if not isinstance(parsed_obj, dict):
            return None, ["Top-level JSON must be an object."]

        normalized: Dict[str, Any] = dict(parsed_obj)
        if self.schema_validator is None:
            return normalized, []

        schema_errors = []
        for err in self.schema_validator.iter_errors(normalized):
            path_tokens = [str(token) for token in err.absolute_path]
            path = ".".join(path_tokens) if path_tokens else "<root>"
            schema_errors.append(f"{path}: {err.message}")
            if len(schema_errors) >= 8:
                break
        return normalized, schema_errors

    def coerce_invalid_output(self, parsed_obj: Optional[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], list[str]]:
        return None, []

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
        schema_text = json.dumps(self.spec.schema, ensure_ascii=False, indent=2)
        return (
            "You previously returned invalid structured output for this same image.\n\n"
            f"Retry attempt #{attempt_number}.\n\n"
            f"Previous invalid output:\n{previous_output_json}\n\n"
            f"Validation issues:\n{issues}\n\n"
            "Return ONLY valid JSON matching this JSON Schema:\n"
            f"{schema_text}\n\n"
            "Do not include markdown, code fences, or explanations."
        )

    def format_success_status(self, parsed_obj: Dict[str, Any], *, schema_coerced: bool = False) -> str:
        if schema_coerced:
            return "✓ structured output (schema coerced)"
        return "✓ structured output"


def build_extraction_adapter(
    *,
    spec: ExtractionSpec,
    schema_validator: Optional[Any] = None,
) -> ExtractionAdapter:
    if spec.mode == "custom":
        return GenericSchemaExtractionAdapter(spec=spec, schema_validator=schema_validator)
    return build_preset_adapter(
        preset_id=spec.preset_id,
        spec=spec,
        schema_validator=schema_validator,
    )
