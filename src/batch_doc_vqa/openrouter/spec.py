#!/usr/bin/env python3
"""
Extraction specification loading for OpenRouter inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Optional
import hashlib
import json

from .presets import DEFAULT_PRESET_ID, resolve_preset_definition


@dataclass(frozen=True)
class ExtractionSpec:
    """Resolved prompt + schema configuration for an inference run."""

    preset_id: str
    default_pages: tuple[int, ...]
    prompt_text: str
    schema: dict[str, Any]
    mode: str
    prompt_source: Optional[str]
    schema_source: Optional[str]
    prompt_hash: str
    schema_hash: str
    strict_schema_default: bool


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_path(raw_path: str) -> str:
    return str(Path(raw_path).expanduser().resolve(strict=False))


def _read_prompt_file(prompt_file: str) -> tuple[str, str]:
    normalized_path = _normalize_path(prompt_file)
    prompt_text = Path(normalized_path).read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise ValueError(f"Prompt file is empty: {normalized_path}")
    return prompt_text, normalized_path


def _read_schema_file(schema_file: str) -> tuple[dict[str, Any], str]:
    normalized_path = _normalize_path(schema_file)
    raw_text = Path(normalized_path).read_text(encoding="utf-8")
    try:
        schema_obj = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Schema file is not valid JSON: {normalized_path}") from exc

    if not isinstance(schema_obj, dict):
        raise ValueError(f"Schema JSON must be an object: {normalized_path}")

    schema_type = schema_obj.get("type")
    if schema_type is not None and schema_type != "object":
        raise ValueError(
            "Top-level schema type must be 'object' for current inference output handling: "
            f"{normalized_path}"
        )

    return schema_obj, normalized_path


def load_extraction_spec(
    *,
    preset_id: str = DEFAULT_PRESET_ID,
    prompt_file: Optional[str] = None,
    schema_file: Optional[str] = None,
) -> ExtractionSpec:
    """Load prompt and schema settings for inference."""

    preset = resolve_preset_definition(preset_id)

    if prompt_file:
        prompt_text, prompt_source = _read_prompt_file(prompt_file)
    else:
        prompt_text = preset.prompt_text
        prompt_source = None

    if schema_file:
        schema_obj, schema_source = _read_schema_file(schema_file)
    else:
        schema_obj = copy.deepcopy(preset.schema)
        schema_source = None

    mode = preset.mode if (prompt_source is None and schema_source is None) else "custom"

    return ExtractionSpec(
        preset_id=preset.preset_id,
        default_pages=tuple(preset.default_pages),
        prompt_text=prompt_text,
        schema=schema_obj,
        mode=mode,
        prompt_source=prompt_source,
        schema_source=schema_source,
        prompt_hash=_hash_text(prompt_text),
        schema_hash=_hash_json(schema_obj),
        strict_schema_default=(schema_source is not None),
    )
