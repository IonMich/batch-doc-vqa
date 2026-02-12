#!/usr/bin/env python3
"""
Preset registry for OpenRouter extraction workflows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from . import student


AdapterBuilder = Callable[..., Any]


@dataclass(frozen=True)
class PresetDefinition:
    preset_id: str
    mode: str
    default_pages: tuple[int, ...]
    prompt_text: str
    schema: dict[str, Any]
    adapter_builder: AdapterBuilder


DEFAULT_PRESET_ID = student.PRESET_ID

_PRESET_REGISTRY: dict[str, PresetDefinition] = {
    student.PRESET_ID: PresetDefinition(
        preset_id=student.PRESET_ID,
        mode=student.MODE,
        default_pages=student.DEFAULT_PAGES,
        prompt_text=student.PROMPT_TEXT,
        schema=student.SCHEMA,
        adapter_builder=student.build_adapter,
    ),
}


def available_preset_ids() -> list[str]:
    return sorted(_PRESET_REGISTRY.keys())


def resolve_preset_definition(preset_id: Optional[str] = None) -> PresetDefinition:
    candidate = (preset_id or DEFAULT_PRESET_ID).strip().lower()
    preset = _PRESET_REGISTRY.get(candidate)
    if preset is None:
        options = ", ".join(available_preset_ids()) or "<none>"
        raise ValueError(f"Unknown extraction preset '{candidate}'. Available: {options}")
    return preset


def build_preset_adapter(
    *,
    preset_id: str,
    spec: Any,
    schema_validator: Optional[Any] = None,
) -> Any:
    definition = resolve_preset_definition(preset_id)
    return definition.adapter_builder(spec=spec, schema_validator=schema_validator)
