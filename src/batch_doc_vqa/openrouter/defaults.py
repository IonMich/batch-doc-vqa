#!/usr/bin/env python3
"""
Centralized defaults for OpenRouter extraction workflows.
"""
from __future__ import annotations

from .presets import DEFAULT_PRESET_ID, resolve_preset_definition


DEFAULT_IMAGES_DIR = "imgs/q11"
DEFAULT_DATASET_MANIFEST_FILE = None
DEFAULT_EXTRACTION_PRESET_ID = DEFAULT_PRESET_ID
DEFAULT_PAGES: tuple[int, ...] = resolve_preset_definition(DEFAULT_EXTRACTION_PRESET_ID).default_pages
