"""Constants shared across TA benchmark tooling."""

from __future__ import annotations

from pathlib import Path


TA_SCHEMA_VERSION = "ta_benchmark_v1"

DEFAULT_TA_WORKSPACE_ROOT = "/tmp/instructor_pilot_ta_v1"

DEFAULT_TA_SCHEMA_PATH = str(
    Path("docs/examples/schemas/ta_benchmark_v1.schema.json").resolve(strict=False)
)
DEFAULT_TA_TAXONOMY_PATH = str(
    Path("docs/examples/schemas/ta_error_taxonomy_v1.json").resolve(strict=False)
)
DEFAULT_TA_RUBRIC_SCHEMA_PATH = str(
    Path("docs/examples/schemas/ta_rubric_v1.schema.json").resolve(strict=False)
)

PII_WARNING_BANNER = (
    "PII WARNING: This workflow may contain real student identifiers. "
    "Keep artifacts local and avoid committing raw datasets."
)
