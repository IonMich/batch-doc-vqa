"""Sanitized, versioned benchmark-run summaries for publication.

Raw inference runs intentionally remain machine-local.  This module exports the
small, auditable subset needed to regenerate the public benchmark tables and
plots without copying model responses, identifiers, absolute paths, or API
metadata into Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PUBLISHED_RUN_SCHEMA_VERSION = 2
PUBLISHED_ARCHIVE_SCHEMA_VERSION = 2
DEFAULT_PUBLISHED_ROOT = Path("benchmarks/published/q11")
DEFAULT_PUBLISHED_RUNS_DIR = DEFAULT_PUBLISHED_ROOT / "runs"
DEFAULT_ARCHIVE_MANIFEST = DEFAULT_PUBLISHED_ROOT / "archive.json"
DEFAULT_DATASET_SOURCE = Path("imgs/quiz11-presidents.pdf")

_ABSOLUTE_PATH_RE = re.compile(
    r"(?:file://[^\s\"']+|(?:^|[\s\"'=(])/(?!/)[^\s\"'<>]+|"
    r"(?:^|[\s\"'=(])[A-Za-z]:[\\/][^\s\"'<>]+|\\\\[^\s\"']+\\[^\s\"']+)"
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\b(?:sk-(?:proj-)?|gh[pousr]_|github_pat_|xox[baprs]-|AKIA|ASIA|AIza)[A-Za-z0-9_\-]{8,}|"
    r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})"
)

_SAFE_MODEL_KEYS = ("org", "model", "variant", "model_size", "open_weights", "license_info")
_SAFE_API_KEYS = (
    "provider",
    "temperature",
    "max_tokens",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
)
_SAFE_FEATURE_KEYS = ("structured_output", "regex_patterns")
_SAFE_ENVIRONMENT_KEYS = ("runtime",)
_SAFE_REPRODUCIBILITY_KEYS = (
    "manifest_version",
    "git_commit",
    "prompt_hash",
    "inference_settings_hash",
    "parser_version",
    "schema_version",
)
_SAFE_ADDITIONAL_KEYS = (
    "actual_model_providers",
    "concurrency",
    "cost_fetch_max_workers",
    "dataset_manifest_sha256",
    "endpoint_type",
    "extraction_mode",
    "generation_param_sources",
    "generation_params_effective",
    "model_context_length",
    "model_name",
    "model_pricing",
    "pages",
    "preset_id",
    "provider_routing_effective",
    "provider_routing_requested",
    "rate_limit",
    "response_format",
    "retry_base_delay",
    "retry_max",
    "schema_hash",
    "schema_retry_max",
    "strict_schema",
    "token_escalation_repetition_threshold",
)
_PUBLISHED_STATS_KEYS = (
    "digit_top1",
    "id_top1",
    "lastname_top1",
    "id_avg_lev",
    "lastname_avg_lev",
    "docs_detected",
    "docs_detected_count",
    "expected_docs_count",
    "runtime_seconds",
    "fully_parallelizable_runtime_available",
    "fully_parallelizable_runtime_seconds",
    "timed_images",
    "total_images",
    "cost_per_image",
    "total_cost",
    "observed_total_cost",
    "cost_status",
    "cost_complete",
    "total_requests",
    "costed_requests",
    "precise_cost_requests",
    "estimated_cost_requests",
    "missing_cost_requests",
    "zero_cost_precise_requests",
)
_PREDICTION_FIELDS = ("student_full_name", "university_id", "ufid", "section_number")
_FAILURE_MARKERS = (
    "_schema_failed",
    "_parse_failed",
    "_no_response",
    "_empty_response",
    "_server_error",
    "_api_error",
    "_exception",
    "_retry_failed",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    """Return a SHA-256 hash of a canonical JSON representation."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    """Return a content hash without exposing the source path in the result."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset_provenance(
    doc_info_file: str | Path,
    test_ids_file: str | Path,
    *,
    dataset_id: str = "q11",
    dataset_source_file: str | Path | None = None,
) -> Dict[str, Any]:
    """Create a stable logical dataset identity from the scoring inputs.

    Generated image filenames are deliberately excluded from the identity.
    They contain random suffixes and differ between otherwise equivalent
    renders on different machines.  The canonical document/page structure,
    scoring targets, and optional source artifact identify the dataset; the
    exact local manifest is still used separately when validating raw results.
    """
    doc_info_path = Path(doc_info_file)
    test_ids_path = Path(test_ids_file)
    with doc_info_path.open(newline="", encoding="utf-8") as source:
        doc_rows = list(csv.DictReader(source))
    with test_ids_path.open(newline="", encoding="utf-8") as source:
        test_rows = list(csv.DictReader(source))

    canonical_manifest: List[Dict[str, Any]] = sorted(
        (
            {
                "doc": str(row.get("doc", "")).strip(),
                "page": int(str(row.get("page", "")).strip()),
            }
            for row in doc_rows
            if str(row.get("doc", "")).strip()
            and str(row.get("page", "")).strip().isdigit()
        ),
        key=_canonical_json,
    )
    if len(canonical_manifest) != len(doc_rows):
        raise ValueError("doc_info.csv must contain non-empty doc and integer page values")

    docs = {str(row.get("doc", "")).strip() for row in test_rows if str(row.get("doc", "")).strip()}
    pages = sorted(
        {row["page"] for row in canonical_manifest}
    )
    page_row_counts = Counter(row["page"] for row in canonical_manifest)
    source_hash = file_hash(dataset_source_file) if dataset_source_file is not None else None
    provenance = {
        "identity_version": 2,
        "dataset_id": dataset_id,
        "manifest_content_sha256": stable_hash(canonical_manifest),
        "dataset_source_sha256": source_hash,
        "test_ids_sha256": file_hash(test_ids_path),
        "expected_docs": len(docs) or len(test_rows),
        "manifest_rows": len(doc_rows),
        "pages": pages,
        "page_row_counts": {str(page): page_row_counts[page] for page in pages},
    }
    provenance["content_hash"] = stable_hash(dataset_identity_inputs(provenance))
    return provenance


def dataset_identity_inputs(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the fields that define logical dataset equivalence."""
    return {
        "identity_version": dataset.get("identity_version"),
        "manifest_content_sha256": dataset.get("manifest_content_sha256"),
        "dataset_source_sha256": dataset.get("dataset_source_sha256"),
        "test_ids_sha256": dataset.get("test_ids_sha256"),
        "expected_docs": dataset.get("expected_docs"),
        "manifest_rows": dataset.get("manifest_rows"),
        "pages": dataset.get("pages"),
        "page_row_counts": dataset.get("page_row_counts"),
    }


def _safe_subset(source: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def _contains_forbidden_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _SENSITIVE_KEY_RE.search(str(key)) or _contains_forbidden_value(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_value(child) for child in value)
    return isinstance(value, str) and bool(
        _ABSOLUTE_PATH_RE.search(value) or _SECRET_VALUE_RE.search(value)
    )


def _contains_non_finite_number(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_non_finite_number(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite_number(child) for child in value)
    return isinstance(value, float) and not math.isfinite(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sanitize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only safe, reproducibility-relevant configuration fields."""
    run_info = config.get("run_info", {}) if isinstance(config.get("run_info"), Mapping) else {}
    reproducibility = run_info.get("reproducibility", {}) if isinstance(run_info.get("reproducibility"), Mapping) else {}
    additional = config.get("additional", {}) if isinstance(config.get("additional"), Mapping) else {}

    sanitized = {
        "run_info": {
            "run_name": run_info.get("run_name"),
            "timestamp": run_info.get("timestamp"),
            "timestamp_iso": run_info.get("timestamp_iso"),
            "reproducibility": _safe_subset(reproducibility, _SAFE_REPRODUCIBILITY_KEYS),
        },
        "model": _safe_subset(config.get("model", {}), _SAFE_MODEL_KEYS),
        "api": _safe_subset(config.get("api", {}), _SAFE_API_KEYS),
        "features": _safe_subset(config.get("features", {}), _SAFE_FEATURE_KEYS),
        "environment": _safe_subset(config.get("environment", {}), _SAFE_ENVIRONMENT_KEYS),
        "additional": _safe_subset(additional, _SAFE_ADDITIONAL_KEYS),
    }
    sanitized = _json_safe(sanitized)
    if _contains_forbidden_value(sanitized):
        raise ValueError("sanitized configuration still contains a secret-like key or absolute local path")
    return sanitized


def build_aggregation_inputs(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    request_scope: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return all fields that must match before runs are aggregated."""
    safe = sanitize_config(config)
    reproducibility = safe["run_info"].get("reproducibility", {})
    additional = safe.get("additional", {})
    scope_pages = (
        request_scope.get("pages")
        if isinstance(request_scope, Mapping)
        else additional.get("pages")
    )
    return {
        "version": 2,
        "dataset_content_hash": dataset.get("content_hash"),
        "request_pages": scope_pages,
        "model": safe.get("model", {}),
        "api": safe.get("api", {}),
        "features": safe.get("features", {}),
        "generation_params": additional.get("generation_params_effective", {}),
        "generation_param_sources": additional.get("generation_param_sources", {}),
        "schema_hash": additional.get("schema_hash"),
        "strict_schema": additional.get("strict_schema"),
        "extraction_mode": additional.get("extraction_mode"),
        "response_format": additional.get("response_format"),
        "routing": {
            "requested": additional.get("provider_routing_requested", {}),
            "effective": additional.get("provider_routing_effective", {}),
            "actual_model_providers": additional.get("actual_model_providers", []),
        },
        "runtime_settings": {
            key: additional.get(key)
            for key in ("concurrency", "rate_limit", "retry_base_delay", "retry_max", "schema_retry_max")
            if key in additional
        },
        "reproducibility": {
            key: reproducibility.get(key)
            for key in ("git_commit", "prompt_hash", "inference_settings_hash", "parser_version", "schema_version")
            if key in reproducibility
        },
    }


def build_aggregation_fingerprint(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    request_scope: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the explicit, content-addressed compatibility fingerprint."""
    return stable_hash(build_aggregation_inputs(config, dataset, request_scope=request_scope))


def build_sanitized_request_evidence(raw_results: Mapping[str, Any]) -> Dict[str, Any]:
    """Aggregate publication evidence without retaining per-document records."""
    request_count = len(raw_results)
    invalid_entries = 0
    failure_counts: Counter[str] = Counter()
    field_presence: Counter[str] = Counter()
    elapsed_values: List[float] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0
    token_usage_requests = 0
    precise_cost_requests = 0
    zero_cost_precise_requests = 0
    observed_precise_cost_total = 0.0

    for source_key in sorted(raw_results):
        entries = raw_results.get(source_key)
        if not (isinstance(entries, list) and entries and isinstance(entries[0], Mapping)):
            invalid_entries += 1
            continue
        entry = entries[0]
        token_usage = entry.get("_token_usage", {}) if isinstance(entry.get("_token_usage"), Mapping) else {}
        timing = entry.get("_timing", {}) if isinstance(entry.get("_timing"), Mapping) else {}
        for field in _PREDICTION_FIELDS:
            if field in entry and entry.get(field) not in (None, ""):
                field_presence[field] += 1
        for marker in _FAILURE_MARKERS:
            if entry.get(marker) not in (None, False, 0, ""):
                failure_counts[marker] += 1

        elapsed = timing.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0:
            elapsed_values.append(float(elapsed))

        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")
        total_tokens = token_usage.get("total_tokens")
        if (
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and prompt_tokens >= 0
            and isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
            and completion_tokens >= 0
            and isinstance(total_tokens, int)
            and not isinstance(total_tokens, bool)
            and total_tokens >= 0
        ):
            token_usage_requests += 1
            prompt_tokens_total += prompt_tokens
            completion_tokens_total += completion_tokens
            total_tokens_total += total_tokens

        actual_cost = token_usage.get("actual_cost")
        if (
            isinstance(actual_cost, (int, float))
            and not isinstance(actual_cost, bool)
            and math.isfinite(float(actual_cost))
            and actual_cost >= 0
        ):
            precise_cost_requests += 1
            observed_precise_cost_total += float(actual_cost)
            if actual_cost == 0:
                zero_cost_precise_requests += 1

    return {
        "schema_version": 2,
        "request_count": request_count,
        "invalid_entries": invalid_entries,
        "failure_count": sum(failure_counts.values()),
        "failure_markers": dict(sorted(failure_counts.items())),
        "response_field_presence": {
            field: field_presence[field] for field in _PREDICTION_FIELDS
        },
        "timing": {
            "timed_requests": len(elapsed_values),
            "max_elapsed_seconds": max(elapsed_values) if elapsed_values else None,
        },
        "token_usage": {
            "complete_requests": token_usage_requests,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens_total": completion_tokens_total,
            "total_tokens_total": total_tokens_total,
        },
        "cost": {
            "precise_requests": precise_cost_requests,
            "zero_cost_precise_requests": zero_cost_precise_requests,
            "observed_precise_cost_total": observed_precise_cost_total,
        },
    }


def _compact_legacy_request_evidence(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert schema-v1 per-request evidence to the aggregate schema."""
    if evidence.get("schema_version") == 2:
        return _json_safe(dict(evidence))
    records = evidence.get("records")
    if not isinstance(records, list):
        raise ValueError("legacy request evidence is missing records")

    failure_counts: Counter[str] = Counter()
    field_presence: Counter[str] = Counter()
    elapsed_values: List[float] = []
    token_usage_requests = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0
    precise_cost_requests = 0
    zero_cost_precise_requests = 0
    observed_precise_cost_total = 0.0
    invalid_entries = 0

    for record in records:
        if not isinstance(record, Mapping):
            invalid_entries += 1
            continue
        for marker in record.get("failure_markers", []):
            if isinstance(marker, str):
                failure_counts[marker] += 1
        response_fields = record.get("response_fields", {})
        if isinstance(response_fields, Mapping):
            for field in _PREDICTION_FIELDS:
                details = response_fields.get(field, {})
                if isinstance(details, Mapping) and details.get("present") is True:
                    field_presence[field] += 1
        timing = record.get("timing", {})
        elapsed = timing.get("elapsed_seconds") if isinstance(timing, Mapping) else None
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0:
            elapsed_values.append(float(elapsed))
        token_usage = record.get("token_usage", {})
        if isinstance(token_usage, Mapping):
            prompt_tokens = token_usage.get("prompt_tokens")
            completion_tokens = token_usage.get("completion_tokens")
            total_tokens = token_usage.get("total_tokens")
            if (
                isinstance(prompt_tokens, int)
                and not isinstance(prompt_tokens, bool)
                and prompt_tokens >= 0
                and isinstance(completion_tokens, int)
                and not isinstance(completion_tokens, bool)
                and completion_tokens >= 0
                and isinstance(total_tokens, int)
                and not isinstance(total_tokens, bool)
                and total_tokens >= 0
            ):
                token_usage_requests += 1
                prompt_tokens_total += prompt_tokens
                completion_tokens_total += completion_tokens
                total_tokens_total += total_tokens
        cost = record.get("cost", {})
        actual_cost = cost.get("actual_cost") if isinstance(cost, Mapping) else None
        if (
            isinstance(actual_cost, (int, float))
            and not isinstance(actual_cost, bool)
            and math.isfinite(float(actual_cost))
            and actual_cost >= 0
        ):
            precise_cost_requests += 1
            observed_precise_cost_total += float(actual_cost)
            if actual_cost == 0:
                zero_cost_precise_requests += 1

    return {
        "schema_version": 2,
        "request_count": len(records),
        "invalid_entries": invalid_entries,
        "failure_count": sum(failure_counts.values()),
        "failure_markers": dict(sorted(failure_counts.items())),
        "response_field_presence": {
            field: field_presence[field] for field in _PREDICTION_FIELDS
        },
        "timing": {
            "timed_requests": len(elapsed_values),
            "max_elapsed_seconds": max(elapsed_values) if elapsed_values else None,
        },
        "token_usage": {
            "complete_requests": token_usage_requests,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens_total": completion_tokens_total,
            "total_tokens_total": total_tokens_total,
        },
        "cost": {
            "precise_requests": precise_cost_requests,
            "zero_cost_precise_requests": zero_cost_precise_requests,
            "observed_precise_cost_total": observed_precise_cost_total,
        },
    }


def _expected_requests_for_pages(dataset: Mapping[str, Any], pages: Sequence[int]) -> int:
    page_row_counts = dataset.get("page_row_counts")
    if not isinstance(page_row_counts, Mapping):
        raise ValueError("dataset provenance is missing page row counts")
    try:
        return sum(int(page_row_counts[str(page)]) for page in pages)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("request scope contains pages outside the dataset manifest") from exc


def _request_scope_from_config_and_stats(
    config: Mapping[str, Any],
    stats: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> Dict[str, Any]:
    additional = config.get("additional", {}) if isinstance(config.get("additional"), Mapping) else {}
    pages = additional.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("request scope is required when run configuration does not record pages")
    normalized_pages = sorted({int(page) for page in pages})
    return {
        "pages": normalized_pages,
        "expected_requests": _expected_requests_for_pages(dataset, normalized_pages),
        "observed_requests": stats.get("total_requests"),
    }


def make_published_run_summary(
    run_info: Mapping[str, Any],
    stats: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    request_evidence: Mapping[str, Any],
    request_scope: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a redacted published-run summary from one local raw run."""
    config = run_info.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("run config is missing or invalid")
    safe_config = sanitize_config(config)
    run_name = str(run_info.get("run_name") or safe_config["run_info"].get("run_name") or "")
    if not run_name:
        raise ValueError("run name is required")
    published_stats = {key: _json_safe(stats[key]) for key in _PUBLISHED_STATS_KEYS if key in stats}
    scope = _json_safe(
        dict(request_scope)
        if isinstance(request_scope, Mapping)
        else _request_scope_from_config_and_stats(config, published_stats, dataset)
    )
    aggregation_inputs = build_aggregation_inputs(config, dataset, request_scope=scope)
    summary = {
        "schema_version": PUBLISHED_RUN_SCHEMA_VERSION,
        "run_name": run_name,
        "dataset": _json_safe(dict(dataset)),
        "config": safe_config,
        "aggregation_fingerprint": stable_hash(aggregation_inputs),
        "aggregation_inputs": aggregation_inputs,
        "request_scope": scope,
        "request_evidence": _json_safe(dict(request_evidence)),
        "stats": published_stats,
    }
    validate_published_summary(summary)
    return summary


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _contains_key(value: Any, forbidden_key: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden_key in value or any(
            _contains_key(child, forbidden_key) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, forbidden_key) for child in value)
    return False


def _validate_dataset(dataset: Mapping[str, Any]) -> None:
    if dataset.get("identity_version") != 2:
        raise ValueError("published dataset identity version is unsupported")
    for key in ("manifest_content_sha256", "test_ids_sha256", "content_hash"):
        value = dataset.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"published dataset is missing valid {key}")
    source_hash = dataset.get("dataset_source_sha256")
    if source_hash is not None and (
        not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
    ):
        raise ValueError("published dataset has invalid dataset_source_sha256")
    _nonnegative_int(dataset.get("expected_docs"), "dataset expected_docs")
    manifest_rows = _nonnegative_int(dataset.get("manifest_rows"), "dataset manifest_rows")
    pages = dataset.get("pages")
    if (
        not isinstance(pages, list)
        or not pages
        or any(not isinstance(page, int) or isinstance(page, bool) for page in pages)
        or pages != sorted(set(pages))
    ):
        raise ValueError("published dataset pages must be a sorted unique integer list")
    page_row_counts = dataset.get("page_row_counts")
    if not isinstance(page_row_counts, Mapping):
        raise ValueError("published dataset is missing page_row_counts")
    expected_count_keys = {str(page) for page in pages}
    if set(page_row_counts) != expected_count_keys:
        raise ValueError("published dataset page_row_counts do not match pages")
    row_total = sum(
        _nonnegative_int(page_row_counts[str(page)], f"page {page} row count")
        for page in pages
    )
    if row_total != manifest_rows:
        raise ValueError("published dataset page row counts do not match manifest_rows")
    if dataset.get("content_hash") != stable_hash(dataset_identity_inputs(dataset)):
        raise ValueError("published dataset content hash does not match its identity inputs")


def _validate_request_evidence(evidence: Mapping[str, Any], request_count: int) -> None:
    if evidence.get("schema_version") != 2:
        raise ValueError("published request evidence schema version is unsupported")
    if _nonnegative_int(evidence.get("request_count"), "evidence request_count") != request_count:
        raise ValueError("published request evidence count does not match request scope")
    if _nonnegative_int(evidence.get("invalid_entries"), "evidence invalid_entries") != 0:
        raise ValueError("published request evidence contains invalid result entries")
    failure_count = _nonnegative_int(evidence.get("failure_count"), "evidence failure_count")
    failure_markers = evidence.get("failure_markers")
    if not isinstance(failure_markers, Mapping):
        raise ValueError("published request evidence is missing failure marker counts")
    if sum(_nonnegative_int(count, f"failure marker {marker}") for marker, count in failure_markers.items()) != failure_count:
        raise ValueError("published request failure counts are inconsistent")
    if failure_count != 0:
        raise ValueError("published run contains inference failure markers")

    field_presence = evidence.get("response_field_presence")
    if not isinstance(field_presence, Mapping) or set(field_presence) != set(_PREDICTION_FIELDS):
        raise ValueError("published request evidence has invalid response-field counts")
    for field in _PREDICTION_FIELDS:
        count = _nonnegative_int(field_presence[field], f"response field {field}")
        if count > request_count:
            raise ValueError("published response-field count exceeds request count")

    timing = evidence.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("published request evidence is missing timing summary")
    timed_requests = _nonnegative_int(timing.get("timed_requests"), "timed requests")
    if timed_requests > request_count:
        raise ValueError("published timed request count exceeds request count")
    max_elapsed = timing.get("max_elapsed_seconds")
    if timed_requests == 0:
        if max_elapsed is not None:
            raise ValueError("published timing maximum exists without timed requests")
    else:
        _nonnegative_number(max_elapsed, "maximum elapsed time")

    token_usage = evidence.get("token_usage")
    if not isinstance(token_usage, Mapping):
        raise ValueError("published request evidence is missing token summary")
    complete_token_requests = _nonnegative_int(token_usage.get("complete_requests"), "complete token requests")
    if complete_token_requests > request_count:
        raise ValueError("published token request count exceeds request count")
    for key in ("prompt_tokens_total", "completion_tokens_total", "total_tokens_total"):
        _nonnegative_int(token_usage.get(key), key)

    cost = evidence.get("cost")
    if not isinstance(cost, Mapping):
        raise ValueError("published request evidence is missing cost summary")
    precise_requests = _nonnegative_int(cost.get("precise_requests"), "evidence precise requests")
    zero_requests = _nonnegative_int(cost.get("zero_cost_precise_requests"), "evidence zero-cost requests")
    if zero_requests > precise_requests or precise_requests > request_count:
        raise ValueError("published precise-cost evidence counts are inconsistent")
    _nonnegative_number(cost.get("observed_precise_cost_total"), "observed precise cost")


def _validate_cost_stats(stats: Mapping[str, Any], request_count: int) -> None:
    total_requests = _nonnegative_int(stats.get("total_requests"), "stats total_requests")
    if total_requests != request_count:
        raise ValueError("published stats request count does not match request scope")
    costed = _nonnegative_int(stats.get("costed_requests"), "stats costed_requests")
    precise = _nonnegative_int(stats.get("precise_cost_requests"), "stats precise_cost_requests")
    estimated = _nonnegative_int(stats.get("estimated_cost_requests"), "stats estimated_cost_requests")
    missing = _nonnegative_int(stats.get("missing_cost_requests"), "stats missing_cost_requests")
    zero_precise = _nonnegative_int(stats.get("zero_cost_precise_requests"), "stats zero_cost_precise_requests")
    if precise + estimated != costed or costed + missing != total_requests:
        raise ValueError("published cost request counts are inconsistent")
    if zero_precise > precise:
        raise ValueError("published zero-cost request count exceeds precise request count")

    complete = stats.get("cost_complete")
    if not isinstance(complete, bool) or complete != (total_requests > 0 and missing == 0):
        raise ValueError("published cost completeness is inconsistent with request counts")
    total_cost = stats.get("total_cost")
    cost_per_image = stats.get("cost_per_image")
    observed_total = stats.get("observed_total_cost")
    if complete:
        total = _nonnegative_number(total_cost, "published total cost")
        per_image = _nonnegative_number(cost_per_image, "published cost per image")
        observed = _nonnegative_number(observed_total, "published observed total cost")
        if not math.isclose(per_image, total / total_requests, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("published cost per image does not match total cost")
        if not math.isclose(observed, total, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("published observed and complete total costs differ")
    elif total_cost is not None or cost_per_image is not None:
        raise ValueError("partial or unavailable published costs must not expose a complete total")
    elif costed == 0:
        if observed_total is not None:
            raise ValueError("unavailable cost must not expose an observed subtotal")
    else:
        _nonnegative_number(observed_total, "published observed subtotal")

    if costed == 0:
        expected_status = "unavailable"
    elif not complete:
        expected_status = "partial"
    elif precise == total_requests and _nonnegative_number(total_cost, "published total cost") == 0:
        expected_status = "verified-zero"
    elif precise == total_requests:
        expected_status = "precise"
    else:
        expected_status = "estimated"
    if stats.get("cost_status") != expected_status:
        raise ValueError("published cost status is inconsistent with cost provenance")


def validate_published_summary(summary: Mapping[str, Any]) -> None:
    """Reject malformed, internally inconsistent, or unsafe summaries."""
    if summary.get("schema_version") != PUBLISHED_RUN_SCHEMA_VERSION:
        raise ValueError("unsupported published run schema version")
    run_name = summary.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("published run summary is missing run_name")

    dataset = summary.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("published run summary is missing dataset provenance")
    _validate_dataset(dataset)

    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("published run summary is missing config")
    if sanitize_config(config) != config:
        raise ValueError("published run config contains fields outside the sanitized schema")
    config_run_name = config.get("run_info", {}).get("run_name")
    if config_run_name not in (None, run_name):
        raise ValueError("published run name does not match sanitized config")

    stats = summary.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError("published run summary is missing stats")
    scope = summary.get("request_scope")
    if not isinstance(scope, Mapping):
        raise ValueError("published run summary is missing request scope")
    pages = scope.get("pages")
    if (
        not isinstance(pages, list)
        or not pages
        or any(not isinstance(page, int) or isinstance(page, bool) for page in pages)
        or pages != sorted(set(pages))
    ):
        raise ValueError("published request pages must be a sorted unique integer list")
    expected_requests = _expected_requests_for_pages(dataset, pages)
    if _nonnegative_int(scope.get("expected_requests"), "scope expected_requests") != expected_requests:
        raise ValueError("published request scope does not cover the complete selected pages")
    observed_requests = _nonnegative_int(scope.get("observed_requests"), "scope observed_requests")
    if observed_requests != expected_requests:
        raise ValueError("published run does not contain exactly the expected requests")
    config_pages = config.get("additional", {}).get("pages")
    if isinstance(config_pages, list) and config_pages:
        if sorted({int(page) for page in config_pages}) != pages:
            raise ValueError("published request scope differs from configured pages")

    evidence = summary.get("request_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("published run summary is missing request evidence")
    _validate_request_evidence(evidence, observed_requests)
    _validate_cost_stats(stats, observed_requests)
    if evidence["cost"]["precise_requests"] != stats["precise_cost_requests"]:
        raise ValueError("published precise-cost evidence does not match stats")
    if evidence["cost"]["zero_cost_precise_requests"] != stats["zero_cost_precise_requests"]:
        raise ValueError("published zero-cost evidence does not match stats")
    if "total_images" in stats and _nonnegative_int(stats["total_images"], "stats total_images") != observed_requests:
        raise ValueError("published total image count does not match request scope")
    if "expected_docs_count" in stats and stats["expected_docs_count"] != dataset["expected_docs"]:
        raise ValueError("published expected-doc count does not match dataset")

    expected_inputs = build_aggregation_inputs(config, dataset, request_scope=scope)
    if summary.get("aggregation_inputs") != expected_inputs:
        raise ValueError("published aggregation inputs do not match config, dataset, and request scope")
    expected_fingerprint = stable_hash(expected_inputs)
    if summary.get("aggregation_fingerprint") != expected_fingerprint:
        raise ValueError("published aggregation fingerprint does not match aggregation inputs")

    if _contains_forbidden_value(summary):
        raise ValueError("published run summary contains a secret-like value or absolute local path")
    if _contains_key(summary, "document_ref"):
        raise ValueError("published run summary contains a reversible per-document reference")
    if _contains_non_finite_number(summary):
        raise ValueError("published run summary contains a non-finite number")


def published_summary_path(published_runs_dir: str | Path, run_name: str) -> Path:
    """Return a safe deterministic output filename for a published run."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip(".-")
    if not safe_name:
        raise ValueError("run name cannot produce an empty published filename")
    return Path(published_runs_dir) / f"{safe_name}.json"


def write_published_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    validate_published_summary(summary)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_published_summaries(published_runs_dir: str | Path) -> List[Dict[str, Any]]:
    """Load validated published summaries in deterministic order."""
    root = Path(published_runs_dir)
    if not root.exists():
        return []
    summaries: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_published_summary(payload)
        summaries.append(payload)
    return summaries


def summary_to_run_info(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Adapt a published summary to the table generator's run-info shape."""
    validate_published_summary(summary)
    config = dict(summary["config"])
    run_info = dict(config.get("run_info", {}))
    reproducibility = dict(run_info.get("reproducibility", {}))
    reproducibility["aggregation_fingerprint"] = summary["aggregation_fingerprint"]
    run_info["reproducibility"] = reproducibility
    config["run_info"] = run_info
    return {
        "run_name": summary["run_name"],
        "config": config,
        "has_results": True,
        "has_table_results": True,
        "has_manifest": True,
        "published_summary": dict(summary),
        "dataset_content_hash": summary["dataset"]["content_hash"],
        "request_scope": dict(summary["request_scope"]),
    }


def build_archive_manifest(summaries: Sequence[Mapping[str, Any]], dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Build an explicit completeness marker for a published archive."""
    _validate_dataset(dataset)
    run_entries = []
    seen_run_names = set()
    for summary in sorted(summaries, key=lambda item: str(item["run_name"])):
        validate_published_summary(summary)
        if summary["dataset"] != dataset:
            raise ValueError("cannot finalize an archive containing different dataset provenance")
        if summary["run_name"] in seen_run_names:
            raise ValueError(f"cannot finalize duplicate run name: {summary['run_name']}")
        seen_run_names.add(summary["run_name"])
        run_entries.append(
            {
                "run_name": summary["run_name"],
                "aggregation_fingerprint": summary["aggregation_fingerprint"],
                "summary_sha256": stable_hash(summary),
            }
        )
    manifest = {
        "schema_version": PUBLISHED_ARCHIVE_SCHEMA_VERSION,
        "publication_status": "complete",
        "dataset": _json_safe(dict(dataset)),
        "run_count": len(run_entries),
        "runs": run_entries,
    }
    manifest["archive_sha256"] = stable_hash(manifest)
    return manifest


def load_complete_archive_manifest(
    published_runs_dir: str | Path = DEFAULT_PUBLISHED_RUNS_DIR,
    archive_manifest: str | Path = DEFAULT_ARCHIVE_MANIFEST,
) -> Dict[str, Any]:
    """Load a finalized archive and verify every summary and manifest hash."""
    manifest_path = Path(archive_manifest)
    if not manifest_path.exists():
        raise ValueError(f"published archive manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summaries = load_published_summaries(published_runs_dir)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("published archive could not be read") from exc
    if manifest.get("schema_version") != PUBLISHED_ARCHIVE_SCHEMA_VERSION:
        raise ValueError("published archive schema version is unsupported")
    if manifest.get("publication_status") != "complete":
        raise ValueError("published archive has not been finalized")
    try:
        expected = build_archive_manifest(summaries, manifest["dataset"])
    except KeyError as exc:
        raise ValueError("published archive is missing dataset provenance") from exc
    if manifest != expected:
        raise ValueError("published archive manifest is stale or internally inconsistent")
    return manifest


def is_complete_published_archive(
    published_runs_dir: str | Path = DEFAULT_PUBLISHED_RUNS_DIR,
    archive_manifest: str | Path = DEFAULT_ARCHIVE_MANIFEST,
) -> bool:
    """Return whether the archive has been explicitly finalized and validates."""
    try:
        load_complete_archive_manifest(published_runs_dir, archive_manifest)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def migrate_published_archive(
    *,
    published_runs_dir: str | Path,
    doc_info_file: str | Path,
    test_ids_file: str | Path,
    dataset_id: str,
    dataset_source_file: str | Path | None,
    default_request_pages: Sequence[int],
    check: bool = False,
) -> List[Path]:
    """Migrate existing sanitized summaries without accessing raw responses."""
    dataset = build_dataset_provenance(
        doc_info_file,
        test_ids_file,
        dataset_id=dataset_id,
        dataset_source_file=dataset_source_file,
    )
    root = Path(published_runs_dir)
    source_paths = sorted(root.glob("*.json"))
    if not source_paths:
        raise RuntimeError("No published summaries were available to migrate")
    default_pages = sorted({int(page) for page in default_request_pages})
    if not default_pages:
        raise ValueError("default migration request pages cannot be empty")

    migrated: List[Dict[str, Any]] = []
    output_paths: List[Path] = []
    failures: List[str] = []
    for source_path in source_paths:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        config = payload.get("config")
        stats = payload.get("stats")
        evidence = payload.get("request_evidence")
        run_name = payload.get("run_name")
        if not isinstance(config, Mapping) or not isinstance(stats, Mapping) or not isinstance(evidence, Mapping):
            raise ValueError(f"cannot migrate malformed published summary: {source_path}")
        config_pages = config.get("additional", {}).get("pages")
        pages = (
            sorted({int(page) for page in config_pages})
            if isinstance(config_pages, list) and config_pages
            else default_pages
        )
        compact_evidence = _compact_legacy_request_evidence(evidence)
        scope = {
            "pages": pages,
            "expected_requests": _expected_requests_for_pages(dataset, pages),
            "observed_requests": compact_evidence["request_count"],
        }
        summary = make_published_run_summary(
            {"run_name": run_name, "config": config},
            stats,
            dataset,
            request_evidence=compact_evidence,
            request_scope=scope,
        )
        destination = published_summary_path(root, summary["run_name"])
        if destination != source_path:
            raise ValueError(f"published filename is not canonical: {source_path}")
        serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if check:
            if source_path.read_text(encoding="utf-8") != serialized:
                failures.append(f"{summary['run_name']}: migrated summary is stale")
        else:
            write_published_summary(source_path, summary)
        migrated.append(summary)
        output_paths.append(source_path)

    manifest = build_archive_manifest(migrated, dataset)
    manifest_path = root.parent / "archive.json"
    serialized_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if check:
        if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != serialized_manifest:
            failures.append("archive manifest is stale")
    else:
        manifest_path.write_text(serialized_manifest, encoding="utf-8")
    output_paths.append(manifest_path)
    if failures:
        raise RuntimeError("Published archive migration check failed:\n" + "\n".join(failures))
    return output_paths


def _matching_runs(generator: Any, patterns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    return generator._collect_matching_runs(list(patterns) if patterns else None)


def publish_local_runs(
    *,
    runs_dir: str | Path,
    published_runs_dir: str | Path,
    doc_info_file: str | Path,
    test_ids_file: str | Path,
    dataset_id: str,
    dataset_source_file: str | Path | None = None,
    patterns: Optional[Sequence[str]] = None,
    check: bool = False,
    finalize: bool = False,
    strict: bool = True,
) -> List[Path]:
    """Validate local raw runs and export their sanitized public summaries."""
    # Imported lazily to avoid an import cycle: table_generator also consumes
    # the published summaries.
    from .table_generator import BenchmarkTableGenerator

    generator = BenchmarkTableGenerator(
        str(runs_dir),
        interactive=False,
        source="local",
        # Publication treats raw run folders as an immutable local archive.
        # Recompute in memory, but never refresh table_results.json or metadata.
        cache_results=False,
        write_metadata=False,
    )
    dataset = build_dataset_provenance(
        doc_info_file,
        test_ids_file,
        dataset_id=dataset_id,
        dataset_source_file=dataset_source_file,
    )
    output_paths: List[Path] = []
    summaries: List[Dict[str, Any]] = []
    skipped: List[str] = []
    failures: List[str] = []
    for run_info in _matching_runs(generator, patterns):
        dataset_ok, reason = generator._matches_requested_dataset(
            run_info,
            doc_info_file=str(doc_info_file),
            test_ids_file=str(test_ids_file),
            dataset_provenance=dataset,
            validate_result_paths=True,
        )
        if not dataset_ok:
            skipped.append(f"{run_info.get('run_name', 'unknown')}: {reason or 'dataset mismatch'}")
            continue
        eligible, reason = generator._is_run_eligible_for_cohort(run_info)
        if not eligible:
            skipped.append(f"{run_info.get('run_name', 'unknown')}: {reason or 'ineligible'}")
            continue
        try:
            stats = generator.compute_run_stats(
                run_info,
                str(doc_info_file),
                str(test_ids_file),
                dataset_provenance=dataset,
            )
        except Exception as exc:
            skipped.append(f"{run_info.get('run_name', 'unknown')}: unable to compute stats ({exc})")
            continue
        if not stats:
            skipped.append(f"{run_info.get('run_name', 'unknown')}: unable to compute stats")
            continue
        raw_results = generator.run_manager.load_results(run_info["run_name"])
        summary = make_published_run_summary(
            run_info,
            stats,
            dataset,
            request_evidence=build_sanitized_request_evidence(raw_results),
            request_scope=run_info.get("_request_scope"),
        )
        destination = published_summary_path(published_runs_dir, summary["run_name"])
        serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if check:
            if not destination.exists() or destination.read_text(encoding="utf-8") != serialized:
                failures.append(f"{summary['run_name']}: published summary is missing or stale")
        else:
            write_published_summary(destination, summary)
        output_paths.append(destination)
        summaries.append(summary)

    if strict and skipped:
        failures.extend(skipped)
    if failures:
        raise RuntimeError("Publish validation failed:\n" + "\n".join(failures))
    if not summaries:
        details = "\n".join(skipped)
        raise RuntimeError(f"No eligible runs were available to publish{(':\n' + details) if details else ''}")

    if skipped:
        print(f"Skipped {len(skipped)} run(s) that did not meet publication validation.")
        for detail in skipped:
            print(f"  - {detail}")

    if finalize:
        all_summaries = load_published_summaries(published_runs_dir)
        manifest = build_archive_manifest(all_summaries, dataset)
        manifest_path = Path(published_runs_dir).parent / "archive.json"
        serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if check:
            if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != serialized:
                raise RuntimeError("Publish validation failed:\narchive manifest is missing or stale")
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(serialized, encoding="utf-8")
        output_paths.append(manifest_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish sanitized benchmark summaries from local raw runs")
    parser.add_argument("--runs-dir", default="tests/output/runs", help="Machine-local raw-run directory")
    parser.add_argument("--published-runs-dir", default=str(DEFAULT_PUBLISHED_RUNS_DIR), help="Tracked published summary directory")
    parser.add_argument("--doc-info", default="imgs/q11/doc_info.csv", help="Scoring document manifest")
    parser.add_argument("--test-ids", default="tests/data/test_ids.csv", help="Scoring ground-truth CSV")
    parser.add_argument(
        "--dataset-source",
        default=str(DEFAULT_DATASET_SOURCE),
        help="Stable source artifact used in the logical dataset identity",
    )
    parser.add_argument("--dataset-id", default="q11", help="Stable public dataset identifier")
    parser.add_argument("--patterns", nargs="*", help="Optional run-name regex filters")
    parser.add_argument("--check", action="store_true", help="Validate that the published archive is current without writing")
    parser.add_argument("--finalize", action="store_true", help="Write or verify the complete archive manifest")
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail when any selected run is excluded from publication (default)",
    )
    strict_group.add_argument(
        "--allow-skips",
        dest="strict",
        action="store_false",
        help="Allow explicitly selected invalid runs to be skipped",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--migrate-existing",
        action="store_true",
        help="Migrate the existing sanitized archive to the current schema without raw runs",
    )
    parser.add_argument(
        "--migration-default-pages",
        nargs="+",
        type=int,
        default=[1, 3],
        help="Request pages for legacy summaries that did not record them",
    )
    args = parser.parse_args()
    if args.migrate_existing:
        paths = migrate_published_archive(
            published_runs_dir=args.published_runs_dir,
            doc_info_file=args.doc_info,
            test_ids_file=args.test_ids,
            dataset_id=args.dataset_id,
            dataset_source_file=args.dataset_source,
            default_request_pages=args.migration_default_pages,
            check=args.check,
        )
        mode = "Validated migration of" if args.check else "Migrated"
    else:
        paths = publish_local_runs(
            runs_dir=args.runs_dir,
            published_runs_dir=args.published_runs_dir,
            doc_info_file=args.doc_info,
            test_ids_file=args.test_ids,
            dataset_id=args.dataset_id,
            dataset_source_file=args.dataset_source,
            patterns=args.patterns,
            check=args.check,
            finalize=args.finalize,
            strict=args.strict,
        )
        mode = "Validated" if args.check else "Published"
    print(f"{mode} {len(paths)} artifact(s).")


if __name__ == "__main__":
    main()
